import sys
from typing import List, Optional

import numpy as np

from docqa.data_processing.document_splitter import DocumentSplitter, ParagraphFilter, \
    DocParagraphWithAnswers
from docqa.data_processing.multi_paragraph_qa import DocumentParagraph, MultiParagraphQuestion
from docqa.data_processing.preprocessed_corpus import Preprocessor, FilteredData
from docqa.data_processing.qa_training_data import ParagraphAndQuestion, Answer
from docqa.data_processing.span_data import TokenSpans
from docqa.text_preprocessor import TextPreprocessor
from docqa.triviaqa.read_data import TriviaQaQuestion
from docqa.utils import flatten_iterable


from docqa.triviaqa.answer_detection import  FastNormalizedAnswerDetector
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from tqdm import tqdm
import pickle
import random
"""
Tools to convert pre-procossed TriviaQa questions and pre-tokenized docuemnts
into (question, paragraph) data we can train our model on. 
"""


class DocumentParagraphQuestion(ParagraphAndQuestion):
    def __init__(self, q_id: str, doc_id: str, para_range, question: List[str],
                 context: List[str], answer: Answer, rank=None):
        super().__init__(context, question, answer, q_id)
        self.doc_id = doc_id
        self.para_range = para_range
        self.rank = rank


class ExtractSingleParagraph(Preprocessor):
    """ Grab a single paragraph for each (document, question) pair, builds a list of
     (filtered) `DocumentParagraphQuestion` objects """

    def __init__(self, splitter: DocumentSplitter,
                 para_filter: Optional[ParagraphFilter],
                 text_preprocess: Optional[TextPreprocessor],
                 intern,
                 require_answer=True):
        self.splitter = splitter
        self.para_filter = para_filter
        self.text_preprocess = text_preprocess
        self.intern = intern
        self.require_answer = require_answer

    def preprocess(self, questions: List[TriviaQaQuestion], evidence) -> FilteredData:
        splitter = self.splitter
        paragraph_filter = self.para_filter
        output = []
        read_only = splitter.reads_first_n
        for q in questions:
            for doc in q.all_docs:
                text = evidence.get_document(doc.doc_id, n_tokens=read_only)
                if text is None:
                    raise ValueError(doc.doc_id, doc.doc_id)

                paragraphs = splitter.split_annotated(text, doc.answer_spans)
                if paragraph_filter is not None:
                    paragraphs = paragraph_filter.prune(q.question, paragraphs)
                if self.require_answer:
                    paragraphs = [x for x in paragraphs if len(x.answer_spans) > 0]
                if len(paragraphs) == 0:
                    continue
                paragraph = paragraphs[0]
                if self.text_preprocess is not None:
                    ex = self.text_preprocess.encode_extracted_paragraph(q.question, paragraph)
                    if not self.require_answer or len(ex.answer_spans) > 0:
                        output.append(DocumentParagraphQuestion(q.question_id, doc.doc_id, (paragraph.start, paragraph.end),
                                                                q.question, ex.text,
                                                                TokenSpans(q.answer.all_answers, ex.answer_spans), 1))
                else:
                    output.append(DocumentParagraphQuestion(q.question_id, doc.doc_id, (paragraph.start, paragraph.end),
                                                            q.question, flatten_iterable(paragraph.text),
                                                            TokenSpans(q.answer.all_answers, paragraph.answer_spans), 1))
        return FilteredData(output, sum(len(x.all_docs) for x in questions))

    def finalize_chunk(self, x: FilteredData):
        if self.intern:
            question_map = {}
            for q in x.data:
                q.question_id = sys.intern(q.question_id)
                if q.question_id in question_map:
                    q.question = question_map[q.question_id]
                else:
                    q.question = tuple(sys.intern(w) for w in q.question)
                    question_map[q.question_id] = q.question
                q.doc_id = sys.intern(q.doc_id)
                q.context = [sys.intern(w) for w in q.context]

    def __setstate__(self, state):
        if "state" in state:
            if "require_answer" not in state["state"]:
                state["state"]["require_answer"] = True
        super().__setstate__(state)


def intern_mutli_question(questions):
    for q in questions:
        q.question = [sys.intern(x) for x in q.question]
        for para in q.paragraphs:
            para.doc_id = sys.intern(para.doc_id)
            para.text = [sys.intern(x) for x in para.text]


class ExtractMultiParagraphs(Preprocessor):
    """
    Grab multiple paragraphs per (document, question) pair, return a list of (filtered) `MultiParagraphQuestion`
    """

    def __init__(self, splitter: DocumentSplitter, ranker: Optional[ParagraphFilter],
                 text_process: Optional[TextPreprocessor], intern: bool=False, require_an_answer=True):
        self.intern = intern
        self.splitter = splitter
        self.ranker = ranker
        self.text_process = text_process
        self.require_an_answer = require_an_answer

    def preprocess(self, questions: List[TriviaQaQuestion], evidence):
        true_len = 0
        splitter = self.splitter
        para_filter = self.ranker

        with_paragraphs = []
        for q in questions:
            true_len += len(q.all_docs)
            for doc in q.all_docs:
                if self.require_an_answer and len(doc.answer_spans) == 0:
                    continue
                text = evidence.get_document(doc.doc_id, splitter.reads_first_n)
                if text is None:
                    raise ValueError("No evidence text found document: " + doc.doc_id)
                if doc.answer_spans is not None:
                    paras = splitter.split_annotated(text, doc.answer_spans)
                else:
                    # this is kind of a hack to make the rest of the pipeline work, only
                    # needed for test cases
                    paras = splitter.split_annotated(text, np.zeros((0, 2), dtype=np.int32))

                if para_filter is not None:
                    paras = para_filter.prune(q.question, paras)

                if len(paras) == 0:
                    continue
                if self.require_an_answer:
                    if all(len(x.answer_spans) == 0 for x in paras):
                        continue
                if self.text_process is not None:
                    prepped = [self.text_process.encode_extracted_paragraph(q.question, p) for p in paras]
                    if self.require_an_answer:
                        if all(len(x.answer_spans) == 0 for x in prepped):
                            continue
                    doc_paras = []
                    for i, (preprocessed, para) in enumerate(zip(prepped, paras)):
                        doc_paras.append(DocumentParagraph(doc.doc_id, para.start, para.end,
                                                           i, preprocessed.answer_spans, preprocessed.text))

                else:
                    doc_paras = [DocumentParagraph(doc.doc_id, x.start, x.end,
                                                   i, x.answer_spans, flatten_iterable(x.text))
                                 for i, x in enumerate(paras)]
                with_paragraphs.append(MultiParagraphQuestion(q.question_id, q.question,
                                                              None if q.answer is None else q.answer.all_answers,
                                                              doc_paras))

        return FilteredData(with_paragraphs, true_len)

    def finalize_chunk(self, q: FilteredData):
        if self.intern:
            intern_mutli_question(q.data)


class ExtractMultiParagraphsPerQuestion(Preprocessor):
    """
    Get multiple paragraph per question, using all document. Returns a filtered list of
    `MultiParagraphQuestion`
    """

    def __init__(self, splitter: DocumentSplitter, ranker: ParagraphFilter,
                 text_preprocess: Optional[TextPreprocessor],
                 intern: bool=False, require_an_answer=True):
        self.intern = intern
        self.text_preprocess = text_preprocess
        self.splitter = splitter
        self.ranker = ranker
        self.require_an_answer = require_an_answer

    def preprocess(self, questions: List[TriviaQaQuestion], evidence) -> object:
        splitter = self.splitter
        para_filter = self.ranker

        with_paragraphs = []
        for q in questions:
            paras = []
            for doc in q.all_docs:
                if self.require_an_answer and len(doc.answer_spans) == 0:
                    continue
                text = evidence.get_document(doc.doc_id, splitter.reads_first_n)
                if doc.answer_spans is not None:
                    split = splitter.split_annotated(text, doc.answer_spans)
                else:
                    # this is kind of a hack to make the rest of the pipeline work, only
                    # needed for test cases
                    split = splitter.split_annotated(text, np.zeros((0, 2), dtype=np.int32))
                paras.extend([DocParagraphWithAnswers(x.text, x.start, x.end, x.answer_spans, doc.doc_id)
                              for x in split])

            if para_filter is not None:
                paras = para_filter.prune(q.question, paras)

            if len(paras) == 0:
                continue
            if self.require_an_answer:
                if all(len(x.answer_spans) == 0 for x in paras):
                    continue

            if self.text_preprocess is not None:
                prepped = [self.text_preprocess.encode_extracted_paragraph(q.question, p) for p in paras]
                if self.require_an_answer:
                    if all(len(x.answer_spans) == 0 for x in prepped):
                        continue
                doc_paras = []
                for i, (preprocessed, para) in enumerate(zip(prepped, paras)):
                    doc_paras.append(DocumentParagraph(para.doc_id, para.start, para.end,
                                                       i, preprocessed.answer_spans, preprocessed.text))
                with_paragraphs.append(
                    MultiParagraphQuestion(q.question_id, q.question,
                                           None if q.answer is None else q.answer.all_answers,
                                           doc_paras))
            else:
                doc_paras = [DocumentParagraph(x.doc_id, x.start, x.end,
                                               i, x.answer_spans, flatten_iterable(x.text))
                             for i, x in enumerate(paras)]
                with_paragraphs.append(MultiParagraphQuestion(q.question_id, q.question, q.answer.all_answers, doc_paras))

        return FilteredData(with_paragraphs, len(questions))

    def finalize_chunk(self, q: FilteredData):
        if self.intern:
            intern_mutli_question(q.data)

    #todo 3
    def init_train_data(self,data_mode,epoch):
        if data_mode =='train':
            flag_shuffle = 8 #0 for sort 1 for shuffle 2 for original 3 for sample 4 for avg_sample 5 for all_true 6 for neg sample 7 for avg&sample 8 avg->sample

            if flag_shuffle ==8:
                if epoch >=50:
                    flag_shuffle = 3
                else:
                    flag_shuffle = 4

            if flag_shuffle == 7:
                tmp = np.random.random()
                if tmp >= 0.5:
                    flag_shuffle = 3
                else:
                    flag_shuffle = 4
        else:
            flag_shuffle = 0
        with open('tmp_data/list_'+data_mode+'_p.pkl', 'rb') as fin:
            list_dev_p = pickle.load(fin)
        with open('tmp_data/list_'+data_mode+'_q.pkl', 'rb') as fin:
            list_dev_q = pickle.load(fin)
        with open('tmp_data/list_'+data_mode+'_q_id.pkl', 'rb') as fin:
            list_dev_q_id = pickle.load(fin)
        with open('tmp_data/list_'+data_mode+'_a.pkl', 'rb') as fin:
            list_dev_a = pickle.load(fin)
        with open('tmp_data/id2scores_'+data_mode+'.pkl', 'rb') as fin:
            id2scores_dev = pickle.load(fin)
        
        if flag_shuffle == 5:
            with open('tmp_data/dict_id2trueindex.pkl', 'rb') as fin:
                dict_id2trueindex =  pickle.load(fin)	

        tokenizer = NltkAndPunctTokenizer()
        word_tokenize = tokenizer.tokenize_paragraph_flat

        list_list_p_words = []
        batch_size = 30
        list_list_list_seg = []
        list_list_list_score = []

        for index, list_p in tqdm(enumerate(list_dev_p)):
            list_p_words = []
            list_list_seg = []
            list_list_score = []

            if index in id2scores_dev.keys():
                scores = id2scores_dev[index]
            else:
                scores = [1/len(list_p)  for _ in range(len(list_p))]
            
            if batch_size >len(scores):
                sample_size = len(scores)
            else:
                sample_size = batch_size

            if flag_shuffle == 3:
                list_sample_index = np.random.choice(a=len(scores), size=sample_size, replace=False, p=scores)
            if flag_shuffle == 4:
                tmp_scores =  [1/len(scores) for _ in range(len(scores))]
                list_sample_index = np.random.choice(a=len(scores), size=sample_size, replace=False, p= tmp_scores )
            if flag_shuffle == 6:
                scores = np.array(1)- scores
                sum = np.sum(scores)
                scores = scores/sum
                list_sample_index = np.random.choice(a=len(scores), size=sample_size, replace=False, p=scores)



            scores = scores[:len(list_p)]
            scores_pair = [(x, scores[x]) for x in range(len(scores))]
            scores_pair = sorted(scores_pair, key=lambda x: x[1], reverse=True)

            scores_pair = scores_pair[:batch_size]
            if flag_shuffle == 1:
                random.shuffle(scores_pair)
            elif flag_shuffle == 2:
                scores_pair = sorted(scores_pair, key=lambda x: x[0])
 
            
            sorted_index = [x[0] for x in scores_pair]
            sorted_scores = [x[1] for x in scores_pair]
            
            if flag_shuffle == 3 or flag_shuffle ==4 or flag_shuffle ==6 :
                sorted_index = [ x for x in list_sample_index if x < len(list_p) ]
                sorted_scores = [scores[x] for x in sorted_index ]

            if flag_shuffle == 5:
                trueindex = dict_id2trueindex[index]
                sorted_index = trueindex
                sorted_scores = [1/len(sorted_index) for _ in sorted_index]

            list_p_sorted = [list_p[x] for x in sorted_index]
            batch_len = (len(list_p_sorted) // batch_size) + 1

            for batch in range(batch_len):
                tmp_list_p = list_p_sorted[batch * batch_size:(batch + 1) * batch_size]
                list_score = sorted_scores[batch * batch_size:(batch + 1) * batch_size]

                if len(tmp_list_p) == 0:
                    break

                # tmp_p = tmp_list_p[0]
                # for p in tmp_list_p[1:]:
                #     tmp_p += ' '
                #     tmp_p += p
                # tmp_p_words_0 = tmp_p.split()

                tmp_p_words = []
                list_seg = []
                for p in tmp_list_p:
                    tmp_p_words += p.split()
                    list_seg.append(len(tmp_p_words))

                # assert (len(tmp_p_words_0) == len(tmp_p_words))

                list_p_words.append(tmp_p_words)
                list_list_seg.append(list_seg)
                list_list_score.append(list_score)
                break

            list_list_p_words.append(list_p_words)
            list_list_list_seg.append(list_list_seg)
            list_list_list_score.append(list_list_score)

        # data = []
        # dict_docid2seg = {}
        # dict_docid2score = {}

        with_paragraphs = []
        detector = FastNormalizedAnswerDetector()
        num = 0
        for q_id, q, list_a, list_p_words, list_list_seg, list_list_score in tqdm(
                zip(list_dev_q_id, list_dev_q, list_dev_a, list_list_p_words, list_list_list_seg,
                    list_list_list_score)):
            tokenized_aliases = [word_tokenize(x) for x in list_a]
            detector.set_question(tokenized_aliases)
            question_id = q_id
            question = word_tokenize(q)
            index = 0

            if len(list_p_words) == 0:
                continue

            p_words = list_p_words[0]
            answer_text = list_a
            answer_spans = []
            for s, e in detector.any_found([p_words]):
                answer_spans.append((s, e - 1))
            
            if len(answer_spans) == 0 and data_mode != 'train':
                answer_spans = [(0,0)]
            answer_spans = np.array(answer_spans, dtype=np.int32)

            if len(answer_spans) == 0:
                continue

            doc_id = str(question_id) + '_' + str(index)
            start = 0
            end = len(p_words)
            text = p_words

            # list_score = list_list_score[index]
            # dict_docid2score[doc_id] = list_score
            # list_seg = list_list_seg[index]
            # dict_docid2seg[doc_id] = list_seg
            doc_paras = []
            doc_paras.append(DocumentParagraph(doc_id, start, end, index, answer_spans, text))
            with_paragraphs.append(MultiParagraphQuestion(question_id, question, answer_text, doc_paras))
            num += 1
            # if num == 50:
            #     break
        # print(123)
        #print (len(with_paragraphs))
        return FilteredData(with_paragraphs, len(list_dev_q_id))
