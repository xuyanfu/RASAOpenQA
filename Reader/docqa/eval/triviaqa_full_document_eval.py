import argparse
import json
from os.path import join
from typing import List
import os 
import numpy as np
import pandas as pd
from tqdm import tqdm

from docqa import trainer
from docqa.config import TRIVIA_QA
from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf, ShallowOpenWebRanker, FirstN
from docqa.data_processing.preprocessed_corpus import preprocess_par
from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset
from docqa.data_processing.span_data import TokenSpans
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.dataset import FixedOrderBatcher
from docqa.eval.ranked_scores import compute_ranked_scores
from docqa.evaluator import Evaluator, Evaluation
from docqa.model_dir import ModelDir
from docqa.triviaqa.build_span_corpus import TriviaQaWebDataset, TriviaQaOpenDataset, TriviaQaWikiDataset
from docqa.triviaqa.read_data import normalize_wiki_filename
from docqa.triviaqa.training_data import DocumentParagraphQuestion, ExtractMultiParagraphs, \
    ExtractMultiParagraphsPerQuestion
from docqa.triviaqa.trivia_qa_eval import exact_match_score as trivia_em_score
from docqa.triviaqa.trivia_qa_eval import f1_score as trivia_f1_score
from docqa.utils import ResourceLoader, print_table
import pickle
import random
flag_shuffle = 0
"""
Evaluate on TriviaQA data
"""
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class RecordParagraphSpanPrediction(Evaluator):

    def __init__(self, bound: int, record_text_ans: bool):
        self.bound = bound
        self.record_text_ans = record_text_ans

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        #span_scores = prediction.get_span_scores()
        start_logit,end_logit = prediction.get_start_end_logits()        

        needed = dict(spans=span, model_scores=score,start_logits = start_logit,end_logits = end_logit)
        return needed

    def evaluate(self, data: List[DocumentParagraphQuestion], true_len, **kargs):
        spans, model_scores = np.array(kargs["spans"]), np.array(kargs["model_scores"])

        pred_f1s = np.zeros(len(data))
        pred_em = np.zeros(len(data))
        text_answers = []

        for i in tqdm(range(len(data)), total=len(data), ncols=80, desc="scoring"):
            point = data[i]
            if point.answer is None and not self.record_text_ans:
                continue
            text = point.get_context()
            pred_span = spans[i]
            pred_text = " ".join(text[pred_span[0]:pred_span[1] + 1])
            if self.record_text_ans:
                text_answers.append(pred_text)
                if point.answer is None:
                    continue

            f1 = 0
            em = False
            for answer in data[i].answer.answer_text:
                f1 = max(f1, trivia_f1_score(pred_text, answer))
                if not em:
                    em = trivia_em_score(pred_text, answer)

            pred_f1s[i] = f1
            pred_em[i] = em

        results = {}
        results["n_answers"] = [0 if x.answer is None else len(x.answer.answer_spans) for x in data]
        if self.record_text_ans:
            results["text_answer"] = text_answers
        results["predicted_score"] = model_scores
        results["predicted_start"] = spans[:, 0]
        results["predicted_end"] = spans[:, 1]
        results["text_f1"] = pred_f1s
        results["rank"] = [x.rank for x in data]
        results["text_em"] = pred_em
        results["para_start"] = [x.para_range[0] for x in data]
        results["para_end"] = [x.para_range[1] for x in data]
        results["question_id"] = [x.question_id for x in data]
        results["doc_id"] = [x.doc_id for x in data]
        return Evaluation({}, results)


def main():

    parser = argparse.ArgumentParser(description='Evaluate a model on TriviaQA data')
    parser.add_argument('model', help='model directory')
    parser.add_argument('-p', '--paragraph_output', type=str,
                        help="Save fine grained results for each paragraph in csv format")
    parser.add_argument('-o', '--official_output', type=str, help="Build an offical output file with the model's"
                                                                  " most confident span for each (question, doc) pair")
    parser.add_argument('--no_ema', action="store_true", help="Don't use EMA weights even if they exist")
    parser.add_argument('--n_processes', type=int, default=None,
                        help="Number of processes to do the preprocessing (selecting paragraphs+loading context) with")
    parser.add_argument('-i', '--step', type=int, default=None, help="checkpoint to load, default to latest")
    parser.add_argument('-n', '--n_sample', type=int, default=None, help="Number of questions to evaluate on")
    parser.add_argument('-a', '--async', type=int, default=0)
    parser.add_argument('-t', '--tokens', type=int, default=400,
                        help="Max tokens per a paragraph")
    parser.add_argument('-g', '--n_paragraphs', type=int, default=30,
                        help="Number of paragraphs to run the model on")
    parser.add_argument('-b', '--batch_size', type=int, default=200,
                        help="Batch size, larger sizes might be faster but wll take more memory")
    parser.add_argument('--max_answer_len', type=int, default=8,
                        help="Max answer span to select")
    parser.add_argument('-c', '--corpus',
                        choices=["web-dev", "web-test", "web-verified-dev", "web-train",
                                 "open-dev", "open-train", "wiki-dev", "wiki-test"],
                        default="web-verified-dev")
    parser.add_argument('-shuffle', '--shuffle', type=int, default=0, help="shuffle flag")
    parser.add_argument('-rank', '--rank', type=int, default=1, help="rank flag")#0 not rank   ,   1 rank
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    model = model_dir.get_model()
    first_K = args.n_paragraphs
    flag_shuffle = args.shuffle
    flag_rank = args.rank


    print("Building question/paragraph pairs...")
    # Loads the relevant questions/documents, selects the right paragraphs, and runs the model's preprocessor
    with open('tmp_data/list_test_p.pkl','rb') as fin:
        list_dev_p = pickle.load(fin)
    with open('tmp_data/list_test_q.pkl','rb') as fin:
        list_dev_q = pickle.load(fin)
    with open('tmp_data/list_test_q_id.pkl','rb') as fin:
        list_dev_q_id = pickle.load(fin)
    with open('tmp_data/list_test_a.pkl','rb') as fin:
        list_dev_a = pickle.load(fin)
    with open('tmp_data/id2scores_test.pkl','rb') as fin:
        id2scores_dev = pickle.load(fin)

    list_list_p_words = []
    for index,list_p in enumerate(list_dev_p):
        list_p_words = []
        scores = id2scores_dev[index]
        scores = scores[:len(list_p)]
        scores_pair = [(x,scores[x]) for x in range(len(scores))]
       
        if flag_rank == 1:
            scores_pair = sorted(scores_pair,key = lambda x:x[1],reverse=True) 
        
        
        scores_pair = scores_pair[:first_K]
        if flag_shuffle == 1:
            random.shuffle(scores_pair)
        elif flag_shuffle == 2:
            scores_pair = sorted(scores_pair, key=lambda x: x[0])


        sorted_index = [x[0] for x in scores_pair ]
        list_p_sorted = [list_p[x] for x in sorted_index]
        batch_len  = (len(list_p_sorted) // first_K ) +1

        for batch in range(batch_len):
            tmp_list_p = list_p_sorted[batch*first_K:(batch+1)*first_K]
            if len(tmp_list_p) ==0:
                break
            tmp_p = tmp_list_p[0]
            for p in tmp_list_p[1:]:
                tmp_p += ' '
                tmp_p += p
            tmp_p_words = tmp_p.split()
            list_p_words.append(tmp_p_words)
            break
        list_list_p_words.append(list_p_words)

    data = []
    for q_id,q,list_a,list_p_words in zip(list_dev_q_id,list_dev_q,list_dev_a,list_list_p_words):
        answer_text = list_a
        answer_spans =np.array([[0,1]])
        ans = TokenSpans(answer_text, answer_spans)
        question_id = q_id
        for index,list_p_words in enumerate(list_p_words):
            doc_id = str(question_id)+'_'+str(index)
            start = 0
            end = len(list_p_words)
            question = q.split()
            text = list_p_words
            data.append(DocumentParagraphQuestion(question_id,doc_id,(start,end),question,text,ans,index))


    # Reverse so our first batch will be the largest (so OOMs happen early)
    questions = sorted(data, key=lambda x: (x.n_context_words, len(x.question)), reverse=True)

    print("Done, starting eval")

    if args.step is not None:
        if args.step == "latest":
            checkpoint = model_dir.get_latest_checkpoint()
        else:
            checkpoint = model_dir.get_checkpoint(int(args.step))
    else:
        checkpoint = model_dir.get_best_weights()
        if checkpoint is not None:
            print("Using best weights")
        else:
            print("Using latest checkpoint")
            checkpoint = model_dir.get_latest_checkpoint()

    test_questions = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))

    evaluation = trainer.test(model,
                             [RecordParagraphSpanPrediction(args.max_answer_len, True)],
                              {args.corpus:test_questions}, ResourceLoader(), checkpoint, not args.no_ema, args.async)[args.corpus]

    if not all(len(x) == len(data) for x in evaluation.per_sample.values()):
        raise RuntimeError()

    df = pd.DataFrame(evaluation.per_sample)

    if args.official_output is not None:
        print("Saving question result")

        fns = {}
        answers = {}
        scores = {}
        for q_id, doc_id, start, end, txt, score in df[["question_id", "doc_id", "para_start", "para_end",
                                                        "text_answer", "predicted_score"]].itertuples(index=False):
            key = q_id

            prev_score = scores.get(key)
            if prev_score is None or prev_score < score:
                scores[key] = score
                answers[key] = txt

        with open(args.official_output, "w") as f:
            json.dump(answers, f)

    output_file = args.paragraph_output
    if output_file is not None:
        print("Saving paragraph result")
        df.to_csv(output_file, index=False)

    print("Computing scores")

    
    group_by = ["question_id"]

    # Print a table of scores as more paragraphs are used
    df.sort_values(group_by + ["rank"], inplace=True)
    f1 = compute_ranked_scores(df, "predicted_score", "text_f1", group_by)
    em = compute_ranked_scores(df, "predicted_score", "text_em", group_by)
    table = [["N Paragraphs", "EM", "F1"]]
    table += list([str(i+1), "%.4f" % e, "%.4f" % f] for i, (e, f) in enumerate(zip(em, f1)))
    print_table(table)




def submain(dir):
    parser = argparse.ArgumentParser(description='Evaluate a model on TriviaQA data')
    # parser.add_argument('model',default=model_dir)
    parser.add_argument('mode', choices=["confidence", "merge", "shared-norm",
                                         "sigmoid", "paragraph"])
    parser.add_argument("name", help="Where to store the model")
    parser.add_argument('-p', '--paragraph_output',default='paragraph-output.csv', type=str,
                        help="Save fine grained results for each paragraph in csv format")
    parser.add_argument('-o', '--official_output', default= 'question-output.json' ,type=str, help="Build an offical output file with the model's"
                                                                  " most confident span for each (question, doc) pair")
    parser.add_argument('--no_ema', action="store_true", help="Don't use EMA weights even if they exist")
    parser.add_argument('--n_processes', type=int, default=1,
                        help="Number of processes to do the preprocessing (selecting paragraphs+loading context) with")
    parser.add_argument('-i', '--step', type=int, default=None, help="checkpoint to load, default to latest")
    parser.add_argument('-n', '--n_sample', type=int, default=None, help="Number of questions to evaluate on")
    parser.add_argument('-a', '--async', type=int, default=0)
    parser.add_argument('-t', '--tokens', type=int, default=1600,
                        help="Max tokens per a paragraph")
    parser.add_argument('-g', '--n_paragraphs', type=int, default=15,
                        help="Number of paragraphs to run the model on")
    parser.add_argument('-f', '--filter', type=str, default=None, choices=["tfidf", "truncate", "linear"],
                        help="How to select paragraphs")
    parser.add_argument('-b', '--batch_size', type=int, default=200,
                        help="Batch size, larger sizes might be faster but wll take more memory")
    parser.add_argument('--max_answer_len', type=int, default=8,
                        help="Max answer span to select")
    parser.add_argument('-c', '--corpus',
                        choices=["web-dev", "web-test", "web-verified-dev", "web-train",
                                 "open-dev", "open-train", "wiki-dev", "wiki-test"],
                        default="open-dev")
    parser.add_argument("-cl", "--cl", default=0, type=int,help="continue learning")
    parser.add_argument("-out", "--out", default='result/model', type=str,help="path to model")



    args = parser.parse_args()

    # --n_processes 1 -c open-train --tokens 800 -o question-output.json -p paragraph-output.csv

    model_dir = ModelDir(dir)
    model = model_dir.get_model()


    print("Building question/paragraph pairs...")


    with open('tmp_data/list_dev_p.pkl','rb') as fin:
        list_dev_p = pickle.load(fin)
    with open('tmp_data/list_dev_q.pkl','rb') as fin:
        list_dev_q = pickle.load(fin)
    with open('tmp_data/list_dev_q_id.pkl','rb') as fin:
        list_dev_q_id = pickle.load(fin)
    with open('tmp_data/list_dev_a.pkl','rb') as fin:
        list_dev_a = pickle.load(fin)
    with open('tmp_data/id2scores_dev.pkl','rb') as fin:
        id2scores_dev = pickle.load(fin)

    list_list_p_words = []
    first_K = 30
    flag_shuffle = 0

    for index,list_p in enumerate(list_dev_p):
        list_p_words = []
        scores = id2scores_dev[index]
        scores = scores[:len(list_p)]
        scores_pair = [(x,scores[x]) for x in range(len(scores))]
        scores_pair = sorted(scores_pair,key = lambda x:x[1],reverse=True)

        scores_pair = scores_pair[:first_K]
        if flag_shuffle == 1:
            random.shuffle(scores_pair)
        elif flag_shuffle == 2:
            scores_pair = sorted(scores_pair, key=lambda x: x[0])

        sorted_index = [x[0] for x in scores_pair ]
        list_p_sorted = [list_p[x] for x in sorted_index]
        batch_len  = (len(list_p_sorted) // first_K ) +1

        for batch in range(batch_len):
            tmp_list_p = list_p_sorted[batch*first_K:(batch+1)*first_K]
            if len(tmp_list_p) ==0:
                break
            tmp_p = tmp_list_p[0]
            for p in tmp_list_p[1:]:
                tmp_p += ' '
                tmp_p += p
            tmp_p_words = tmp_p.split()
            list_p_words.append(tmp_p_words)
            break
        list_list_p_words.append(list_p_words)

    data = []
    for q_id,q,list_a,list_p_words in zip(list_dev_q_id,list_dev_q,list_dev_a,list_list_p_words):
        answer_text = list_a
        answer_spans =np.array([[0,1]])
        ans = TokenSpans(answer_text, answer_spans)
        question_id = q_id
        for index,list_p_words in enumerate(list_p_words):
            doc_id = str(question_id)+'_'+str(index)
            start = 0
            end = len(list_p_words)
            question = q.split()
            text = list_p_words
            data.append(DocumentParagraphQuestion(question_id,doc_id,(start,end),question,text,ans,index))


    # Reverse so our first batch will be the largest (so OOMs happen early)
    questions = sorted(data, key=lambda x: (x.n_context_words, len(x.question)), reverse=True)

    print("Done, starting eval")

    if args.step is not None:
        if args.step == "latest":
            checkpoint = model_dir.get_latest_checkpoint()
        else:
            checkpoint = model_dir.get_checkpoint(int(args.step))
    else:
        checkpoint = model_dir.get_best_weights()
        if checkpoint is not None:
            print("Using best weights")
        else:
            print("Using latest checkpoint")
            checkpoint = model_dir.get_latest_checkpoint()

    test_questions = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))

    evaluation = trainer.test(model,
                             [RecordParagraphSpanPrediction(args.max_answer_len, True)],
                              {args.corpus:test_questions}, ResourceLoader(), checkpoint, not args.no_ema, args.async)[args.corpus]

    if not all(len(x) == len(data) for x in evaluation.per_sample.values()):
        raise RuntimeError()

    df = pd.DataFrame(evaluation.per_sample)




    #df = pd.read_csv('paragraph-output.csv')
    if args.official_output is not None:
        print("Saving question result")

        fns = {}
        answers = {}
        scores = {}
        for q_id, doc_id, start, end, txt, score in df[["question_id", "doc_id", "para_start", "para_end",
                                                        "text_answer", "predicted_score"]].itertuples(index=False):


            key = q_id

            prev_score = scores.get(key)
            if prev_score is None or prev_score < score:
                scores[key] = score
                answers[key] = txt

        with open(args.official_output, "w") as f:
            json.dump(answers, f)

    output_file = args.paragraph_output
    if output_file is not None:
        print("Saving paragraph result")
        df.to_csv(output_file, index=False)

    print("Computing scores")

    group_by = ["question_id"]

    # Print a table of scores as more paragraphs are used
    df.sort_values(group_by + ["rank"], inplace=True)
    f1 = compute_ranked_scores(df, "predicted_score", "text_f1", group_by)
    em = compute_ranked_scores(df, "predicted_score", "text_em", group_by)
    table = [["N Paragraphs", "EM", "F1"]]
    table += list([str(i+1), "%.4f" % e, "%.4f" % f] for i, (e, f) in enumerate(zip(em, f1)))
    print_table(table)
    tmp_em = em[0]
    tmp_f1 = f1[0]

    return tmp_em,tmp_f1





if __name__ == "__main__":
    main()
    #submain('result/model-0409-192909')




