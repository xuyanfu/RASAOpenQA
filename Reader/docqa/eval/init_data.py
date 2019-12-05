import argparse
import pickle
import math
import heapq
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from docqa.data_processing.span_data import TokenSpans
from docqa.eval.ranked_scores import compute_ranked_scores
from docqa.triviaqa.training_data import DocumentParagraphQuestion
from docqa.triviaqa.trivia_qa_eval import exact_match_score as trivia_em_score
from docqa.triviaqa.trivia_qa_eval import f1_score as trivia_f1_score
from docqa.utils import print_table

from docqa.triviaqa.answer_detection import  FastNormalizedAnswerDetector
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.data_processing.multi_paragraph_qa import DocumentParagraph, MultiParagraphQuestion
from docqa.data_processing.preprocessed_corpus import FilteredData




def init_revindex(contents):
    #contents : words_list
    stopwords = []
    revindex = {}
    dict_doclen = {}

    for content_index in range(len(contents)):
        words_list = contents[content_index]
        # for word in words_list:
        #     if word not in stopwords:
        tmp_words = words_list
        #tmp_words = words_list
        for word in tmp_words:
            if word not in revindex.keys():
                revindex[word] = {}
                revindex[word][content_index] = 1
            else:
                if content_index not in revindex[word].keys():
                    revindex[word][content_index] = 1
                else:
                    revindex[word][content_index] += 1

        dict_doclen[content_index] = len(words_list)

    return revindex,dict_doclen

def interset(query,revindex):
    #query: word_list

    result = []
    for i in query:
        if i in revindex.keys():
            tmp = revindex[i].keys()
            result+=tmp
    result = list(set(result))
    return result

def init_tfidf(revindex,N):
    tfidf = {}
    item_list = list(revindex.keys())
    for item in item_list:
        docnum_list = list(revindex[item].keys())
        df_i = len(docnum_list)
        idf = math.log(N / df_i + 1)
        for doc_index in docnum_list:
            tf = int(revindex[item][doc_index])
            if doc_index not in tfidf.keys():
                tfidf[doc_index] = {}
                tfidf[doc_index][item] = float(tf * idf)
            else:
                tfidf[doc_index][item] = float(tf * idf)
    return tfidf

def BM25 (query,intset,revindex,dict_doclen,k=1.5,b=0.75):
    avdl = 0
    for doc in intset:
        avdl += dict_doclen[doc]
    avdl = avdl / len(intset)
    N = len(dict_doclen)
    result = []
    for doc in intset:
        grade = 0.
        for item in query:
            if item in revindex.keys():
                df_i = len(revindex[item].keys())
                idf = math.log(N / df_i+1)
                dl = dict_doclen[doc]
                if doc not in revindex[item].keys():
                    tf_i = 0
                else:
                    tf_i = int(revindex[item][doc])
                grade += idf*(((k + 1) * tf_i) / (k * ((1 - b) + b * dl / avdl) + tf_i))
        result.append([doc, grade])
    result = sorted(result, key=lambda x: x[1], reverse=True)
    #print(result)
    return [ x[0] for x in result]

def rank_p(q,list_p,first_n = 200):

    tmp_content_sentences_words = [list(x.split()) for x in list_p ]
    tmp_question_words = list(q.split())
    revindex, dict_doclen = init_revindex(tmp_content_sentences_words)
    intset = list(range(len(tmp_content_sentences_words)))
    index_choosen = BM25(tmp_question_words, intset, revindex, dict_doclen)
    index_choosen = index_choosen[:first_n]
    # index_choosen = sorted(index_choosen)
    list_p_sorted = [list_p[x] for x in index_choosen]

    return list_p_sorted



def find_best_span_topn(start,end,top_n = 2,max_len = 8,sen_score = 1):

    start = np.array(start)
    end = np.array(end)
    start = np.expand_dims(start,1)
    end = np.expand_dims(end,0)

    answer = []
    mat = sen_score * (start * end)

    for i in range(0, mat.shape[0]):
        row = mat[i,i:i+max_len]
        # if i == 574:
        #     print (123)
        for ii in range(len(row)):
            one_answer = {}
            one_answer['start_idx']=i
            one_answer['end_idx'] = i+ii
            one_answer['prob'] = row[ii]
            answer.append(one_answer)
    result = heapq.nlargest(top_n, answer, key=lambda d: d['prob'])
    return result

def find_best_answer_multi_topn(start_prob, end_prob, list_seg,tokens,ans_topn = 1,max_len = 8,list_scores = []):

    dict_ans_prob = {}

    for i in range(len(list_seg) - 1):

        if len(list_scores) == 0:
            sen_score = 1
        else:
            sen_score = list_scores[i]

        s_index = list_seg[i]
        e_index = list_seg[i + 1]
        list_start = start_prob[s_index:e_index]
        list_end = end_prob[s_index:e_index]
        list_word = tokens[s_index:e_index]
        answer_pairs = find_best_span_topn(list_start, list_end,ans_topn,max_len,sen_score)

        for answer_pair in answer_pairs:
            pair = (answer_pair['start_idx'],answer_pair['end_idx'])
            prob = answer_pair['prob']
            answer_words = list_word[pair[0]:pair[1] + 1]
            answer = ' '.join(answer_words)

            if answer in dict_ans_prob.keys():
                dict_ans_prob[answer] += prob
            else:
                dict_ans_prob[answer] = prob

    best_ans = ''
    best_prob = -10000000
    for answer in dict_ans_prob.keys():
        prob = dict_ans_prob[answer]
        if prob > best_prob:
            best_prob = prob
            best_ans = answer
    return best_ans,best_prob


def init():
    data_mode = 'dev'
    with open('list_'+data_mode+'_a.pkl','rb') as fin:
        list_list_a = pickle.load(fin)

    with open('list_'+data_mode+'_p.pkl','rb') as fin:
        list_list_p = pickle.load(fin)

    with open('list_'+data_mode+'_q.pkl','rb') as fin:
        list_q = pickle.load(fin)

    list_list_p_filtered = []
    first_n = 100


    for list_p in tqdm(list_list_p):
        list_p_filtered = []

        for para in list_p[:first_n]:
            para = para.replace('DOCUMENT','')
            para = para.replace('PARAGRAPH_GROUP', '')
            para = para.replace('PARAGRAPH', '')
            para = para.replace('%%%%','')
            para = para.lower()
            if para[0] == '.':
                para = para[1:]
            para = para.strip(' ')
            para_words = para.split()
            # print (len(para_words))
            list_p_filtered.append(para)
        list_list_p_filtered.append(list_p_filtered)




    assert (len(list_list_a) == len(list_list_p))
    num_in = 0
    num_total = len(list_list_a)
    list_num_200 = []
    list_list_a_out = []
    list_list_p_out = []
    list_q_out = []
    first_n = 5


    for list_a,list_p,q in tqdm(zip(list_list_a,list_list_p_filtered,list_q)):
        #todo 对pq进行数据清洗

        q = q.lower()
        # list_p_lower = [ x.lower() for x in list_p]
        list_p_lower = list_p

        # list_p = rank_p(q,list_p_lower,first_n)
        if len(list_p) <first_n:
            list_num_200.append(len(list_p))

        # print (len(list_p))
        tmp_flag = 0
        for ans in list_a:
            for para in list_p:
                para = para.strip().lower()
                if ans in para:
                    tmp_flag = 1
                    break

            if tmp_flag ==1:
                break

        if tmp_flag ==1:
            num_in+=1

        list_list_a_out.append(list_a)
        list_list_p_out.append(list_p)
        list_q_out.append(q)

    print ('first_n:',first_n)
    print ('data_mode:',data_mode)
    print ('num_in',num_in)
    print ('num_total',num_total)
    print ('list_num_200',len(list_num_200))
    print (list_num_200)


    with open('tmp_data/list_'+data_mode+'_p.pkl','wb') as fout:
        pickle.dump(list_list_p_out,fout)

    with open('tmp_data/list_'+data_mode+'_a.pkl','wb') as fout:
        pickle.dump(list_list_a_out,fout)

    with open('tmp_data/list_'+data_mode+'_q.pkl','wb') as fout:
        pickle.dump(list_q_out,fout)

def analyze():


    data_mode = 'dev'
    with open('tmp_data/list_'+data_mode+'_p.pkl','rb') as fin:
        list_list_p = pickle.load(fin)

    with open('tmp_data/list_'+data_mode+'_q.pkl','rb') as fin:
        list_q = pickle.load(fin)

    with open('tmp_data/list_'+data_mode+'_a.pkl','rb') as fin:
        list_list_a = pickle.load(fin)

    num_in_list  = []
    num_words_list = []
    total_num_in = 0
    list_list_p_re = []
    for list_p,list_a,q in zip(list_list_p,list_list_a,list_q):
        # print (len(list_p))

        num_in = 0
        num_words = 0
        d_words= []
        for p in list_p:
            for a in list_a:
                if a in p:
                    num_in+=1
                    break
            p_words = p.split()
            d_words += p_words
            # print (len(p_words))
        num_in_list.append(num_in)
        num_words_list.append(len(d_words))
        # print (num_in)
        # print(len(d_words))
        if num_in>0:
            total_num_in+=1
        # break
        list_p_re = []
        num_w = 100
        num_p = (len(d_words) // num_w) +1
        for index in range(num_p):
            tmp_p_words = d_words[index*100:(index+1)*100]
            tmp_p = ' '.join(tmp_p_words)
            list_p_re.append(tmp_p)
        # print (len(list_p_re))
        list_list_p_re.append(list_p_re)

    print (np.mean(num_in_list))
    print (np.mean(num_words_list))
    print (total_num_in)

    with open('tmp_data/list_'+data_mode+'_p_re.pkl','wb') as fout:
        pickle.dump(list_list_p_re,fout)

    with open('tmp_data/list_'+data_mode+'_p_re.pkl','rb') as fin:
        list_list_p_re =  pickle.load(fin)

    num_in_list = []
    num_words_list = []
    total_num_in = 0


    list_list_p_select = []
    for list_p, list_a, q in zip(list_list_p_re, list_list_a, list_q):

        num_in = 0
        d_words = []
        index_in = []
        index_no = []

        for index,p in enumerate(list_p):

            flag = 0
            for a in list_a:
                if a in p:
                    num_in += 1
                    flag = 1
                    break
            if flag ==1:
                index_in.append(index)
            else:
                index_no.append(index)

            p_words = p.split()
            d_words += p_words
            # print (len(p_words))
        num_select_p  = 20
        index_select  = index_in[:20]
        index_select += index_no[:(20-len(index_select))]
        index_select = sorted(index_select)
        list_p_select = [list_p[x] for x in index_select]
        list_list_p_select.append(list_p_select)



        num_in_list.append(num_in)
        num_words_list.append(len(d_words))
        # print (num_in)
        # print(len(d_words))
        if num_in > 0:
            total_num_in += 1
        # break

    print(np.mean(num_in_list))
    print(np.mean(num_words_list))
    print(total_num_in)
    with open('tmp_data/list_'+data_mode+'_p_select.pkl','wb') as fout:
        pickle.dump(list_list_p_select,fout)
    with open('tmp_data/list_'+data_mode+'_p_select.pkl','rb') as fin:
        list_list_p_select =  pickle.load(fin)

    num_in_list = []
    num_words_list = []
    total_num_in = 0
    for list_p, list_a, q in zip(list_list_p_select, list_list_a, list_q):

        num_in = 0
        d_words = []
        index_in = []
        index_no = []

        for index,p in enumerate(list_p):

            flag = 0
            for a in list_a:
                if a in p:
                    num_in += 1
                    flag = 1
                    break
            if flag ==1:
                index_in.append(index)
            else:
                index_no.append(index)

            p_words = p.split()
            d_words += p_words

        num_in_list.append(num_in)
        num_words_list.append(len(d_words))
        # print (num_in)
        # print(len(d_words))
        if num_in > 0:
            total_num_in += 1
        # break

    print(np.mean(num_in_list))
    print(np.mean(num_words_list))
    print(total_num_in)
    with open('tmp_data/list_'+data_mode+'_p_select.pkl','wb') as fout:
        pickle.dump(list_list_p_select,fout)

def select_first():

    data_mode = 'train'
    with open('tmp_data/list_' + data_mode + '_p.pkl', 'rb') as fin:
        list_list_p = pickle.load(fin)

    with open('tmp_data/list_' + data_mode + '_q.pkl', 'rb') as fin:
        list_q = pickle.load(fin)

    with open('tmp_data/list_' + data_mode + '_a.pkl', 'rb') as fin:
        list_list_a = pickle.load(fin)


    list_p_select = []
    list_list_a_select = []
    list_q_select = []
    if data_mode == 'train':

        for list_p,list_a,q in zip(list_list_p,list_list_a,list_q):
            for p in list_p:
                for a in list_a:
                    if a in p:
                        list_p_select.append([p])
                        list_list_a_select.append(list_a)
                        list_q_select.append(q)

    if data_mode == 'dev':
        for list_p,list_a,q in zip(list_list_p,list_list_a,list_q):
            p = list_p[0]
            list_p_select.append([p])
            list_list_a_select.append(list_a)
            list_q_select.append(q)



    with open('tmp_data/list_' + data_mode + '_p_se.pkl', 'wb') as fout:
        pickle.dump(list_p_select,fout)

    with open('tmp_data/list_' + data_mode + '_q_se.pkl', 'wb') as fout:
        pickle.dump(list_q_select,fout)

    with open('tmp_data/list_' + data_mode + '_a_se.pkl', 'wb') as fout:
        pickle.dump(list_list_a_select,fout)

    print (len(list_list_a_select))

def evaluate_doc_qa():
    parser = argparse.ArgumentParser(description='Evaluate a model on TriviaQA data')
    parser.add_argument('-p', '--n_paragraphs', type=int, default=30,help="Number of paragraphs to run the model on")
    parser.add_argument('-a', '--ans_topn', type=int, default=5,help="ans_topn")
    parser.add_argument('-m', '--flag_mode', type=int, default=0,help="flag_mode")
    args = parser.parse_args()

    batch_size = args.n_paragraphs
    ans_topn = args.ans_topn
    flag_mode = args.flag_mode # 0 for every sentence decode 1 for all sentence decode

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

    # with open('probs/start_probs_'+str(batch_size)+'.pkl','rb') as fin:
    #     all_start_probs = pickle.load(fin)
    #
    # with open('probs/end_probs_'+str(batch_size)+'.pkl','rb') as fin:
    #     all_end_probs = pickle.load(fin)

    with open('probs/start_probs.pkl','rb') as fin:
        all_start_probs = pickle.load(fin)

    with open('probs/end_probs.pkl','rb') as fin:
        all_end_probs = pickle.load(fin)

    start_probs = []
    end_probs = []
    for tmp_start_prob in all_start_probs:
        start_probs += list(tmp_start_prob)
    for tmp_end_prob in all_end_probs:
        end_probs   += list(tmp_end_prob)

    # dict_id2_listans = {}
    # for q_id, list_a in zip(list_dev_q_id, list_dev_a):
    #     dict_id2_listans[q_id] = list_a
    #
    # print('len(dict_id2_listans)', len(dict_id2_listans))
    # with open('dict_id2_listans.pkl', 'wb') as fout:
    #     pickle.dump(dict_id2_listans, fout)

    list_list_p_words = []
    list_list_list_seg = []
    list_list_list_score = []

    for index,list_p in tqdm(enumerate(list_dev_p)):
        list_p_words = []
        list_list_seg = []
        list_list_score = []

        scores = id2scores_dev[index]
        scores = scores[:len(list_p)]
        scores_pair = [(x,scores[x]) for x in range(len(scores))]
        scores_pair = sorted(scores_pair,key = lambda x:x[1],reverse=True)
        sorted_index = [x[0] for x in scores_pair ]
        sorted_scores = [x[1] for x in scores_pair ]

        list_p_sorted = [list_p[x] for x in sorted_index]
        batch_len  = (len(list_p_sorted) // batch_size ) +1

        for batch in range(batch_len):
            tmp_list_p = list_p_sorted[batch*batch_size:(batch+1)*batch_size]
            list_score = sorted_scores[batch*batch_size:(batch+1)*batch_size]

            if len(tmp_list_p) ==0:
                break

            tmp_p = tmp_list_p[0]
            for p in tmp_list_p[1:]:
                tmp_p += ' '
                tmp_p += p
            tmp_p_words_0 = tmp_p.split()

            tmp_p_words = []
            list_seg = []
            for p in tmp_list_p:
                tmp_p_words+= p.split()
                list_seg.append(len(tmp_p_words))

            assert (len(tmp_p_words_0) == len(tmp_p_words))

            list_p_words.append(tmp_p_words)
            list_list_seg.append(list_seg)
            list_list_score.append(list_score)
            break
        list_list_p_words.append(list_p_words)
        list_list_list_seg.append(list_list_seg)
        list_list_list_score.append(list_list_score)




    data = []
    dict_docid2seg = {}
    dict_docid2score = {}

    for q_id,q,list_a,list_p_words,list_list_seg,list_list_score in tqdm(zip(list_dev_q_id,list_dev_q,list_dev_a,list_list_p_words,list_list_list_seg,list_list_list_score)):
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

            list_score = list_list_score[index]
            dict_docid2score[doc_id] = list_score

            list_seg = list_list_seg[index]
            dict_docid2seg[doc_id] = list_seg
            # dict_docid2seg[doc_id] = [10000]

    questions = sorted(data, key=lambda x: (x.n_context_words, len(x.question)), reverse=True)
    print('len(questions)',len(questions))
    print('len(start_probs)',len(start_probs))
    print('len(end_probs)',len(end_probs))
    print('len(dict_docid2seg)',len(dict_docid2seg))


    # ans_topn = 1
    # dict_qid2dict_rank2pred = {}
    # dict_qid2pred = {}

    pred_f1s = np.zeros(len(questions))
    pred_em = np.zeros(len(questions))
    text_answers = []
    model_scores = []

    max_len = 8
    for index,q,start,end in tqdm(zip( list(range(len(questions))) ,questions,start_probs,end_probs)):
        doc_id    = q.doc_id
        list_seg  = dict_docid2seg[doc_id]
        list_score = dict_docid2seg[doc_id]
        len_words = q.n_context_words

        assert ( len(list_score) == len(list_seg) )
        # start = start[:len_words]
        # end  =  end[:len_words]
        start = np.exp(start[:len_words])
        end = np.exp(end[:len_words])

        if flag_mode == 0:
            list_seg = [0]+list_seg
        elif flag_mode == 1:
            list_seg = [0] + [len_words]

        tokens = q.context
        q_id = q.question_id
        pred,score = find_best_answer_multi_topn(start,end,list_seg,tokens,ans_topn,max_len,list_scores=[])
        # pred,score = find_best_answer_multi_topn(start,end,list_seg,tokens,ans_topn,max_len,list_scores=list_score)

        text_answers.append(pred)
        model_scores.append(score)

        f1 = 0
        em = False
        for answer in q.answer.answer_text:
            f1 = max(f1, trivia_f1_score(pred, answer))
            if not em:
                em = trivia_em_score(pred, answer)

        pred_f1s[index] = f1
        pred_em[index] = em
        # if q_id not in dict_qid2dict_rank2pred:
        #     dict_qid2dict_rank2pred[q_id] = {}
        # dict_qid2dict_rank2pred[q_id][q.rank] = pred

    results = {}
    results["n_answers"] = [0 if x.answer is None else len(x.answer.answer_spans) for x in questions]
    results["text_answer"] = text_answers
    results["predicted_score"] = model_scores
    results["text_f1"] = pred_f1s
    results["rank"] = [x.rank for x in questions]
    results["text_em"] = pred_em
    results["para_start"] = [x.para_range[0] for x in questions]
    results["para_end"] = [x.para_range[1] for x in questions]
    results["question_id"] = [x.question_id for x in questions]
    results["doc_id"] = [x.doc_id for x in questions]

    # for q_id in dict_qid2dict_rank2pred.keys():
    #     dict_qid2pred[q_id] = dict_qid2dict_rank2pred[q_id][0]
    # print ('len(dict_qid2pred)',len(dict_qid2pred))
    # with open('dict_id2pred.pkl','wb') as fout:
    #     pickle.dump(dict_qid2pred,fout)

    df = pd.DataFrame(results)
    # official_output= 'question-output.json'
    # if official_output is not None:
    #     print("Saving question result")
    #
    #     fns = {}
    #
    #     answers = {}
    #     scores = {}
    #     for q_id, doc_id, start, end, txt, score in df[["question_id", "doc_id", "para_start", "para_end",
    #                                                     "text_answer", "predicted_score"]].itertuples(index=False):
    #         key = q_id
    #         prev_score = scores.get(key)
    #         if prev_score is None or prev_score < score:
    #             scores[key] = score
    #             answers[key] = txt
    #     with open(official_output, "w") as f:
    #         json.dump(answers, f)

    # output_file = 'paragraph-output.csv'
    # if output_file is not None:
    #     print("Saving paragraph result")
    #     df.to_csv(output_file, index=False)

    print("Computing scores")
    group_by = ["question_id"]

    # Print a table of scores as more paragraphs are used
    df.sort_values(group_by + ["rank"], inplace=True)
    f1 = compute_ranked_scores(df, "predicted_score", "text_f1", group_by)
    em = compute_ranked_scores(df, "predicted_score", "text_em", group_by)
    table = [["N Paragraphs", "EM", "F1"]]
    table += list([str(i + 1), "%.4f" % e, "%.4f" % f] for i, (e, f) in enumerate(zip(em, f1)))
    print_table(table)

    # questions_0 = questions[:200]
    # max_words_len = max([ x.n_context_words for x in questions_0 ])
    # print (max_words_len)
    # print (len(start_probs[0]))



def init_train_data():

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

    tokenizer = NltkAndPunctTokenizer()
    word_tokenize = tokenizer.tokenize_paragraph_flat

    # with open('start_probs.pkl','rb') as fin:
    #     end_probs= pickle.load(fin)
    #
    # with open('end_probs.pkl','rb') as fin:
    #     start_probs= pickle.load(fin)

    # dict_id2_listans = {}
    # for q_id, list_a in zip(list_dev_q_id, list_dev_a):
    #     dict_id2_listans[q_id] = list_a
    #
    # print('len(dict_id2_listans)', len(dict_id2_listans))
    # with open('dict_id2_listans.pkl', 'wb') as fout:
    #     pickle.dump(dict_id2_listans, fout)

    list_list_p_words = []
    batch_size = 10
    list_list_list_seg = []
    list_list_list_score = []


    for index,list_p in tqdm(enumerate(list_dev_p)):
        list_p_words = []
        list_list_seg = []
        list_list_score = []

        scores = id2scores_dev[index]
        scores = scores[:len(list_p)]
        scores_pair = [(x,scores[x]) for x in range(len(scores))]
        scores_pair = sorted(scores_pair,key = lambda x:x[1],reverse=True)
        sorted_index = [x[0] for x in scores_pair ]
        sorted_scores = [x[1] for x in scores_pair ]

        list_p_sorted = [list_p[x] for x in sorted_index]
        batch_len  = (len(list_p_sorted) // batch_size ) +1

        for batch in range(batch_len):
            tmp_list_p = list_p_sorted[batch*batch_size:(batch+1)*batch_size]
            list_score = sorted_scores[batch*batch_size:(batch+1)*batch_size]

            if len(tmp_list_p) ==0:
                break

            # tmp_p = tmp_list_p[0]
            # for p in tmp_list_p[1:]:
            #     tmp_p += ' '
            #     tmp_p += p
            # tmp_p_words_0 = tmp_p.split()

            tmp_p_words = []
            list_seg = []
            for p in tmp_list_p:
                tmp_p_words+= word_tokenize(p)
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
    for q_id,q,list_a,list_p_words,list_list_seg,list_list_score in tqdm(zip(list_dev_q_id,list_dev_q,list_dev_a,list_list_p_words,list_list_list_seg,list_list_list_score)):
        tokenized_aliases = [word_tokenize(x) for x in list_a]
        detector.set_question(tokenized_aliases)
        question_id = q_id
        question = word_tokenize(q)
        index = 0
        p_words = list_p_words[0]
        answer_text = list_a
        answer_spans = []
        for s, e in detector.any_found([p_words]):
            answer_spans.append((s, e - 1))
        answer_spans = np.array(answer_spans, dtype=np.int32)

        if len(answer_spans) == 0:
            continue

        doc_id = question_id+'_'+str(index)
        start = 0
        end = len(p_words)
        text = p_words

        # list_score = list_list_score[index]
        # dict_docid2score[doc_id] = list_score
        # list_seg = list_list_seg[index]
        # dict_docid2seg[doc_id] = list_seg
        doc_paras = []
        doc_paras.append(DocumentParagraph(doc_id,start,end,index,answer_spans,text))
        with_paragraphs.append(MultiParagraphQuestion(question_id, question,answer_text,doc_paras))
        num+=1
        if num == 50:
            break

    print(123)
    return FilteredData(with_paragraphs, len(list_dev_q_id) )



evaluate_doc_qa()


