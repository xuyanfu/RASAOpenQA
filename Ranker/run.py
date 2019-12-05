import numpy as np
import os
import logging

from tqdm import  tqdm
import pickle
import tensorflow as tf
import s_norm_sa_lstm_model


logging.getLogger().setLevel(logging.INFO)


def snorm_batch_init_pre(list_para, list_question, list_list_answer,first_N = 200):

    list_p = []
    list_q = []
    list_p_words = []
    list_q_words = []
    list_list_start = []
    list_list_end = []
    list_y = []


    num_total_true = 0
    num_100 = 0
    for para,list_answer,question in zip(list_para,list_list_answer, list_question):
        passage_words = para.strip().split()
        if len(passage_words) > first_N:
            num_100 += 1

        passage_words = passage_words[:first_N]
        question_words = question.strip().split()
        passage = ' '.join(passage_words)




        list_start = []
        list_end = []
        label_y = 0

        for answer in list_answer:

            if answer in passage:
                label_y = 1

            ans_words = answer.strip().split()
            for i in range(0, len(passage_words) - len(ans_words) + 1):
                if ans_words == passage_words[i: i + len(ans_words)]:
                    if i not in list_start:
                        list_start.append(i)
                    tmp_end_index = i + len(ans_words) - 1
                    if tmp_end_index not in list_end:
                        list_end.append(tmp_end_index)
                    num_total_true+=1

        list_p.append(passage)
        list_q.append(question)
        list_p_words.append(passage_words)
        list_q_words.append(question_words)
        list_list_start.append(list_start)
        list_list_end.append(list_end)
        list_y.append(label_y)

    return list_p,list_q,list_p_words,list_q_words,list_list_start,list_list_end,num_total_true,list_y

def snorm_batch_init_after(list_p,list_p_words,list_q,list_q_words,list_list_start,list_list_end, dict_word2id, dict_char2id, char_dim):

    word_pad_token = '<blank>'
    word_unk_token = '<unknow>'
    char_pad_token = 'BLANK'
    char_unk_token = 'UNK'
    word_pad_id = dict_word2id[word_pad_token]
    word_unk_id = dict_word2id[word_unk_token]
    char_pad_id = dict_char2id[char_pad_token]
    char_unk_id = dict_char2id[char_unk_token]


    list_p_len = [len(x) for x in list_p_words]
    list_q_len = [len(x) for x in list_q_words]
    p_max_len = max(list_p_len)
    q_max_len = max(list_q_len)


    list_cw = []
    list_cw_q = []

    for p_words, q in zip(list_p_words, list_q):
        cw = [0 for _ in range(p_max_len)]
        for index, word in enumerate(p_words):
            if word in q:
                cw[index] = 1
        list_cw.append(cw)
    for q_words,p in zip(list_q_words,list_p):
        cw_q = [0 for _ in range(q_max_len)]
        for index,word in enumerate(q_words):
            if word in p:
                cw_q[index] = 1
        list_cw_q.append(cw_q)

    list_p_ids = []
    list_q_ids = []
    list_ph_ids = []
    list_qh_ids = []

    for p_words, p_len in zip(list_p_words, list_p_len):

        p_ids = [word_pad_id for _ in range(p_max_len)]
        for index in range(p_len):
            my_word = p_words[index]
            if my_word in dict_word2id.keys():
                p_ids[index] = dict_word2id[my_word]
            else:
                p_ids[index] = word_unk_id
        list_p_ids.append(p_ids)

        list_p_char = []
        for index in range(p_max_len):
            if index < p_len:
                word = p_words[index]
            else:
                word = ''
            tmp_char = [char_pad_id for _ in range(char_dim)]
            for i in range(min(len(word), char_dim)):
                my_char = word[i]
                if my_char in dict_char2id.keys():
                    tmp_char[i] = dict_char2id[my_char]
                else:
                    tmp_char[i] = char_unk_id
            list_p_char.append(tmp_char)
        list_ph_ids.append(list_p_char)
    for q_words, q_len in zip(list_q_words, list_q_len):

        q_ids = [word_pad_id for _ in range(q_max_len)]
        for index in range(q_len):
            my_word = q_words[index]
            if my_word in dict_word2id.keys():
                q_ids[index] = dict_word2id[my_word]
            else:
                q_ids[index] = word_unk_id
        list_q_ids.append(q_ids)

        list_q_char = []
        for index in range(q_max_len):
            if index < q_len:
                word = q_words[index]
            else:
                word = ''
            tmp_char = [char_pad_id for _ in range(char_dim)]
            for i in range(min(len(word), char_dim)):
                my_char = word[i]
                if my_char in dict_char2id.keys():
                    tmp_char[i] = dict_char2id[my_char]
                else:
                    tmp_char[i] = char_unk_id
            list_q_char.append(tmp_char)
        list_qh_ids.append(list_q_char)

    list_start_n_hot = []
    list_end_n_hot = []
    for list_start,list_end in zip(list_list_start,list_list_end):

        start_n_hot = [0 for _ in range(p_max_len)]
        end_n_hot = [0 for _ in range(p_max_len)]

        for start in list_start:
            if start < len(start_n_hot):
                start_n_hot[start] = 1
        for end in list_end:
            if end < len(end_n_hot):
                end_n_hot[end] = 1
        list_start_n_hot.append(start_n_hot)
        list_end_n_hot.append(end_n_hot)

    return list_p_ids, list_q_ids, list_p_len, list_q_len, list_ph_ids, list_qh_ids, list_cw,list_cw_q,list_start_n_hot,list_end_n_hot

def snorm_batch_init_tuple(data_mode='dev',first_S = None,batch_size = 6):
    char_dim = 16

    # # input
    # with open('tmp_data/list_' + data_mode + '_d.pkl', 'rb') as fin:
    #     list_document = pickle.load(fin)

    with open('tmp_data/list_' + data_mode + '_q.pkl', 'rb') as fin:
        list_question = pickle.load(fin)

    with open('tmp_data/list_' + data_mode + '_a.pkl', 'rb') as fin:
        list_list_answer = pickle.load(fin)

    with open('tmp_data/list_' + data_mode + '_p.pkl', 'rb') as fin:
        list_list_para = pickle.load(fin)

    with open('tmp_data/dict_word2id.pkl', 'rb') as fin:
        dict_word2id = pickle.load(fin)

    with open('tmp_data/dict_char2id.pkl', 'rb') as fin:
        dict_char2id = pickle.load(fin)


    batch_list_p = []
    batch_list_p_words = []
    batch_list_q = []
    batch_list_q_words = []
    batch_list_list_strat = []
    batch_list_list_end = []
    batch_list_y = []

    batch_id = []
    batch_list_sen_len = []
    batch_list_ansewr = []


    for id, list_para, question, list_answer in zip(range(len(list_question)), list_list_para, list_question, list_list_answer):

        list_p  = list_para
        print ('len(list_p)',len(list_p))

        # list_p = [x for x in list_p if len(x.split()) > 5]

        if first_S != None:
            list_p = list_p[:first_S]

        # list_p = sorted(list_p, key=lambda x: len(x.split()))

        if first_S !=None:
            if len(list_p) < first_S:
                len_list_p = len(list_p)
                for i in range(len_list_p, first_S):
                    list_p.append(list_p[i - len_list_p])

        list_q = [question for _ in range(len(list_p))]
        list_list_a = [list_answer for _ in range(len(list_p))]

        if len(list_p) == 0:
            continue

        list_p_pre, list_q_pre, list_p_words_pre, list_q_words_pre, list_list_start_pre, list_list_end_pre, num_total_true,list_y = \
            snorm_batch_init_pre(list_p,list_q,list_list_a,first_N=40)

        if data_mode == 'train' and num_total_true == 0:
            continue


        batch_id.append(id)
        batch_list_sen_len.append(len(list_p_pre))
        batch_list_ansewr.append(list_answer)

        batch_list_p +=list_p_pre
        batch_list_p_words +=list_p_words_pre
        batch_list_q += list_q_pre
        batch_list_q_words += list_q_words_pre
        batch_list_list_strat += list_list_start_pre
        batch_list_list_end += list_list_end_pre
        batch_list_y +=list_y

        if len(batch_id) == batch_size or (id == len(list_question)-1 and len(batch_id) !=0 ):

            list_p_ids, list_q_ids, list_p_len, list_q_len, list_ph_ids, list_qh_ids, list_cw, list_cw_q, list_start_n_hot, list_end_n_hot = \
                snorm_batch_init_after(batch_list_p, batch_list_p_words, batch_list_q, batch_list_q_words, batch_list_list_strat, batch_list_list_end,dict_word2id, dict_char2id, char_dim)
            tuple = (batch_id, list_p_ids, list_q_ids, list_p_len, list_q_len, list_ph_ids, list_qh_ids, list_cw, list_cw_q,list_start_n_hot, list_end_n_hot,batch_list_sen_len ,batch_list_p_words, batch_list_ansewr, batch_list_y)

            batch_list_p = []
            batch_list_p_words = []
            batch_list_q = []
            batch_list_q_words = []
            batch_list_list_strat = []
            batch_list_list_end = []
            batch_list_y = []

            batch_id = []
            batch_list_sen_len = []
            batch_list_ansewr = []

            yield tuple


def snorm_salstm_rank_train(model_rc = 'models/model',data_mode = 'dev',p_pool_flag= 1,pre_word_saflag  = 1,after_word_saflag  = 1,sen_sa_flag = 2,batch_size = 6,learning_rate = 1e-3,first_S = 100):

    epoches = 40

    with open('tmp_data/dict_word2id.pkl', 'rb') as fin:
        dict_word2id = pickle.load(fin)
    word_mat = np.load('tmp_data/word_mat.npy')

    with open('tmp_data/dict_char2id.pkl', 'rb') as fin:
        dict_char2id = pickle.load(fin)
    char_mat = np.load('tmp_data/char_mat.npy')

    rc_model = s_norm_sa_lstm_model.RCModel(word_mat, char_mat,pre_word_saflag=pre_word_saflag, after_word_saflag=after_word_saflag,sen_sa_flag = sen_sa_flag,learning_rate=learning_rate,p_pool_flag= p_pool_flag)
    #rc_model.select_restore(model_rc + '.ckpt')
    #rc_model.select_restore('models_pre/sprt_salstm_rank1_sa2_b4_5e-4_S100_dropout_chs64' + '.ckpt')

    best_acc = -1
    for epoch in range(epoches):

        list_tuple = snorm_batch_init_tuple(data_mode=data_mode,first_S=first_S,batch_size = batch_size)
        list_loss = []
        num_train_set = 0

        for tuple in tqdm(list_tuple):
            batch_id, list_p_ids, list_q_ids, list_p_len, list_q_len, list_ph_ids, list_qh_ids, list_cw, list_cw_q, list_start_n_hot, list_end_n_hot, batch_list_sen_len ,batch_list_p_words, batch_ansewr,batch_list_y = tuple
            loss = rc_model.rank_train_batch(list_p_ids, list_q_ids, list_p_len, list_q_len, list_ph_ids, list_qh_ids, list_cw, list_cw_q,batch_list_y,batch_list_sen_len,len(batch_id))
            list_loss.append(loss)
            num_train_set +=len(batch_id)

        train_loss = np.mean(list_loss)
        logging.info('Average train loss mrc for epoch {} is {}'.format(epoch, train_loss))
        logging.info('len of rc set for epoch {} is {}'.format(epoch, num_train_set))
        rc_model.select_save(model_rc + '.ckpt')


        g_evaluate = tf.Graph()
        with g_evaluate.as_default():
            f1 = snorm_salstm_rank_evaluate(model_rc = model_rc,data_mode = 'dev',pre_word_saflag=pre_word_saflag, after_word_saflag=after_word_saflag,sen_sa_flag = sen_sa_flag,batch_size = batch_size,first_S=first_S )
            if f1 > best_acc:
                best_acc = f1
                rc_model.select_save(model_rc + '_best.ckpt')

def snorm_salstm_rank_evaluate(model_rc = 'models/model',data_mode = 'dev',p_pool_flag= 1,pre_word_saflag=1, after_word_saflag=1,sen_sa_flag = 2,batch_size = 6,first_S = 100):

    with open('tmp_data/dict_word2id.pkl', 'rb') as fin:
        dict_word2id = pickle.load(fin)
    word_mat = np.load('tmp_data/word_mat.npy')

    with open('tmp_data/dict_char2id.pkl', 'rb') as fin:
        dict_char2id = pickle.load(fin)
    char_mat = np.load('tmp_data/char_mat.npy')

    rc_model = s_norm_sa_lstm_model.RCModel(word_mat, char_mat,pre_word_saflag=pre_word_saflag, after_word_saflag=after_word_saflag,sen_sa_flag = sen_sa_flag,p_pool_flag= p_pool_flag)
    rc_model.select_restore(model_rc + '.ckpt')

    list_loss = []
    list_p_1 = []
    list_p_3 = []
    list_p_5 = []

    num_train_set = 0
    dict_id2scores = {}


    list_tuple = snorm_batch_init_tuple(data_mode=data_mode, first_S=first_S, batch_size=batch_size)
    for tuple in tqdm(list_tuple):
        batch_id, list_p_ids, list_q_ids, list_p_len, list_q_len, list_ph_ids, list_qh_ids, list_cw, list_cw_q, list_start_n_hot, list_end_n_hot, batch_list_sen_len, batch_list_p_words, batch_ground_answer,batch_list_y = tuple
        loss, score_p_1, score_p_3, score_p_5,list_scores = rc_model.rank_test_batch(list_p_ids, list_q_ids, list_p_len, list_q_len, list_ph_ids, list_qh_ids, list_cw, list_cw_q, batch_list_y,batch_list_sen_len,len(batch_id))
        list_loss.append(loss)
        list_p_1.append(score_p_1)
        list_p_3.append(score_p_3)
        list_p_5.append(score_p_5)
        num_train_set += len(batch_id)

        for tmp_id,scores in zip (batch_id,list_scores):
            dict_id2scores[tmp_id] = scores


    train_loss = np.mean(list_loss)
    test_p_1 = np.mean(list_p_1)
    test_p_3 = np.mean(list_p_3)
    test_p_5 = np.mean(list_p_5)


    logging.info('Average train loss mrc is {}'.format(train_loss))
    logging.info('len of rc set is {}'.format(num_train_set))
    logging.info('Average test p@k is {},{},{}'.format(test_p_1, test_p_3, test_p_5))

    with open('tmp_data/id2scores_'+data_mode+'.pkl','wb') as fout:
        pickle.dump(dict_id2scores,fout)
    logging.info('save file in tmp_data/id2scores_'+data_mode+'.pkl')
    return test_p_1

def snorm_salstm_rank_analysis(data_mode = 'dev',batch_size = 8,first_S = 100,threshold = 0.5,topn = 20):

    with open('tmp_data/id2scores_'+data_mode+'.pkl','rb') as fin:
        dict_id2scores = pickle.load(fin)

    num_train_set = 0
    list_tuple = snorm_batch_init_tuple(data_mode=data_mode, first_S=first_S, batch_size=batch_size)

    threshold = threshold
    list_a = []
    list_p = []
    list_r = []
    list_f = []
    list_num_select = []
    list_num_total_true = []

    num_h1 = 0
    num_h3 = 0
    num_h5 = 0
    num_all = 0


    for tuple in tqdm(list_tuple):
        batch_id, list_p_ids, list_q_ids, list_p_len, list_q_len, list_ph_ids, list_qh_ids, list_cw, list_cw_q, list_start_n_hot, list_end_n_hot, batch_list_sen_len, batch_list_p_words, batch_ground_answer,batch_list_y = tuple
        batch_list_y = np.reshape(batch_list_y,(len(batch_id),-1))
        for id,list_y in zip(batch_id,batch_list_y):
            scores = dict_id2scores[id]
            total_true = np.sum(list_y)
            list_pair = [(list_y[x],scores[x]) for x in range(len(list_y))]
            #list_pair = sorted(list_pair,key = lambda x:x[1],reverse=True)
            list_pair_true = [x for x in list_pair if x[0] == 1 ]
            list_pair_threshold = []
            sum_score = 0
            for pair in list_pair[:topn]:
                score = pair[1]
                sum_score += score
                list_pair_threshold.append(pair)
                if sum_score >= threshold:
                    break
            list_y_threshold = [x[0] for x in list_pair_threshold]
            num_select = len(list_y_threshold)
            p_score = np.sum(list_y_threshold) / num_select
            if total_true ==0:
                r_score = 0
            else:
                r_score = np.sum(list_y_threshold) / total_true
            if p_score+r_score ==0:
                f_score = 0
            else:
                f_score = (2*p_score*r_score) / (p_score+r_score)
            a_score = 0
            if p_score>0:
                a_score = 1
            list_a.append(a_score)
            list_p.append(p_score)
            list_r.append(r_score)
            list_f.append(f_score)
            list_num_select.append(num_select)
            list_num_total_true.append(total_true)

            list_top_y = [x[0] for x in list_pair]
            if 1 in list_top_y[:1]:
                num_h1+=1
            if 1 in list_top_y[:3]:
                num_h3+=1
            if 1 in list_top_y[:5]:
                num_h5+=1
            num_all+=1

        num_train_set += len(batch_id)

    print ('threshold:',threshold)
    print ('accuracy:',np.mean(list_a))
    print ('p:', np.mean(list_p))
    print ('r:', np.mean(list_r))
    print ('f:', np.mean(list_f))
    print('list_num_select:', np.mean(list_num_select))
    print('list_num_total_true:', np.mean(list_num_total_true))
    print ('num_train_set:',num_train_set)
    print('num_h1', num_h1 / num_all)
    print('num_h3', num_h3 / num_all)
    print('num_h5', num_h5 / num_all)


'''train'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



snorm_salstm_rank_train(model_rc ='models/ranker_pp1_sa2_b8_5e-4',data_mode='train',p_pool_flag= 1,sen_sa_flag = 2,batch_size=8,learning_rate = 5e-4,pre_word_saflag=0,after_word_saflag=0,first_S = 100)
snorm_salstm_rank_evaluate(model_rc ='models/ranker_pp1_sa2_b8_5e-4_best',data_mode='test',p_pool_flag= 1,sen_sa_flag = 2,batch_size=8,pre_word_saflag=0,after_word_saflag=0,first_S = 100)
snorm_salstm_rank_evaluate(model_rc ='models/ranker_pp1_sa2_b8_5e-4_best',data_mode='train',p_pool_flag= 1,sen_sa_flag = 2,batch_size=8,pre_word_saflag=0,after_word_saflag=0,first_S = 100)
snorm_salstm_rank_evaluate(model_rc ='models/ranker_pp1_sa2_b8_5e-4_best',data_mode='dev',p_pool_flag= 1,sen_sa_flag = 2,batch_size=8,pre_word_saflag=0,after_word_saflag=0,first_S = 100)
snorm_salstm_rank_analysis(threshold=2,data_mode='test',first_S=100,topn = 1)










