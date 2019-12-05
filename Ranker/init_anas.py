import numpy as np
import sys
import unicodedata
from tqdm import  tqdm
import pickle

sys_dir = './'
sys.path.append(sys_dir)

import json
import tokenizers
from multiprocessing.util import Finalize
tokenizers.set_default('corenlp_classpath', sys_dir+'/data/corenlp/*')

import string
import regex as re

tok_class = tokenizers.get_class("corenlp")
tok_opts = {}
PROCESS_TOK = tok_class(**tok_opts)
Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)



def tokenizer_text(my_str = '',uncased = True):
    '''
    :param my_str: string
    :param cased: bool
    :return: list[str]
    '''
    text = unicodedata.normalize('NFD', my_str)
    answer = PROCESS_TOK.tokenize(text)

    if uncased == True:
        answer_word = answer.words(uncased= True)
    else:
        answer_word = answer.words(uncased=False)

    return answer_word

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def produce_data_list(data_mode = 'test'):
    # data_mode = 'test'

    with open('./data/datasets/quasart/'+data_mode+'.txt') as fin:
        q_a = fin.readlines()

    with open('./data/datasets/quasart/'+data_mode+'.json') as fin:
        q_d = fin.readlines()


    print ('len(q_a):',len(q_a))
    print ('len(q_d):',len(q_d))


    list_list_ans = []
    list_question = []
    num_answers = 0

    for line in tqdm(q_a):
        line = line.strip()
        tmp = json.loads(line)

        question = ' '.join(tokenizer_text(tmp['question']))
        list_question.append(question)

        tmp_list_answer = []

        answers = tmp['answers']
        if len(answers) > 1:
            print(answers)
            num_answers += 1

        for answer in answers:
            answer = ' '.join(tokenizer_text( answer ))
            tmp_list_answer.append(answer)

        list_list_ans.append(tmp_list_answer)

    print ('num_answers>1 = ',num_answers)



    max_len = -1
    list_documents = []
    list_list_para = []
    num_3000 = 0
    id = 0

    set_doc_len = set()
    for line in tqdm(q_d):

        list_para = []
        # answer = list_answer[id]
        id += 1
        tmp = json.loads(line)
        documents = []

        # assert len(tmp) == 50
        set_doc_len.add(len(tmp))

        for index,dict in enumerate(tmp):

            # document_words = dict['document']
            # documents += document_words
            if len(dict['document']) == 0:
                continue
            try:
                document = ' '.join(dict['document'])
                document_words = tokenizer_text(document)
                documents += document_words
            except:
                print ('tokenizer document error')
                continue

            sen = ' '.join(document_words)

            list_para.append(sen)

        list_list_para.append(list_para)

        if len(documents) > max_len:
            max_len = len(documents)

        if len(documents) >3000:
            num_3000+=1
        documents = ' '.join(documents)
        list_documents.append(documents)


    print ('max_len:',max_len)
    print ('num > 3000:',num_3000)
    print ('len(list_question):',len(list_question))
    print ('len(list_list_ans):',len(list_list_ans))
    print ('len(list_documents):',len(list_documents))
    print ('len(list_list_para):',len(list_list_para))
    print ('set_doc_len',set_doc_len)

    with open('tmp_data/list_'+data_mode+'_d.pkl','wb') as fout:
        pickle.dump(list_documents,fout)

    with open('tmp_data/list_'+data_mode+'_q.pkl','wb') as fout:
        pickle.dump(list_question,fout)

    with open('tmp_data/list_'+data_mode+'_a.pkl','wb') as fout:
        pickle.dump(list_list_ans,fout)

    with open('tmp_data/list_'+data_mode+'_p.pkl','wb') as fout:
        pickle.dump(list_list_para,fout)

    list_q_id = list(range(len(list_question)))
    list_q_id_str = [str(x) for x in list_q_id]
    with open('tmp_data/list_' + data_mode + '_q_id.pkl', 'wb') as fout:
        pickle.dump(list_q_id_str, fout)

def produce_ebd(word_embeddding_path = './data/embeddings/vec_eng.txt' ):

    list_documents = []
    list_question  = []
    list_answer = []

    for data_mode in ['train','dev','test'] :
    # for data_mode in ['test']:
        with open('tmp_data/list_'+data_mode+'_d.pkl','rb') as fin:
            list_documents += pickle.load(fin)

        with open('tmp_data/list_'+data_mode+'_q.pkl','rb') as fin:
            list_question += pickle.load(fin)

    for data_mode in ['train','dev'] :
        with open('tmp_data/list_'+data_mode+'_a.pkl','rb') as fin:
            tmp_list_list_answer = pickle.load(fin)
            for tmp_list_answer in tmp_list_list_answer:
                list_answer+= tmp_list_answer

    set_char = set()
    set_word = set()

    list_all = list_documents + list_question + list_answer

    for tmp_str in tqdm(list_all):
        tmp_words = tmp_str.strip().split()
        tmp_chars = [x for x in tmp_str]
        for word in tmp_words:
            set_word.add(word)
        for char in tmp_chars:
            set_char.add(char)

    print ('len(set_word)',len(set_word))
    print ('len(set_char)',len(set_char))


    '''produce word ebd'''
    dict_word2id = {}
    word_mat = []
    word_pad_token = '<blank>'
    word_unk_token = '<unknow>'
    word_dim  = 300

    with open(word_embeddding_path, 'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            contents = line.strip().split()
            token = contents[0]
            if token not in set_word:
                continue
            try:
                tmp_list = list(map(float, contents[1:]))
                if len(tmp_list) == 300:
                    if token not in dict_word2id.keys():
                        dict_word2id[token] = len(dict_word2id)
                        word_mat.append(tmp_list)
            except:
                print ('load ebd error!')

    dict_word2id['<unknow>'] = len(dict_word2id)
    dict_word2id['<blank>'] = len(dict_word2id)
    word_mat.append( [0. for _ in range(word_dim)])
    word_mat.append( [0. for _ in range(word_dim)])

    word_mat = np.array(word_mat)
    print('word_mat_shape',word_mat.shape)
    print('len(dict_char2id)',len(dict_word2id))
    np.save('tmp_data/word_mat.npy', word_mat)
    with open('tmp_data/dict_word2id.pkl', 'wb') as fout:
        pickle.dump(dict_word2id, fout)
    print (word_mat[dict_word2id['<unknow>']])
    print ('_'*100)


    '''produce char ebd'''
    print ('len(set_char)',len(set_char))
    dict_char2id = {}
    dim = 64

    for char in set_char:
        dict_char2id[char] = len(dict_char2id)

    char_mat = np.random.rand(len(set_char), dim)
    char_mat = list(char_mat)

    dict_char2id['UNK'] = len(dict_char2id)
    dict_char2id['BLANK'] = len(dict_char2id)

    char_mat.append(np.random.normal(size=dim, loc=0, scale=0.05))
    char_mat.append(np.random.normal(size=dim, loc=0, scale=0.05))
    char_mat = np.array(char_mat, dtype=np.float32)

    print('char_mat_shape',char_mat.shape)
    print('len(dict_char2id)',len(dict_char2id))
    np.save('tmp_data/char_mat.npy', char_mat)
    with open('tmp_data/dict_char2id.pkl', 'wb') as fout:
        pickle.dump(dict_char2id, fout)


produce_data_list('train')
produce_data_list('dev')
produce_data_list('test')
produce_ebd()