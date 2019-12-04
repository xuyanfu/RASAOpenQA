import os
import time
import logging
import numpy as np
import tensorflow as tf
from layers_my import rnn
from layers_my import cudnnrnn
from layers_my import AttentionFlowMatchLayer
from layers_my import highway, conv, my_self_attention ,my_co_attention_pool, my_self_attention_pool,residual_block

from self_attention import SequenceMapperSeq, FullyConnected, ResidualLayer, VariationalDropoutLayer, TriLinear, ConcatWithProduct
from self_attention import SelfAttention


def exp_mask(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


class RCModel(object):
    """
    Implements the main reading comprehension model.
    """
    def __init__(self, word_mat,char_mat,after_word_saflag = 0,pre_word_saflag = 0,p_pool_flag = 1,sen_sa_flag = 2,learning_rate = 1e-3):


        #other parameters
        self.word_sa_flag = 1
        self.cw_flag = 1
        self.cudnnflag = 0
        self.rnnflag = 'bi-lstm'
        self.sen_sa_flag = sen_sa_flag
        self.p_pool_flag = p_pool_flag
        self.pre_word_saflag = pre_word_saflag
        self.after_word_saflag = after_word_saflag


        #self_attention_para
        self.dropout_sa = 0.1
        #pre word
        self.num_blocks_pw = 1
        self.num_heads_pw = 1
        #sen
        self.num_blocks_s = 1
        self.num_heads_s = 1
        #after word
        self.num_blocks_aw = 1
        self.num_heads_aw = 1
        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = 'BIDAF'
        self.hidden_size = 150
        self.optim_type = 'adam'
        self.learning_rate = learning_rate
        self.weight_decay = 0
        self.use_dropout = 0
        self.l2 = 0
        self.top_layer_only = False


        # the vocab
        self.word_mat = word_mat
        self.char_mat = char_mat

        # session info
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver(max_to_keep=0)

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._pre_self_attention()
        self._encode()
        self._match()
        self._fuse()
        self._self_attention()
        self._fuse_2()
        self._rank()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])#8*100,max_p
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.ph = tf.placeholder(tf.int32, [None, None, 16], "context_char")
        self.qh = tf.placeholder(tf.int32, [None, None, 16], "question_char")
        self.dropout_ebd = tf.placeholder_with_default(0.0, (), name="dropout")
        self.start_label = tf.placeholder(tf.float32, [None,None])
        self.end_label = tf.placeholder(tf.float32, [None,None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool,(),name = 'is_train')
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y") #8*100


        self.cw = tf.placeholder(tf.int32, [None, None], name="common_word")
        self.cw_q = tf.placeholder(tf.int32, [None, None], name="common_word_q")
        self.len_sen = tf.placeholder(tf.int32, [None],name='len_sen')
        self.batch_size = tf.placeholder(tf.int32,name='batch_size')

    def _embed(self):

        """
        The embedding layer, question and passage share embeddings
        """

        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
        #with tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                initializer=tf.constant(self.word_mat, dtype=tf.float32),
                trainable=False
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

            self.sep_p_encodes_type4 = self.p_emb
            self.sep_q_encodes_type4 = self.q_emb

        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(self.char_mat, dtype=tf.float32))

        with tf.variable_scope('cw_embedding'):
            self.cw_mat = tf.get_variable("cw_mat",shape=[2,4],dtype=tf.float32)

        self.cw_ebd = tf.nn.embedding_lookup(self.cw_mat, self.cw)#batch_size,max_p_len,4
        self.cw_q_ebd = tf.nn.embedding_lookup(self.cw_mat, self.cw_q)

        # there is still a len_word the character's number of one word. you must change it.
        batch_size, len_para, len_ques, len_word, hidden_size, dim_char = tf.shape(self.p)[0], tf.shape(self.p)[-1], \
                                                                          tf.shape(self.q)[-1], 16, 200, 64

        with tf.variable_scope("Input_Embedding_Layer"):

            ph_emb = tf.nn.embedding_lookup(self.char_mat, self.ph)
            ph_emb_res = tf.reshape(ph_emb, [-1, len_word, dim_char])

            qh_emb = tf.nn.embedding_lookup(self.char_mat, self.qh)
            qh_emb_res = tf.reshape(qh_emb, [-1, len_word, dim_char])

            ph_emb_conv = conv(ph_emb_res, hidden_size, bias=True, activation=tf.nn.relu, kernel_size=3,
                               name="char_conv", reuse=None)
            qh_emb_conv = conv(qh_emb_res, hidden_size, bias=True, activation=tf.nn.relu, kernel_size=3,
                               name="char_conv", reuse=True)

            ph_emb_pool = tf.reduce_max(ph_emb_conv, axis=1)
            qh_emb_pool = tf.reduce_max(qh_emb_conv, axis=1)


            ph_emb_pool_re = tf.reshape(ph_emb_pool, [batch_size, len_para, ph_emb_pool.shape[-1]])
            qh_emb_pool_re = tf.reshape(qh_emb_pool, [batch_size, len_ques, qh_emb_pool.shape[-1]])



            #concat cw
            if self.cw_flag ==1:
                p_emb_conc = tf.concat([self.p_emb, ph_emb_pool_re,self.cw_ebd], axis=2)
                q_emb_conc = tf.concat([self.q_emb, qh_emb_pool_re,self.cw_q_ebd], axis=2)
            elif self.cw_flag ==0:
                p_emb_conc = tf.concat([self.p_emb, ph_emb_pool_re], axis=2)
                q_emb_conc = tf.concat([self.q_emb, qh_emb_pool_re], axis=2)


            self.p_emb = highway(p_emb_conc, size=2*self.hidden_size, scope="highway", dropout=self.dropout_ebd, reuse=None)
            self.q_emb = highway(q_emb_conc, size=2*self.hidden_size, scope="highway", dropout=self.dropout_ebd, reuse=True)

            self.q_emb_hw = self.q_emb


        self.mask_p_emb = tf.cast(tf.sequence_mask(self.p_length), tf.float32)  # batch*len_s,max_p
        self.mask_p_emb_re = tf.reshape(self.mask_p_emb, [self.batch_size, -1])  # batch,len_s*max_p

    #todo pre word_level self_attention
    def _pre_self_attention(self):
        '''
        input : p_emb #batch*len_s,max_p,504
        out: p_emb
        '''

        if self.pre_word_saflag == 1:

            self.p_emb_dim = 2*self.hidden_size

            self.p_emb_re =tf.reshape(self.p_emb,[self.batch_size,-1,self.p_emb_dim ])#batch,len_s*max_p,504

            self.p_emb_sa = residual_block(self.p_emb_re,
                               num_blocks=self.num_blocks_pw,
                               mask=self.mask_p_emb_re,#batch,len_s*maxp
                               num_filters=self.p_emb_dim,
                               num_heads=self.num_heads_pw,
                               scope="pre_word_level_Residual_Block",
                               bias=False,
                               dropout=self.dropout_sa)

            self.p_emb_sa_re = tf.reshape(self.p_emb_sa,[-1,tf.shape(self.p)[1],self.p_emb_dim ])#batch*len_s,max_p,504

            self.p_emb = self.p_emb_sa_re

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_question_encoding',reuse=tf.AUTO_REUSE):
            if self.cudnnflag ==False:
                self.sep_p_encodes, _ = rnn(self.rnnflag, self.p_emb, self.p_length, self.hidden_size)
            else:
                self.sep_p_encodes, _ = cudnnrnn(self.rnnflag, self.p_emb, self.p_length, self.hidden_size)

        # with tf.variable_scope('question_encoding'):
            if self.cudnnflag == False:
                self.sep_q_encodes, _ = rnn(self.rnnflag, self.q_emb, self.q_length, self.hidden_size)
            else:
                self.sep_q_encodes, _ = cudnnrnn(self.rnnflag, self.q_emb, self.q_length, self.hidden_size)

        self.sep_p_encodes_type3 = self.sep_p_encodes
        self.sep_q_encodes_type3 = self.sep_q_encodes

        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            if self.cudnnflag == False:
                self.fuse_p_encodes, _ = rnn(self.rnnflag, self.match_p_encodes, self.p_length,
                                             self.hidden_size, layer_num=1)
            else:
                self.fuse_p_encodes, _ =cudnnrnn(self.rnnflag, self.match_p_encodes, self.p_length,
                                             self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

            self.fuse_p_encodes_f1 = self.fuse_p_encodes

    def _self_attention(self):
        """
        Add the self_attention to the result of the fuse, the fuse_p_encode's size is
        (batch_size, time, 2*dim)
        """
        dim = self.fuse_p_encodes.shape[-1]
        atten_layer = SequenceMapperSeq(VariationalDropoutLayer(0.8),
                                        ResidualLayer(SequenceMapperSeq(
                                            SelfAttention(attention=TriLinear(bias=True), merge=ConcatWithProduct()),
                                            FullyConnected(dim, activation="relu")
                                        )),
                                        VariationalDropoutLayer(0.8)
                                        )
        self.fuse_p_encodes = atten_layer.apply(self.is_train, self.fuse_p_encodes, self.p_length)

    def _fuse_2(self):

        with tf.variable_scope('fusion_2_start'):
            if self.cudnnflag == False:
                self.fuse_p_encodes_f2, _ = rnn(self.rnnflag, self.fuse_p_encodes, self.p_length,
                                         self.hidden_size, layer_num=1)
            else:
                self.fuse_p_encodes_f2, _ = cudnnrnn(self.rnnflag, self.fuse_p_encodes, self.p_length,
                                             self.hidden_size, layer_num=1)

            if self.use_dropout:
                self.fuse_p_encodes_f2 = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

            self.m1 = self.fuse_p_encodes_f2##8*100,max_p,dim

    def _rank(self):

        if self.word_sa_flag == 0:
            self.p_sa_fuse = self.fuse_p_encodes_f1
        elif self.word_sa_flag == 1:
            self.p_sa_fuse = self.m1###8*100,max_p,dim


        '''q self attention pool '''
        with tf.variable_scope('self_attention_pool_q'):
            random_attn_vector_q_sa_pool = tf.Variable(tf.random_normal([1, 2 * self.hidden_size]), trainable=True,
                                                       name="random_attn_vector_q")
            self.q_sa_pool = my_self_attention_pool(self.sep_q_encodes, random_attn_vector_q_sa_pool,
                                                    2 * self.hidden_size, p_len=self.q_length)  # 8*100,dim

        if self.p_pool_flag ==1:

            with tf.variable_scope('co_attention_pool_p'):
                self.p_sa_sa_pool = my_co_attention_pool(self.p_sa_fuse,self.q_sa_pool,self.p_length)#8*100,dim
            self.p_sa_sa_pool = tf.reshape(self.p_sa_sa_pool,[self.batch_size,-1,2*self.hidden_size])#8,100,dim


        elif self.p_pool_flag ==0:
            self.p_sa_sa_pool = tf.reduce_max(self.p_sa_fuse,axis= 1)
            self.p_sa_sa_pool = tf.reshape(self.p_sa_sa_pool, [self.batch_size, -1, 2 * self.hidden_size])  # 8,100,dim



        #todo fix sen level self attention

        if self.sen_sa_flag == 0:
            self.z_c = self.p_sa_sa_pool

        elif self.sen_sa_flag == 1:
            with tf.variable_scope('self_attention_2'):
                self.p_sa_2 = residual_block(self.p_sa_sa_pool,
                                             num_blocks=self.num_blocks_s,
                                             mask=None,  # batch,len_s*maxp
                                             num_filters=2 * self.hidden_size,
                                             num_heads=self.num_heads_s,
                                             scope="sen_level_Residual_Block",
                                             bias=False,
                                             dropout=self.dropout_sa)
                self.p_sa_2_fuse = self.p_sa_2
            self.z_c = self.p_sa_2_fuse


        elif self.sen_sa_flag == 2:
            with tf.variable_scope('my_self_attention_2'):
                _,self.p_sa_2 = my_self_attention(self.p_sa_sa_pool)#8,100,dim

            with tf.variable_scope('rnn_3'):
                if self.cudnnflag == False:
                    self.p_sa_2_fuse, _ = rnn(self.rnnflag, self.p_sa_2, self.len_sen,self.hidden_size, layer_num=1)
                else:
                    self.p_sa_2_fuse, _ =cudnnrnn(self.rnnflag, self.p_sa_2, self.len_sen,self.hidden_size, layer_num=1)
                if self.use_dropout:
                    self.p_sa_2_fuse = tf.nn.dropout(self.p_sa_2_fuse, self.dropout_keep_prob)#1,batch,dim
            self.z_c = self.p_sa_2_fuse #8,100,dim

        self.q_sa_pool_re = tf.reshape(self.q_sa_pool,[self.batch_size,-1,2*self.hidden_size])#8,100,dim
        self.scores = tf.nn.softmax(tf.reduce_sum(tf.multiply(self.z_c, self.q_sa_pool_re), -1)) #8,100
        self.scores_re = tf.reshape(self.scores,[-1])#8*100

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))

        self.input_y_re = tf.reshape(self.input_y,[self.batch_size,-1])
        self.rank_loss = tf.reduce_mean(-tf.reduce_sum(self.input_y_re * tf.log(self.scores+1e-9) + (1 - self.input_y_re) * tf.log(1 - self.scores+1e-9)))
        self.rank_train_op = self.optimizer.minimize(self.rank_loss)

    def rank_train_batch(self, list_p_ids, list_q_ids, list_p_len, list_q_len, list_ph_ids, list_qh_ids,list_cw,list_cw_q,input_y,list_sen_len,batch_size):

        feed_dict = {self.p: list_p_ids,
                     self.q: list_q_ids,
                     self.p_length: list_p_len,
                     self.q_length: list_q_len,
                     # self.start_label: list_start_n_hot,
                     # self.end_label: list_end_n_hot,
                     self.dropout_keep_prob: 1.0,
                     self.is_train:True,
                     self.dropout_ebd: 0,
                     self.ph: list_ph_ids,
                     self.qh: list_qh_ids,
                     self.cw:list_cw,
                     self.cw_q:list_cw_q,
                     self.len_sen:list_sen_len,
                     self.batch_size:batch_size,
                     self.input_y:input_y
                     }
        _,loss = self.sess.run([self.rank_train_op, self.rank_loss], feed_dict)
        return loss

    def rank_test_batch(self,list_p_ids, list_q_ids, list_p_len, list_q_len, list_ph_ids,list_qh_ids,list_cw,list_cw_q,input_y,list_sen_len,batch_size):

        feed_dict = {self.p: list_p_ids,
                     self.q: list_q_ids,
                     self.p_length: list_p_len,
                     self.q_length: list_q_len,
                     # self.start_label: list_start_n_hot,
                     # self.end_label: list_end_n_hot,
                     self.dropout_keep_prob: 1.0,
                     self.is_train:False,
                     self.dropout_ebd: 0,
                     self.ph: list_ph_ids,
                     self.qh: list_qh_ids,
                     self.cw: list_cw,
                     self.cw_q: list_cw_q,
                     self.len_sen: list_sen_len,
                     self.batch_size:batch_size,
                     self.input_y:input_y
                     }

        list_scores, loss = self.sess.run([self.scores, self.rank_loss], feed_dict)
        input_y_re = np.reshape(input_y,(batch_size,-1))

        list_p_1 = []
        list_p_3 = []
        list_p_5 = []

        for scores,list_y in zip(list_scores,input_y_re):

            pairs = [(scores[x], list_y[x]) for x in range(len(scores))]
            pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
            total_num = np.sum(list_y)
            sorted_y_list = [x[1] for x in pairs_sorted]

            p_1 = np.sum(sorted_y_list[:1])
            p_3 = np.sum(sorted_y_list[:3]) / len(sorted_y_list[:3])
            p_5 = np.sum(sorted_y_list[:5]) / len(sorted_y_list[:5])

            list_p_1.append(p_1)
            list_p_3.append(p_3)
            list_p_5.append(p_5)

        score_p_1 = np.mean(list_p_1)
        score_p_3 = np.mean(list_p_3)
        score_p_5 = np.mean(list_p_5)

        return loss,score_p_1,score_p_3,score_p_5,list_scores

    def select_save(self, path):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, path)

    def select_restore(self, path):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, path)

    def my_try(self):

        p = np.random.randint(1, 10, size=[3*2, 10])

        q = []
        for x in range(3):
            tmp_q = np.random.randint(1, 10, size=[8])
            q.append(tmp_q)
            q.append(tmp_q)
        q = np.array(q)
        # q = np.random.randint(1, 10, size=[3*2, 8])
        p_len = [10, 10, 7,8,8,8]
        q_len = [8, 8, 6,8,8,5]
        ph = np.random.randint(1, 10, size=[3*2, 10, 16])

        qh = []
        for x in range(3):
            tmp_qh = np.random.randint(1, 10, size=[8, 16])
            qh.append(tmp_qh)
            qh.append(tmp_qh)
        qh = np.array(qh)
        # qh = np.random.randint(1, 10, size=[3*2, 8, 16])

        cw = np.random.randint(0, 2, size=[3*2, 10])
        cw_q = np.random.randint(0, 2, size=[3*2, 8])
        input_y = np.random.randint(0, 2, size=[3*2])

        strat_labels = np.random.randint(0, 2, size=[3*2, 10])
        end_labels   = np.random.randint(0, 2, size=[3*2, 10])

        len_sen = [2,2,2]
        batch_size = 3

        feed_dict = {self.p: p,
                     self.q: q,
                     self.p_length: p_len,
                     self.q_length: q_len,
                     self.dropout_keep_prob: 1.0,
                     self.is_train: False,
                     self.dropout_ebd: 0,
                     self.ph: ph,
                     self.qh: qh,
                     self.cw:cw,
                     self.cw_q:cw_q,
                     self.len_sen:len_sen,
                     self.batch_size:batch_size,
                     self.start_label:strat_labels,
                     self.end_label:end_labels,
                     self.input_y:input_y
                     }

        xxx,hidden_states = self.sess.run([self.fuse_p_encodes_f1, self.scores], feed_dict)
        # print(xxx[0])
        print (xxx.shape)
        print (hidden_states.shape)
        # print(hidden_states)
        # print(hidden_states[1])

if __name__ == '__main__':
    word_mat = np.load('tmp_data/word_mat.npy')
    char_mat = np.load('tmp_data/char_mat.npy')
    model = RCModel(word_mat, char_mat,snormflag=0,rankflag=0,pre_word_saflag=0,after_word_saflag=0,p_pool_flag=0,sen_sa_flag=2)
    model.my_try()
