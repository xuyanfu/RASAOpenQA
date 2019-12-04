import tensorflow as tf
from tensorflow.contrib.keras import initializers
from tensorflow.contrib.keras import activations
from tensorflow.python.layers.core import fully_connected
from typing import Union
from typing import Optional


class SimilarityFunction():
    """
    Computes a pairwise score between elements in each sequence
    (batch, time1, dim1], (batch, time2, dim2) -> (batch, time1, time2)
    """
    def get_scores(self, tensor_1, tensor_2):
        raise NotImplementedError

    def get_one_sided_scores(self, tensor_1, tensor_2):
        return tf.squeeze(self.get_scores(tf.expand_dims(tensor_1, 1), tensor_2), squeeze_dims=[1])


class _WithBias(SimilarityFunction):
    def __init__(self, bias: bool):
        # Note since we typically do softmax on the result, having a bias is usually redundant
        self.bias = bias

    # 加一个bias
    def get_scores(self, tensor_1, tensor_2):
        out = self._distance_logits(tensor_1, tensor_2)
        if self.bias:
            bias = tf.get_variable("bias", shape=(), dtype=tf.float32)
            out += bias
        return out

    def _distance_logits(self, tensor_1, tensor_2):
        raise NotImplemented()


class TriLinear(_WithBias):
    """ Function used by BiDaF, bi-linear with an extra component for the dots of the vectors """
    def __init__(self, init="glorot_uniform", bias=False):
        super().__init__(bias)
        self.init = init

    def _distance_logits(self, x, keys):
        init = get_keras_initialization(self.init)

        # w1×hi
        key_w = tf.get_variable("key_w", shape=keys.shape.as_list()[-1], initializer=init, dtype=tf.float32)
        key_logits = tf.tensordot(keys, key_w, axes=[[2], [0]])  # (batch, key_len)

        # w2*qj
        x_w = tf.get_variable("input_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)
        x_logits = tf.tensordot(x, x_w, axes=[[2], [0]])  # (batch, x_len)

        dot_w = tf.get_variable("dot_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)

        # Compute x * dot_weights first, the batch mult with x
        x_dots = x * tf.expand_dims(tf.expand_dims(dot_w, 0), 0)
        dot_logits = tf.matmul(x_dots, keys, transpose_b=True)

        return dot_logits + tf.expand_dims(key_logits, 1) + tf.expand_dims(x_logits, 2)


def _wrap_init(init_fn):
    def wrapped(shape, dtype=None, partition_info=None):
        if partition_info is not None:
            raise ValueError()
        return init_fn(shape, dtype)
    return wrapped


def get_keras_initialization(name):
    if name is None:
        return None
    return _wrap_init(initializers.get(name))


def get_keras_activation(name: str):
    return activations.get(name)


class MergeLayer():
    def apply(self, is_train, tensor1, tensor2):
        raise NotImplemented()


class SequenceMapper():
    """ (batch, time, in_dim) -> (batch, time, out_dim) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class Mapper(SequenceMapper):
    """ (dim1, dim2, ...., input_dim) -> (im1, dim2, ...., output_dim) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()

class Updater(Mapper):
    """ (dim1, dim2, ...., input_dim) -> (im1, dim2, ...., input_dim) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class ResidualLayer(Mapper):

    def __init__(self, other: Union[Mapper, SequenceMapper]):
        self.other = other

    def apply(self, is_train, x, mask=None):
        return x + self.other.apply(is_train, x, mask)


class ConcatWithProduct(MergeLayer):
    def apply(self, is_train, tensor1, tensor2) -> tf.Tensor:
        return tf.concat([tensor1, tensor2, tensor1 * tensor2], axis=len(tensor1.shape) - 1)


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None):
    if keep_prob >= 1.0:
        return x
    return tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed), lambda: x)


class DropoutLayer(Updater):
    def __init__(self, keep_probs: float):
        self.keep_prob = keep_probs

    def apply(self, is_train, x, mask=None):
        return dropout(x, self.keep_prob, is_train)


class VariationalDropoutLayer(SequenceMapper):
    """
    `VariationalDropout` is an overload term, but this is in particular referring to
    https://arxiv.org/pdf/1506.02557.pdf were the dropout mask is consistent across the time dimension
    """

    def __init__(self, keep_prob: float):
        self.keep_prob = keep_prob

    def apply(self, is_train, x, mask=None):
        shape = tf.shape(x)
        return dropout(x, self.keep_prob, is_train, [shape[0], 1, shape[2]])


class MapperSeq(Mapper):
    def __init__(self, *layers: Mapper):
        self.layers = layers

    def apply(self, is_train, x, mask=None):
        for i, layer in enumerate(self.layers):
            with tf.variable_scope("layer_" + str(i)):
                x = layer.apply(is_train, x, mask)
        return x


class FullyConnected(Mapper):
    def __init__(self, n_out,
                 w_init="glorot_uniform",
                 activation: Union[str,  None]="relu",
                 bias=True):
        self.w_init = w_init
        self.activation = activation
        self.n_out = n_out
        self.bias = bias

    def apply(self, is_train, x, mask=None):
        bias = (self.bias is None) or self.bias  # for backwards compat

        return fully_connected(x, self.n_out,
                               use_bias=bias,
                               activation=get_keras_activation(self.activation),
                               kernel_initializer=get_keras_initialization(self.w_init))


class SequenceMapperSeq(SequenceMapper):
    def __init__(self, *layers: SequenceMapper):
        self.layers = layers

    def apply(self, is_train, x, mask=None):
        for i, layer in enumerate(self.layers):
            with tf.variable_scope("layer_" + str(i)):
                x = layer.apply(is_train, x, mask)
        return x






VERY_NEGATIVE_NUMBER = -1e29

def compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim):
    """ computes a (batch, x_word_dim, key_word_dim) bool mask for clients that want masking """
    if x_mask is None and mem_mask is None:
        return None
    elif x_mask is None or mem_mask is None:
        raise NotImplementedError()

    x_mask = tf.sequence_mask(x_mask, x_word_dim)
    mem_mask = tf.sequence_mask(mem_mask, key_word_dim)
    join_mask = tf.logical_and(tf.expand_dims(x_mask, 2), tf.expand_dims(mem_mask, 1))
    return join_mask

class SelfAttention():
    """
    Basic non-recurrent attention a sequence and itself using the given SimilarityFunction
    """
    def __init__(self, attention: SimilarityFunction,
                 merge: Optional[MergeLayer]=None,
                 alignment_bias=True):
        self.alignment_bias = alignment_bias
        self.attention = attention
        self.merge = merge

    def apply(self, is_train, x, x_mask=None):
        x_word_dim = tf.shape(x)[1]

        # (batch, x_word, key_word)
        dist_matrix = self.attention.get_scores(x, x)
        # print(dist_matrix.shape)
        # a_(ii) = -inf
        dist_matrix += tf.expand_dims(tf.eye(x_word_dim) * VERY_NEGATIVE_NUMBER, 0)  # Mask out self
        # print(dist_matrix.shape)
        joint_mask = compute_attention_mask(x_mask, x_mask, x_word_dim, x_word_dim)
        if joint_mask is not None:
            dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))
        # print(dist_matrix.shape)
        if not self.alignment_bias:
            select_probs = tf.nn.softmax(dist_matrix)
        else:
            # Allow zero-attention by adding a learned bias to the normalizer
            bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
            dist_matrix = tf.exp(dist_matrix)
            select_probs = dist_matrix / (tf.reduce_sum(dist_matrix, axis=2, keep_dims=True) + bias)
        # print(select_probs.shape)
        response = tf.matmul(select_probs, x)  # (batch, x_words, q_dim)

        if self.merge is not None:
            with tf.variable_scope("merge"):
                response = self.merge.apply(is_train, response, x)
            return response
        else:
            return response

