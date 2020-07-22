"""Transformer Encoder Class.

Taken and adjusted from:
https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
https://github.com/auvisusAI/detr-tensorflow
"""

import tensorflow as tf


def scaled_dot_product_attention(q, k, v):
    """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
  Returns:
    output, attention_weights
  """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(object):
    def __init__(self, dim_transformer, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dim_transformer = dim_transformer
        self.depth = dim_transformer // self.num_heads

        self.wq = tf.layers.Dense(dim_transformer)
        self.wk = tf.layers.Dense(dim_transformer)
        self.wv = tf.layers.Dense(dim_transformer)

        self.dense = tf.layers.Dense(dim_transformer)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, dim_transformer)
        k = self.wk(k)  # (batch_size, seq_len, dim_transformer)
        v = self.wv(v)  # (batch_size, seq_len, dim_transformer)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dim_transformer)
        )  # (batch_size, seq_len_q, dim_transformer)

        output = self.dense(
            concat_attention
        )  # (batch_size, seq_len_q, dim_transformer)

        return output, attention_weights


class TransformerEncoder(object):
    def __init__(
        self, num_layers, dim_transformer, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.dim_transformer = dim_transformer
        self.num_layers = num_layers

        self.enc_layers = [
            TransformerEncoderLayer(
                dim_transformer, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ]

    def __call__(self, src, positional_encodings, training):

        enc_output = src

        for i in range(self.num_layers):
            enc_output = self.enc_layers[i](enc_output, positional_encodings, training)

        return enc_output


class TransformerEncoderLayer(object):
    def __init__(self, dim_transformer, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.selt_attn = MultiHeadAttention(dim_transformer, num_heads)

        self.linear1 = tf.layers.Dense(dim_feedforward, activation="relu")
        self.linear2 = tf.layers.Dense(dim_transformer)

        self.dropout1 = tf.layers.Dropout(dropout)
        self.dropout2 = tf.layers.Dropout(dropout)

    def __call__(self, src, positional_encodings, training=True):

        if positional_encodings is not None:
            q = k = src + positional_encodings
        else:
            q = k = src

        attn_output, attention_weights = self.selt_attn(src, k, q)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = tf.contrib.layers.layer_norm(src + attn_output)

        ffn_output = self.linear2(self.dropout2(self.linear1(out1)))
        ffn_output = self.dropout2(ffn_output, training=training)

        out1 = tf.cast(out1, dtype=tf.float32)
        ffn_output = tf.cast(ffn_output, dtype=tf.float32)

        out2 = tf.contrib.layers.layer_norm(out1 + ffn_output)

        return out2
