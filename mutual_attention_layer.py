import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

from losses import squared_l2


class MutualAttentionLayer(object):

    def __init__(self, method, feature_size, cluster_size):

        self.method = method
        self.feature_size = feature_size
        self.cluster_size = cluster_size

        if method == 'old':
            self.attention_layer = tf.get_variable("attention_layer_weights", [2*self.feature_size, 1],
                                                   initializer=tf.random_normal_initializer(
                                                       stddev=1 / math.sqrt(2*self.feature_size)))
        elif method == 'dual-cg':
            self.gating_weights = tf.get_variable("attention_layer_weights", [2*self.feature_size, self.feature_size],
                                                  initializer=tf.random_normal_initializer(
                                                      stddev=1 / math.sqrt(2*self.feature_size)))
            self.gating_biases = tf.get_variable("attention_layer_biases", [self.feature_size],
                                                 initializer=tf.random_normal_initializer(
                                                     stddev=1 / math.sqrt(self.feature_size)))

    def forward(self, anchor, sample, is_training):

        # If anchor has more dims than 3, collapse : [batch_size, cluster_no, feature_size]
        reshape = None
        if len(anchor.shape) == 4:
            reshape = anchor.shape[:2]
            anchor = tf.reshape(anchor, tuple([-1] + list(anchor.shape[2:])))
        elif len(anchor.shape) != 3:
            raise ValueError("Do not know how to handle it!")

        # If anchor has more dims than 3, collapse : [batch_size, cluster_no, feature_size]
        if len(sample.shape) == 4:
            reshape = sample.shape[:2]
            sample = tf.reshape(sample, tuple([-1] + list(sample.shape[2:])))
        elif len(sample.shape) != 3:
            raise ValueError("Do not know how to handle it!")

        # If anchor needs to be tiled
        if anchor.get_shape()[0] == 1 and anchor.get_shape()[0] != sample.get_shape()[0]:
            sample_no = sample.get_shape()[0]
            anchor = tf.tile(anchor, (sample_no, 1, 1))

        # anchor and sample are of size [batch_size, cluster_no, feature_size]
        attention_input = tf.concat([anchor, sample], axis=-1)
        print('[MutualAttentionLayer] attention_input', attention_input.shape)

        #######################################################################
        # OLD METHOD
        #######################################################################

        if self.method == 'old':

            # attention_input is of size [batch_size, cluster_no, feature_size*2]
            attention_weights = tf.matmul(tf.reshape(attention_input, (-1, self.feature_size*2)), self.attention_layer)
            attention_weights = tf.squeeze(attention_weights, axis=-1)
            # print('[MutualAttentionLayer] attention_weights', attention_weights.shape)

            # reshape to previous: [batch_size, cluster_no]
            attention_weights = tf.reshape(attention_weights, (-1, self.cluster_size))
            # print('[MutualAttentionLayer] attention_weights', attention_weights.shape)

            # Normalize weights with softmax
            attention_weights = tf.nn.softmax(attention_weights, axis=-1)
            # print('[MutualAttentionLayer] attention_weights', attention_weights.shape)

            # Calculate distances
            distances = tf.compat.v1.losses.cosine_distance(anchor, sample, axis=-1,
                                                            reduction=tf.compat.v1.losses.Reduction.NONE)
            distances = tf.squeeze(distances, axis=-1)
            # print('[MutualAttentionLayer] distances', distances.shape)

            # Weight and sum distances
            distances = tf.multiply(distances, attention_weights)
            # print('[MutualAttentionLayer] distances', distances.shape)
            distances = tf.math.reduce_sum(distances, axis=-1)
            # print('[MutualAttentionLayer] distances', distances.shape)

        #######################################################################
        # DUAL CG
        #######################################################################

        elif self.method == 'dual-cg':

            # Calc gates
            gates = tf.matmul(attention_input, self.gating_weights)
            # gates = slim.batch_norm(gates, center=True, scale=True, is_training=is_training, scope="attention_layer_bn")
            gates += self.gating_biases
            gates = tf.sigmoid(gates)
            print('[MutualAttentionLayer] gates', gates.shape)

            # Apply weights
            anchor = tf.multiply(anchor, gates)
            print('[MutualAttentionLayer] anchor', anchor.shape)
            sample = tf.multiply(sample, gates)
            print('[MutualAttentionLayer] sample', sample.shape)

            # Calc distances
            distances = squared_l2(anchor, sample, reduce=True)
            print('[MutualAttentionLayer] distances', distances.shape)

        #######################################################################
        # Reshape if needed
        #######################################################################

        # If input had more dims than 3, expand :
        print(reshape)
        if reshape is not None:
            distances = tf.reshape(distances, tuple(reshape))
            print('[MutualAttentionLayer] distances', distances.shape)

        # Return either [batch_size] or first two dimensions if input had 4 dims
        return distances
