import math
import tensorflow as tf


class MutualAttentionLayer(object):

    def __init__(self, feature_size, cluster_size):

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.attention_layer = tf.get_variable("attention_layer_weights", [2*self.feature_size, 1],
                                               initializer=tf.random_normal_initializer(
                                                   stddev=1 / math.sqrt(2*self.feature_size)))

    def forward(self, anchor, sample):

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

        # Reshape anchor and sample to have feature size at the very last dim: [batch_size, cluster_no, feature_size]
        anchor = tf.transpose(anchor, perm=[0, 2, 1])
        sample = tf.transpose(sample, perm=[0, 2, 1])

        # If anchor needs to be tiled
        if anchor.get_shape()[0] == 1 and anchor.get_shape()[0] != sample.get_shape()[0]:
            sample_no = sample.get_shape()[0]
            anchor = tf.tile(anchor, (sample_no, 1, 1))

        # anchor and sample are of size [batch_size, feature_size, cluster_no]
        attention_input = tf.concat([anchor, sample], axis=-1)
        # print('[MutualAttentionLayer] attention_input', attention_input.shape)

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

        # If input had more dims than 3, expand :
        if reshape is not None:
            distances = tf.reshape(distances, tuple(reshape))

        # Return either [batch_size] or first two dimensions if input had 4 dims
        return distances


def best_pos_distance_with_att(mutual_attention, query, pos_vecs):

    with tf.name_scope('best_pos_distance') as scope:

        # print('[best_pos_distance] query', query.shape)
        # print('[best_pos_distance] pos_vecs', pos_vecs.shape)

        batch_size, num_pos, feature_size, cluster_size = pos_vecs.get_shape()
        query_copies = tf.tile(query, [1, int(num_pos), 1, 1])  # shape num_pos x output_dim
        # print('[best_pos_distance] query_copies', query_copies.shape)

        distances = mutual_attention.forward(query_copies, pos_vecs)
        # print('[best_pos_distance] distances', distances.shape)

        # Find min positive
        best_pos = tf.reduce_min(distances, 1)
        return best_pos


def lazy_triplet_loss_with_att(mutual_attention, q_vec, pos_vecs, neg_vecs, margin):

    # Calc best_pos
    best_pos = best_pos_distance_with_att(mutual_attention, q_vec, pos_vecs)

    # Prepare to calc triplet loss
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg), 1, 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m = tf.fill([int(batch), int(num_neg)], margin)

    # Calulate negative distances
    neg_distances = mutual_attention.forward(query_copies, neg_vecs)

    # Calc triplet loss
    triplet_loss = tf.reduce_mean(
        tf.reduce_max(
            tf.maximum(
                tf.add(m, tf.subtract(best_pos, neg_distances)),
                tf.zeros([int(batch), int(num_neg)])), 1))
    return triplet_loss


def lazy_quadruplet_loss_with_att(mutual_attention, q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):

    # Calc lazy triplet loss
    trip_loss = lazy_triplet_loss_with_att(mutual_attention, q_vec, pos_vecs, neg_vecs, m1)

    # Calc bes pos
    best_pos = best_pos_distance_with_att(mutual_attention, q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    # Calc other negatives loss
    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1, 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)

    # Calulate negative distances
    neg_distances = mutual_attention.forward(neg_vecs, other_neg_copies)

    # Calc second loss
    second_loss = tf.reduce_mean(
        tf.reduce_max(
            tf.maximum(
                tf.add(m2, tf.subtract(best_pos, neg_distances)),
                tf.zeros([int(batch), int(num_neg)])), 1))

    # Return total loss
    total_loss = trip_loss + second_loss
    return total_loss
