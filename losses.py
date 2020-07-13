import tensorflow as tf


def best_pos_distance_with_att(mutual_attention, query, pos_vecs):

    with tf.name_scope('best_pos_distance') as _:

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


def best_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as _:

        # print('[best_pos_distance] query', query.shape)
        # print('[best_pos_distance] pos_vecs', pos_vecs.shape)

        #######################################################################
        # Tile query
        #######################################################################

        num_pos = pos_vecs.get_shape()[1]
        if len(pos_vecs.get_shape()) == 3:
            query_copies = tf.tile(query, [1, int(num_pos), 1])  # shape num_pos x output_dim
        elif len(pos_vecs.get_shape()) == 4:
            query_copies = tf.tile(query, [1, int(num_pos), 1, 1])  # shape num_pos x output_dim
        else:
            assert False, 'WTF?'
        # print('[best_pos_distance] query_copies', query_copies.shape)

        #######################################################################
        # Calc diff
        #######################################################################

        squared_diff = tf.reduce_sum(tf.squared_difference(pos_vecs, query_copies), axis=-1)
        if len(pos_vecs.get_shape()) == 3:
            distances = squared_diff
        elif len(pos_vecs.get_shape()) == 4:
            distances = tf.reduce_mean(squared_diff, axis=-1)
        else:
            assert False, 'WTF?'
        # print('[best_pos_distance] distances', distances.shape)

        #######################################################################
        # Calc best pos
        #######################################################################

        best_pos = tf.reduce_min(distances, axis=-1)
        # print('[best_pos_distance] best_pos', best_pos.shape)

        return best_pos


def lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):

    # Calc best_pos
    best_pos = best_pos_distance(q_vec, pos_vecs)

    # Prepare to calc triplet loss
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    if len(pos_vecs.get_shape()) == 3:
        query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
    elif len(pos_vecs.get_shape()) == 4:
        query_copies = tf.tile(q_vec, [1, int(num_neg), 1, 1])
    else:
        assert False, 'WTF?'
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m = tf.fill([int(batch), int(num_neg)], margin)

    # Calulate negative distances
    squared_diff = tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies), axis=-1)
    if len(pos_vecs.get_shape()) == 3:
        neg_distances = squared_diff
    elif len(pos_vecs.get_shape()) == 4:
        neg_distances = tf.reduce_mean(squared_diff, axis=-1)
    else:
        assert False, 'WTF?'

    # Calc triplet loss
    triplet_loss_ = tf.reduce_mean(
        tf.reduce_max(
            tf.maximum(
                tf.add(m, tf.subtract(best_pos, neg_distances)),
                tf.zeros([int(batch), int(num_neg)])), 1))
    return triplet_loss_


def lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):

    # Calc lazy triplet loss
    trip_loss = lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, m1)

    # Calc best pos
    best_pos = best_pos_distance(q_vec, pos_vecs)

    # Prepare to calc other negatives loss
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    if len(pos_vecs.get_shape()) == 3:
        other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    elif len(pos_vecs.get_shape()) == 4:
        other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1, 1])
    else:
        assert False, 'WTF?'
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)

    # Calulate negative distances
    squared_diff = tf.reduce_sum(tf.squared_difference(neg_vecs, other_neg_copies), axis=-1)
    if len(pos_vecs.get_shape()) == 3:
        neg_distances = squared_diff
    elif len(pos_vecs.get_shape()) == 4:
        neg_distances = tf.reduce_mean(squared_diff, axis=-1)
    else:
        assert False, 'WTF?'

    # Calc second loss
    second_loss = tf.reduce_mean(
        tf.reduce_max(
            tf.maximum(
                tf.add(m2, tf.subtract(best_pos, neg_distances)),
                tf.zeros([int(batch), int(num_neg)])), 1))

    # Return total loss
    total_loss = trip_loss + second_loss
    return total_loss
