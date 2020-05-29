import tensorflow as tf
import numpy as np
import math
import sys
import os

#Taken from Charles Qi's pointnet code
import tf_util
from transform_nets import input_transform_net, feature_transform_net

#Adopted from Antoine Meich
import loupe as lp

def placeholder_inputs(batch_num_queries, num_pointclouds_per_query, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_num_queries, num_pointclouds_per_query, num_point, 3))
    return pointclouds_pl

CLUSTER_SIZE=64
OUTPUT_DIM=256
FEATURE_SIZE = 1024

#Adopted from the original pointnet code
def forward(point_cloud, is_training, bn_decay=None):
    """PointNetVLAD,    INPUT is batch_num_queries X num_pointclouds_per_query X num_points_per_pointcloud X 3, 
                        OUTPUT batch_num_queries X num_pointclouds_per_query X output_dim """
    batch_num_queries = point_cloud.get_shape()[0].value
    num_pointclouds_per_query = point_cloud.get_shape()[1].value
    num_points = point_cloud.get_shape()[2].value
    point_cloud = tf.reshape(point_cloud, [batch_num_queries*num_pointclouds_per_query, num_points,3])

    with tf.variable_scope('transform_net1') as sc:
        input_transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, input_transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        feature_transform = feature_transform_net(net, is_training, bn_decay, K=64)
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), feature_transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    NetVLAD = lp.NetVLADAtt(feature_size=FEATURE_SIZE, max_samples=num_points, cluster_size=CLUSTER_SIZE,
                    output_dim=OUTPUT_DIM, gating=False, add_batch_norm=True,
                    is_training=is_training)

    net= tf.reshape(net,[-1,1024])
    net = tf.nn.l2_normalize(net,1)
    output = NetVLAD.forward(net)
    print('batch_num_queries', batch_num_queries)
    print('num_pointclouds_per_query', num_pointclouds_per_query)
    print('num_points', num_points)
    print('CLUSTER_SIZE', CLUSTER_SIZE)
    print('input', net)
    print('output', output)

    # normalize to have norm 1
    # output = tf.nn.l2_normalize(output,1)
    # output = tf.reshape(output,[batch_num_queries,num_pointclouds_per_query,OUTPUT_DIM])
    output = tf.reshape(output, [batch_num_queries, num_pointclouds_per_query, output.shape[-2], output.shape[-1]])

    return output


class MutualAttentionLayer(object):

    def __init__(self, feature_size=FEATURE_SIZE, cluster_size=CLUSTER_SIZE):

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
        distance = tf.multiply(distances, attention_weights)
        # print('[MutualAttentionLayer] distances', distances.shape)
        distances = tf.math.reduce_sum(distances, axis=-1)
        # print('[MutualAttentionLayer] distances', distances.shape)

        # If input had more dims than 3, expand :
        if reshape is not None:
            distances = tf.reshape(distances, tuple(reshape))

        # Return either [batch_size] or first two dimensions if input had 4 dims
        return distances


def best_pos_distance(mutual_attention, query, pos_vecs):

    with tf.name_scope('best_pos_distance') as scope:

        # print('[best_pos_distance] query', query.shape)
        # print('[best_pos_distance] pos_vecs', pos_vecs.shape)

        batch_size, num_pos, feature_size, cluster_size = pos_vecs.get_shape()
        query_copies = tf.tile(query, [1, int(num_pos), 1, 1]) #shape num_pos x output_dim
        # print('[best_pos_distance] query_copies', query_copies.shape)

        distances = mutual_attention.forward(query_copies, pos_vecs)
        # print('[best_pos_distance] distances', distances.shape)

        # Find min positive
        best_pos = tf.reduce_min(distances, 1)
        return best_pos


##########Losses for PointNetVLAD###########

#Returns average loss across the query tuples in a batch, loss in each is the average loss of the definite negatives against the best positive
def triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
     # ''', end_points, reg_weight=0.001):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss

#Lazy variant
def lazy_triplet_loss(mutual_attention, q_vec, pos_vecs, neg_vecs, margin):

    # Calc best_pos
    best_pos = best_pos_distance(mutual_attention, q_vec, pos_vecs)

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


def softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    ones=tf.fill([int(batch), int(num_neg)],1.0)
    soft_loss=tf.reduce_mean(tf.reduce_sum(tf.log(tf.exp(tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2)))+1.0),1))
    return soft_loss

def lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    ones=tf.fill([int(batch), int(num_neg)],1.0)
    soft_loss=tf.reduce_mean(tf.reduce_max(tf.log(tf.exp(tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2)))+1.0),1))
    return soft_loss

def quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss= softmargin_loss(q_vec, pos_vecs, neg_vecs)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= soft_loss+second_loss

    return total_loss   

def lazy_quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss= lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= soft_loss+second_loss

    return total_loss   

def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss= triplet_loss(q_vec, pos_vecs, neg_vecs, m1)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= trip_loss+second_loss

    return total_loss 

def lazy_quadruplet_loss(mutual_attention, q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):

    # Calc lazy triplet loss
    trip_loss = lazy_triplet_loss(mutual_attention, q_vec, pos_vecs, neg_vecs, m1)

    # Calc bes pos
    best_pos = best_pos_distance(mutual_attention, q_vec, pos_vecs)
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





