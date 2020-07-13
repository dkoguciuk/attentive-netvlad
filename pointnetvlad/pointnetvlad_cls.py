import tensorflow as tf
import numpy as np
import math
import sys
import os

#Taken from Charles Qi's pointnet code
import tf_util
from transform_nets import input_transform_net, feature_transform_net

#Adopted from Antoine Meich
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import loupe as lp

def placeholder_inputs(batch_num_queries, num_pointclouds_per_query, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_num_queries, num_pointclouds_per_query, num_point, 3))
    return pointclouds_pl

CLUSTER_SIZE = 64
OUTPUT_DIM = 256
FEATURE_SIZE = 1024

#Adopted from the original pointnet code
def forward(point_cloud, is_training, bn_decay=None, mutual=False, ordering=''):
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

    net= tf.reshape(net,[-1,FEATURE_SIZE])
    net = tf.nn.l2_normalize(net,1)

    # With mutual attention?
    if mutual:

        NetVLAD = lp.NetVLADAtt(feature_size=FEATURE_SIZE, max_samples=num_points, cluster_size=CLUSTER_SIZE,
                                output_dim=OUTPUT_DIM, gating=False, add_batch_norm=True,
                                is_training=is_training)

        output = NetVLAD.forward(net)
        print(output)

        # Reshape output
        output = tf.reshape(output, [batch_num_queries, num_pointclouds_per_query, output.shape[-2], output.shape[-1]])

    # Standard NetVLAD
    else:
        NetVLAD = lp.SelfAttentiveNetVLAD(feature_size=FEATURE_SIZE, max_samples=num_points, cluster_size=CLUSTER_SIZE,
                                          output_dim=OUTPUT_DIM, add_batch_norm=True, is_training=is_training,
                                          ordering=ordering)

        output = NetVLAD.forward(net)
        print(output)

        # Reshape output
        shape = [batch_num_queries, num_pointclouds_per_query] + list(output.shape[1:])
        output = tf.reshape(output, shape)
        print(output)

    return output
