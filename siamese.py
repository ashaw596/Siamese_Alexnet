from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

# Siamese alexnet model with l2 normalization
# Manages own session with training, testing, saving, and restoring helpers and placeholders
class siamese_network:
    def __init__(self, input_height, input_width, channels, learning_rate=1E-5, dropout_keep_prob=0.5, contrast_loss_margin=1.0, feature_length=100, scope='alexnet_siamese'):
        self.image_1_placeholder = tf.placeholder(tf.float32, shape=[None, input_height, input_width, channels],
                                             name='image_1_placeholder')
        self.image_2_placeholder = tf.placeholder(tf.float32, shape=[None, input_height, input_width, channels],
                                             name='image_2_placeholder')

        self.y_placeholder = tf.placeholder(tf.bool, shape=[None], name='y_placeholder')
        self.is_training_placeholder = tf.placeholder(tf.bool, shape=None, name='is_training_placeholder')

        self.feature1, self.feature2 = siamese_alexnet_model(input1=self.image_1_placeholder, input2=self.image_2_placeholder,
                                                   feature_length=feature_length, dropout_keep_prob=dropout_keep_prob,
                                                   is_training=self.is_training_placeholder, scope=scope)

        self.losses, self.square_dist = contrastive_loss(self.feature1, self.feature2, Y=self.y_placeholder, margin=contrast_loss_margin)

        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='opt')

        self.min_op = self.optimizer.minimize(self.loss)

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        self.saver = tf.train.Saver(self.trainable_variables)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __enter__(self):
        return self

    def train(self, images_1, images_2, labels):
        opt, l, sq_dist = self.sess.run([self.min_op, self.loss, self.square_dist],
                 feed_dict={self.image_1_placeholder: images_1, self.image_2_placeholder: images_2, self.y_placeholder: labels,
                            self.is_training_placeholder: True})

        return opt, l, np.sqrt(sq_dist)

    def test(self, images_1, images_2, labels):
        l, sq_dist, features1, features2 = self.sess.run([self.loss, self.square_dist, self.feature1, self.feature2],
                                        feed_dict={self.image_1_placeholder: images_1,
                                                   self.image_2_placeholder: images_2, self.y_placeholder: labels,
                                                   self.is_training_placeholder: False})
        return l, sq_dist, features1, features2

    def inference(self, images):
        features1, = self.sess.run([self.feature1], feed_dict={self.image_1_placeholder: images, self.is_training_placeholder: False})
        return features1

    def save(self, path, global_step):
        self.saver.save(self.sess, path, global_step)

    def load(self, path):
        self.saver.restore(self.sess, path)

    def __exit__(self, type, value, traceback):
        self.sess.close()

def contrastive_loss(feature1, feature2, Y, margin):
    #Epsilon to prevent NANs from numerical errors
    epsilon = 1e-10
    square_dist = tf.reduce_sum(tf.square(feature1 - feature2), axis=1)
    return (1 - tf.cast(Y, dtype=tf.float32)) * (square_dist) / 2 + tf.cast(Y, dtype=tf.float32) / 2 * tf.square(
        tf.nn.relu(margin - tf.sqrt(square_dist + epsilon))), square_dist


# Modified from tensorflow/contrib/slim/python/slim/nets/alexnet.py
def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               reuse=False,
               scope='alexnet_v2'):
    """AlexNet version 2.
    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    layers-imagenet-1gpu.cfg
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224. To use in fully
          convolutional mode, set spatial_squeeze to false.
          The LRN layers have been removed and change the initializers from
          random_normal_initializer to xavier_initializer.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        with arg_scope(
                [layers.conv2d, layers_lib.fully_connected],
                reuse=reuse):
            net = layers.conv2d(
                inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = layers.conv2d(net, 192, [5, 5], scope='conv2')
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = layers.conv2d(net, 384, [3, 3], scope='conv3')
            net = layers.conv2d(net, 384, [3, 3], scope='conv4')
            net = layers.conv2d(net, 256, [3, 3], scope='conv5')
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers.
            with arg_scope(
                    [layers.conv2d],
                    weights_initializer=trunc_normal(0.005),
                    biases_initializer=init_ops.constant_initializer(0)):
                net = layers.conv2d(net, 200, [5, 5], padding='VALID', scope='fc6')
                net = layers_lib.dropout(
                    net, 1.0, is_training=is_training, scope='dropout6')
                net = layers.conv2d(net, 200, [1, 1], scope='fc7')
                net = layers_lib.dropout(
                    net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = layers.conv2d(
                    net,
                    num_classes, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    scope='fc8')

            print(net)
            if spatial_squeeze:
                net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
            return net

#Siamese alexnet model with l2 normalization
def siamese_alexnet_model(input1, input2, feature_length, is_training=True, dropout_keep_prob=0.5,
                          scope='alexnet_siamese', reuse=False):
    features1 = alexnet_v2(input1, num_classes=feature_length, is_training=is_training,
                           dropout_keep_prob=dropout_keep_prob, spatial_squeeze=True, reuse=reuse, scope=scope)
    features2 = alexnet_v2(input2, num_classes=feature_length, is_training=is_training,
                           dropout_keep_prob=dropout_keep_prob, spatial_squeeze=True, reuse=True, scope=scope)

    print(features1)
    #Do l2 normalization to map feature space to sphere
    features1 = tf.nn.l2_normalize(features1, dim=1, epsilon=1e-12, name=None)
    features2 = tf.nn.l2_normalize(features2, dim=1, epsilon=1e-12, name=None)
    return features1, features2
