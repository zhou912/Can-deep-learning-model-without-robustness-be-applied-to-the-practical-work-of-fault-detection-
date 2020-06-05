"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
input_train_size = 25
global_step = tf.contrib.framework.get_or_create_global_step()
class Model(object):
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None,input_train_size,input_train_size],name='x_input')
    self.y_input = tf.placeholder(tf.float32, shape = [None,2],name='y_input')
    self.x_input_change = tf.reshape(self.x_input,[-1,input_train_size,input_train_size,1])
    self.learning_rate_1 = tf.placeholder(tf.float32,name = 'learning_rate_1')
    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,64])
    b_conv1 = self._bias_variable([64])

    h_conv1 = tf.nn.relu(self._conv2d(self.x_input_change, W_conv1) + b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([3,3,64,128])
    b_conv2 = self._bias_variable([128])

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    #decoder
#    W3 = self._weight_variable([5,5,32,64])
#    b3 = self._bias_variable([32])
#    h3 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool2, W3, output_shape=[tf.shape(h_pool1)[0], 13, 13, 32], strides=[1, 2, 2, 1],
#                                           padding="SAME") + b3)
#    # 第二层解码
#    W4 = self._weight_variable([5, 5, 1, 32])
#    b4 = self._bias_variable([1])
#    h4 = tf.nn.conv2d_transpose(h3, W4, output_shape=[tf.shape(h_pool1)[0], 25, 25, 1], strides=[1, 2, 2, 1],
#                                padding="SAME") + b4


    # first fully connected layer
    W_fc1 = self._weight_variable([7 * 7 * 128, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # output layer
    W_fc2 = self._weight_variable([1024,2])
    b_fc2 = self._bias_variable([2])

    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    y_soft = tf.nn.softmax(self.pre_softmax,name = 'y_soft')
    self.pre_pro = tf.slice(y_soft,[0,1],[-1,-1])
    self.pre_pro = tf.reshape(self.pre_pro,[-1]) 
    
    cond1 = 18 * tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.pre_softmax)
    cond2 = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.pre_softmax)
    y_slice = tf.slice(self.y_input,[0,1],[-1,-1])
    y_slice = tf.reshape(y_slice,[-1])   
    self.pre_loss = tf.reduce_mean(tf.where(tf.equal(y_slice,1),cond1,cond2)) 
    
    self.total_loss = self.pre_loss 
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_1).minimize(self.total_loss,
                                                   global_step=global_step)

    correct_prediction =tf.equal(tf.argmax(self.y_input, 1), tf.argmax(y_soft, 1))
    #预测准确率
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
