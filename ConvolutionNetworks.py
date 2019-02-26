# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:03:01 2019

@author: Ashfaq
"""

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


n_classes = 10
batch_size  = 120 


x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')


def CNNetwork(x):
    weigths = {'weights_cov1':tf.Variable(tf.random_normal([5,5,1,32])),
               'weights_cov2': tf.Variable(tf.random_normal([5,5,32,64])),
               'weights_FC': tf.Variable(tf.random_normal([7*7,64,1024])),
               'out': tf.Variable(tf.random_normal([1024,n_classes]))}

    biases = {'biases_cov1': tf.Variable(tf.random_normal([32])),
              'biases_cov2': tf.Variable(tf.random_normal([64])),
              'biases_FC': tf.Variable(tf.random_normal([1024])),
              'out' : tf.Variable(tf.random_normal([n_classes]))} 
    x = tf.reshape(x, shape = [-1,28,28,1])
 
    conv1 = conv2d(x, weigths['weights_cov1']) 
    conv1 = maxpool2D(conv1)
    
    conv2 = conv2d(conv1, weigths['weights_cov2']) 
    conv2 = maxpool2D(conv2)

    
    fc = tf.reshape(conv2, [-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weigths['weights_FC']) + biases['biases_FC'])
    
    fc = tf.nn.dropout(fc, keep_prob = 0.6)
    output = tf.matmul(fc, weigths['out']) + biases['out']

    return output


def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides = [1,1,1,1], padding = 'SAME')

def maxpool2D(x):
                            #size of the window             movement of window        
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


def train_neural_network(x):
    prediction = CNNetwork(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdagradDAOptimizer(learning_rate=0.1,global_step=tf.contrib.framework.get_global_step()).minimize(cost)
    
    epoch = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
    
    
        for e in range(epoch):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x_lab,y_lab = mnist.train.next_batch(batch_size)
                _, c= sess.run([optimizer,cost],feed_dict = {x:x_lab,y:y_lab})
                epoch_loss +=c
#            print('Epoch',e,'complted out of',epoch,'loss: ',epoch_loss)
                correct  = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                accuracy = tf.reduce_mean(tf.cast(correct,'float')) 
            print('Epoch',e,'complted out of',epoch,'loss: ',epoch_loss)
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    
    
    
train_neural_network(x)
            
    