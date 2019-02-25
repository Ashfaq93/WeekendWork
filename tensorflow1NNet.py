# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:25:54 2019

@author: Ashfaq
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist =  input_data.read_data_sets("/tmp/data/",one_hot=True)

'''
input layer -> weights -> hidden layer -> activation -> weights -> layer l2 -> activatio
-> weights ->  output layer

cost function = predicted out to intended ouput (cross entropy)
optimization function (optimizer) -> minimizing cost (adagrad, adamoptimizer, sgd)

back prop

feed forward + backprop = epoch
'''

n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500
n_nodes_h4 = 500

n_classes= 10

batch_size = 500 #batch of 100 images at a time and learns weights and then so on

#height * width
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_layer1 = {'weights' : tf.Variable(tf.random_normal([784, n_nodes_h1])),
                     'biases' : tf.Variable(tf.random_normal([n_nodes_h1]))}
    
    hidden_layer2 = {'weights' : tf.Variable(tf.random_normal([n_nodes_h1,n_nodes_h2])),
                     'biases' : tf.Variable(tf.random_normal([n_nodes_h2]))}
    
    hidden_layer3 = {'weights' : tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])),
                     'biases' : tf.Variable(tf.random_normal([n_nodes_h3]))}
    
    hidden_layer4 = {'weights' : tf.Variable(tf.random_normal([n_nodes_h3, n_nodes_h4])),
                     'biases' : tf.Variable(tf.random_normal([n_nodes_h4]))}
    
    output_layer = {'weights' :tf.Variable(tf.random_normal([n_nodes_h4,n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    
    #input data * weights + biases
    
    l1  = tf.add(tf.matmul(data,hidden_layer1['weights']), hidden_layer1['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 =tf.add(tf.matmul(l1,hidden_layer2['weights']),hidden_layer2['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_layer3['weights']),hidden_layer3['biases'])
    l3 = tf.nn.relu(l3)
    
    
    l4 = tf.add(tf.matmul(l3,hidden_layer4['weights']),hidden_layer3['biases'])
    l4 = tf.nn.relu(l4)
    
    output = tf.matmul(l4, output_layer['weights'])+ output_layer['biases']
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdagradDAOptimizer(learning_rate=0.001,global_step=tf.contrib.framework.get_global_step()).minimize(cost)
    
    epoch = 20
    
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
            
            
    