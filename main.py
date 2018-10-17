import mxnet as mx
import numpy as np
from random import shuffle
import tensorflow as tf
from sklearn import preprocessing
import random
import cv2
import math

mnist = mx.test_utils.get_mnist()

# Fix the seed
mx.random.seed(42)

# Set the compute context, GPU is available otherwise CPU
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

restart_training = 0

LR = 1e-5
batch_size = 100
training_iters = 500
n_classes = 1

##train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
##val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

training_data = mnist['train_data']
target = np.array([mnist['train_label']]).reshape(60000,1)
testing_data = mnist['test_data']
validation = np.array([mnist['test_label']]).reshape(10000,1)

print(training_data.shape)
print(target.shape)
print(testing_data.shape)
print(validation.shape)

obj_height = 28
obj_width = 28

x = tf.placeholder("float", [None, 1,obj_height,obj_width])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

weights = {
    'wc1': tf.get_variable('W0', shape=(1,obj_height,obj_width,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(5,5,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(5,5,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W9', shape=(1*4*128,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd2': tf.get_variable('W10', shape=(64,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W12', shape=(32,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B9', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bd2': tf.get_variable('B10', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B12', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv_net(x, weights, biases):  
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
    print(conv3.shape)
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])    
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.layers.dropout(inputs=fc1, rate=0.5)    
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.layers.dropout(inputs=fc2, rate=0.5)    
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out 

pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)

init = tf.global_variables_initializer()

p = np.array([0.0]*58)

##gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    with tf.device("/cpu:0"):
        sess.run(init)
        saver = tf.train.Saver()
        if restart_training == 0:
            saver.restore(sess, "./CNN_model.ckpt")
        for i in range(training_iters):
            
            overall_accuracy = 0;
            file = 0
            for batch in range(math.ceil((len(training_data)/batch_size))):
                correct = 0
                batch_x = training_data[batch*batch_size:min((batch+1)*batch_size,len(training_data))]
                batch_y = target[batch*batch_size:min((batch+1)*batch_size,len(target))]    
                p = sess.run(pred, feed_dict={x: batch_x})
                q = np.round_(p).astype(int)
                for c in range(0,len(q)):
                    if q[c] == batch_y[c]:
                        correct += 1
                acc1 = (correct / len(q))*100
                overall_accuracy += acc1
                opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                loss = sess.run([cost], feed_dict={x: batch_x, y: batch_y})
                if i >= 0 and batch == 99:
                    p = sess.run(pred, feed_dict={x: testing_data})
                    q = np.round_(p).astype(int)                    
                    correct = 0
                    for c in range(len(q)):
                        if q[c] == validation[c]:
                            correct += 1
                    acc =(correct / len(q))*100
                    print("Testing Accuracy = ",acc)
            accuracy = (overall_accuracy/ (math.ceil(len(training_data)/batch_size)))
            print("Iter " + str(i) + ", Loss= " + str(loss[0])," Accuracy = ", accuracy)
            if i%1 == 0:
                save_path = saver.save(sess, "./CNN_model.ckpt")
                print("Model Saved")
        save_path = saver.save(sess, "CNN_model.ckpt")
        print("Optimization Finished!")
