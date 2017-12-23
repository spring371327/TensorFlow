# _*_ coding: utf-8 _*_

import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import math

batch_size = 32
num_batches = 100

def print_activations(tensor):
    print(tensor.op.name," ",tensor.get_shape().as_list())

def inference(images):
    parameters = []
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=0.1),name='weights')
        conv = tf.nn.conv2d(images,kernel,strides=[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope)
        print_activations(conv1)
        parameters += [kernel,biases]
        lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
        pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
        print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=0.1),name='weights')
        conv = tf.nn.conv2d(pool1,kernel,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name=scope)
        print_activations(conv2)
        parameters += [kernel,biases]
        lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
        pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
        print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=0.1),name='weights')
        conv = tf.nn.conv2d(pool2,kernel,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias,name=scope)
        print_activations(conv3)
        parameters += [kernel,biases]

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=0.1),name='weights')
        conv = tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name=scope)
        print_activations(conv4)
        parameters += [kernel,biases]

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=0.1),name='weights')
        conv = tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(bias,name=scope)
        print_activations(conv5)
        parameters += [kernel,biases]
        pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool5')
        print_activations(pool5)
        return pool5,parameters

def time_tensdorflow_run(sess,target,info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_steps_burn_in+num_batches):
        start_time = time.time()
        _ = sess.run(target)
        duration_time = time.time() - start_time
        if i>=num_steps_burn_in:
            if i%10==0:
                print('%s:step %d, duration = %.3f' %(datetime.now(),i-num_steps_burn_in,duration_time))
            total_duration += duration_time
            total_duration_squared += duration_time*duration_time
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn**2
    sd = math.sqrt(vr)
    print("%s:%s across %d steps, %.3f +/- %.3f sec / batch" %(datetime.now(),info_string,num_batches,mn,sd))


def run_benchmark():
    with tf.Graph().as_default():
        images_size = 224
        images = tf.Variable(tf.random_normal([batch_size,images_size,images_size,3],dtype=tf.float32,stddev=0.1))
        pool5,parameters = inference(images)  #调用此函数形成初始化计算图，但并没运行计算图

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        time_tensdorflow_run(sess,pool5,'Forward')
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective,parameters)
        time_tensdorflow_run(sess,grad,'Forward-backward')

run_benchmark()






















































