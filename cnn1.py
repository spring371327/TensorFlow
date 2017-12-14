# _*_ coding:utf-8 _*_

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#用户的输入
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])
#设置第一个卷积层
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([1,1,1,32])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#设置第二个卷积层
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([1,1,1,64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)     #图片尺寸7*7*64
#设置全连接层
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1,1024])
h_pool2_plat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_plat,w_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_prob = tf.nn.dropout(h_fc1,keep_prob)
#设置softmax层
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([1,10])
y = tf.nn.softmax(tf.matmul(h_fc1_prob,w_fc2)+b_fc2)
#定义训练模型
cost = -tf.reduce_sum(y_*tf.log(y),reduction_indices=[1])   #None*1
cross_entropy = tf.reduce_mean(cost)
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#定义准确率评测
prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))  #None*1
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))
#开始训练
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
for i in range(20000):
    batch_x,batch_y = mnist.train.next_batch(50)
    if i%100==0:
        tmp = session.run(accuracy,feed_dict={x:batch_x,y_:batch_y,keep_prob:1.0})
        print("step %d,training accuracy is %g" %(i,tmp))
    session.run(train,feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5})

print("test accuracy is %g" %(session.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})))




