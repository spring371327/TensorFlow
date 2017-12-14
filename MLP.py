from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
#定义网络
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))   #标准差是0.1
b1 = tf.Variable(tf.zeros([1,h1_units]))
W2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([1,10]))

x = tf.placeholder(tf.float32,[None,in_units])
keep_prob = tf.placeholder(tf.float32)
hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
#定义损失函数和优化器
y_ = tf.placeholder(tf.float32,[None,10])
cost = -tf.reduce_sum(y_*tf.log(y),reduction_indices=[1])
cross_entropy = tf.reduce_mean(cost)
train = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
#训练
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
for i in range(3000):
    batch_x,batch_y = mnist.train.next_batch(100)
    session.run(train,feed_dict={x:batch_x,y_:batch_y,keep_prob:0.75})
#检验模型训练效果
prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))#None*1
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))
print(session.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))



