import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


dataLenght = 100
num_steps = 5
batch_size = 10
alpha = 0.01
nW_hidden = 10

# t = np.arange(0,20.0, 0.1)
# data = np.sin(t)

t = range(dataLenght)
data = [np.sin(2*np.pi*i/dataLenght)/2 for i in range(dataLenght)]

x = tf.placeholder("float", [batch_size,num_steps])
y_ = tf.placeholder("float", [batch_size,1])

# W = tf.Variable(np.float32(np.random.rand(step, 1))*0.1)
# b = tf.Variable(np.float32(np.random.rand(1))*0.1)

# y = tf.sigmoid(tf.matmul(x, W) + b)

W_hidden = tf.Variable(tf.truncated_normal([num_steps, nW_hidden]))
b_hidden = tf.Variable(tf.truncated_normal([nW_hidden]))

W_output = tf.Variable(tf.truncated_normal([nW_hidden, 1]))
b_output = tf.Variable(tf.truncated_normal([1]))

y_hidden = tf.tanh(tf.matmul(x, W_hidden) + b_hidden)

# y = tf.sigmoid(tf.matmul(y_hidden, W_output) + b_output)
y = tf.tanh(tf.matmul(y_hidden, W_output) + b_output)


error_measure = tf.reduce_sum(tf.square(y_ - y))
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train = tf.train.GradientDescentOptimizer(alpha).minimize(error_measure)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"


for epoch in range(1000):
    xbatch = list()
    ybatch = list()
    for yy in range(batch_size):
        xbatch.append(([data[(i+yy+epoch)%len(data)] for i in range(num_steps)]))
        ybatch.append((data[(num_steps+yy+epoch)%len(data)]))

    xs = np.atleast_2d(xbatch)
    ys = np.atleast_2d(ybatch).T
    sess.run(train, feed_dict={x: xs, y_: ys})
    #print sess.run(error_measure, feed_dict={x: xs, y_: ys})
    if epoch % 71 == 0:
        print sess.run(y, feed_dict={x: xs})
        print "----------------------------------------------------------------------------------"

print "----------------------"
print "   Start testing...  "
print "----------------------"
outs = data[:num_steps]
xs = list()
xt = tf.placeholder("float", [1,num_steps])
for i in range(len(data)):
    xs = np.atleast_2d([outs[jj] for jj in range(num_steps)])
    print xs.shape
    out = sess.run(y, feed_dict={x: xs})
    outs.append(out[0][0])

plt.plot(t, data)
plt.plot(num_steps-1, outs[num_steps-1], 'ro')
plt.plot(t, outs[:len(data)])
plt.show()
