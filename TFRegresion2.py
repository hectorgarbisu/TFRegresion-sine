import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import random

num_epochs = 100
dataset_lenght = 200
window_size = 40
batch_size = 50
alpha = 0.01
nW_hidden = 40

# t = np.arange(0,20.0, 0.1)
# data = np.sin(t)

t = range(dataset_lenght)
data = [np.sin(2*np.pi*i/100)/2 for i in range(dataset_lenght)]

x = tf.placeholder("float", [None, window_size]) #"None" as dimension for versatility between batches and non-batches
y_ = tf.placeholder("float", [batch_size,1])

# W = tf.Variable(np.float32(np.random.rand(step, 1))*0.1)
# b = tf.Variable(np.float32(np.random.rand(1))*0.1)

# y = tf.sigmoid(tf.matmul(x, W) + b)

W_hidden = tf.Variable(tf.truncated_normal([window_size, nW_hidden]))
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


for epoch in range(num_epochs):
    for current_batch in range(dataset_lenght): #One batch
        xbatch = list()
        ybatch = list()
        for yy in range(batch_size):
            xbatch.append(([data[(i+yy+current_batch)%len(data)] for i in range(window_size)]))
            ybatch.append((data[(window_size+yy+current_batch)%len(data)]))
        index_shuffle = range(batch_size)
        xbatch = [xbatch[i] for i in index_shuffle]
        ybatch = [ybatch[i] for i in index_shuffle]
        xs = np.atleast_2d(xbatch)
        ys = np.atleast_2d(ybatch).T
        sess.run(train, feed_dict={x: xs, y_: ys})
        #print sess.run(error_measure, feed_dict={x: xs, y_: ys})
    if (epoch % (num_epochs/10)) == 0:
        print "error:",sess.run(error_measure, feed_dict={x: xs, y_: ys})
        #print sess.run(y, feed_dict={x: xs})
        #print "----------------------------------------------------------------------------------"

print "----------------------"
print "   Start testing...  "
print "----------------------"
outs = data[:window_size]
for i in range(len(data)-window_size):
    xs = np.atleast_2d([outs[jj+i] for jj in range(window_size)])
    out = sess.run(y, feed_dict={x: xs})
    #print xs, out
    outs.append(out[0][0])

plt.plot(t, data)
plt.plot(window_size-1, outs[window_size-1], 'ro')
plt.plot(t, outs[:len(data)])
plt.show()
