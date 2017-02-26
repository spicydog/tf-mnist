# The code is based on
# 1. https://www.youtube.com/watch?v=vq2nnJ4g6N0
# 2. https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# length of the flatten images
LENGTH = 28 * 28

# layers
K = 200
L = 100
M = 60
N = 30
O = 10

# declare variables
W1 = tf.Variable(tf.truncated_normal([LENGTH, K], stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B5 = tf.Variable(tf.zeros([O]))

X = tf.placeholder(tf.float32, [None, LENGTH])
X = tf.reshape(X, [-1, LENGTH])

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)

Y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# load MNIST data
mnist = input_data.read_data_sets("data/MNIST", one_hot=True)

for i in range(1000):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    # train the model
    sess.run(train_step, feed_dict=train_data)

    # evaluate accuracy with train data
    train_accuracy, train_cross_entropy = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    print("training", i+1, ":", train_accuracy, train_cross_entropy)

# evaluate accuracy with test data
test_data = {X: mnist.test.images, Y_: mnist.test.labels}
test_accuracy, test_cross_entropy = sess.run([accuracy, cross_entropy], feed_dict=test_data)

print("testing: ", test_accuracy, test_cross_entropy)
