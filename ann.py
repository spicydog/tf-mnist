# The code is based on
# 1. https://www.youtube.com/watch?v=vq2nnJ4g6N0
# 2. https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

LEARNING_RATE = 0.003

# length of the flatten images
LENGTH = 28 * 28
OUTPUT = 10

# layers
LAYERS = [LENGTH, 500, 100, 60, 30, OUTPUT]


def gen_weight_bias(layer):
    curr_size = LAYERS[layer]
    next_size = LAYERS[layer + 1]
    return [tf.Variable(tf.truncated_normal([curr_size, next_size], stddev=0.1)),
            tf.Variable(tf.zeros([next_size]))]


def gen_hidden(layer):
    lx = X if layer == 0 else gen_hidden(layer - 1)
    lw, lb = gen_weight_bias(layer)
    return tf.nn.relu(tf.matmul(lx, lw) + lb)

# declare variables
X = tf.placeholder(tf.float32, [None, LENGTH])
X = tf.reshape(X, [-1, LENGTH])

[w, b] = gen_weight_bias(len(LAYERS) - 2)
Y = tf.nn.softmax(tf.matmul(gen_hidden(len(LAYERS) - 3), w) + b)

Y_ = tf.placeholder(tf.float32, [None, OUTPUT])
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
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
    print("training", i + 1, ":", train_accuracy, train_cross_entropy)

# evaluate accuracy with test data
test_data = {X: mnist.test.images, Y_: mnist.test.labels}
test_accuracy, test_cross_entropy = sess.run([accuracy, cross_entropy], feed_dict=test_data)

print("testing: ", test_accuracy, test_cross_entropy)
