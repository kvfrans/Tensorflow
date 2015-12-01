import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, hiddenWeights, outputWeights):
    hiddenLayer = tf.nn.sigmoid(tf.matmul(X, hiddenWeights)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(hiddenLayer, outputWeights) # note that we dont take the softmax at the end because our cost fn does that for us


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

hiddenWeights = init_weights([784, 625]) # create symbolic variables
outputWeights = init_weights([625, 10])

network = model(X, hiddenWeights, outputWeights)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer
predict_op = tf.argmax(network, 1) #finds the best prediction for each image

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY}))
