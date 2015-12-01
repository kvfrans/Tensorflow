import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, hiddenWeights, hiddenWeights2, outputWeights, inputDropoutPropability, hiddenDropoutPropability):

    # performs dropout: removes neurons that are duplicate
    X = tf.nn.dropout(X, inputDropoutPropability)

    # ReLU is just linear, but max at 0.
    hiddenLayer = tf.nn.relu(tf.matmul(X, hiddenWeights))

    # dropout for the hidden layer this time
    hiddenLayer = tf.nn.dropout(hiddenLayer,hiddenDropoutPropability)

    hiddenWeights2 = tf.nn.relu(tf.matmul(hiddenLayer,hiddenWeights2))
    hiddenWeights2 = tf.nn.dropout(hiddenWeights2,hiddenDropoutPropability)

    # this is basically the same model as before, but with another hidden layer.
    # there is also dropouts for faster training, and relu instead of sigmoid
    return tf.matmul(hiddenWeights2, outputWeights)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

hiddenWeights = init_weights([784, 625]) # create symbolic variables
hiddenWeights2 = init_weights([625, 625])
outputWeights = init_weights([625, 10])

inputDropoutPropability = tf.placeholder("float")
hiddenDropoutPropability = tf.placeholder("float")
network = model(X, hiddenWeights, hiddenWeights2, outputWeights, inputDropoutPropability, hiddenDropoutPropability)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network, Y)) # compute costs

# this is some kind of optimizer. http://cs231n.github.io/neural-networks-3/
# i'm not really sure how it works but it probably does
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(network, 1) #finds the best prediction for each image

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], inputDropoutPropability: 0.8, hiddenDropoutPropability: 0.5})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY, inputDropoutPropability: 1.0, hiddenDropoutPropability: 1.0}))
