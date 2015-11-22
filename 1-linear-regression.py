import tensorflow as tf
import numpy as np

# a vector of values from -1 to 1, increased by 0.2
trX = np.linspace(-1,1,101)

# a vector of (101,) which increases, but has noise
trY = 80 * trX + np.random.randn(* trX.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")

def model(X,w):
    # linear regression: w is the slope, which we want to train
    return tf.mul(X,w)

#this is a variable. yes.
w = tf.Variable(0.0, name="weights")

#this is not run yet. it's just something that CAN be run.
y_model = model(X,w)

# this is the cost function. it shows how wrong the result was.
cost = (tf.pow(Y - y_model, 2))

# the method of training. basically it will attempt to make cost return something low.
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    for (x,y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})

print sess.run(w)
