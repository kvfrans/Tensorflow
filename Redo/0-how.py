import tensorflow as tf
h = tf.constant("5")
sess = tf.Session()

# sess.run with a variable just displays the variable
print sess.run(h)

x = tf.Variable(1)

# init is a function that the session must run.
init = tf.initialize_all_variables()
print sess.run(init)

# now we can see what x is
print sess.run(x)
