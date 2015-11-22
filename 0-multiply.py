import tensorflow as tf

a = tf.placeholder("float")
b = tf.placeholder("float")

# this builds a graph that just multiplies two numbers.
# it doesnt eval right now, it has to be run first.
x = tf.mul(a,b)
sess = tf.Session()

# whatever is in these vals is what the placeholders are.
print sess.run(x, feed_dict={a: 4, b: 5})

# 4*5 = 20
