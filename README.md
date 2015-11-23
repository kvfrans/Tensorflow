##Tensorflow stuff
It's just a bunch of useful examples that I either wrote or got from somewhere else and modified

###0-how
How session.run works with constants and variables

###0-multiply
Makes a basic graph that just multiplies two floats.

###1-linear-regression
Linear regression with only one weight variable and no bias.
Shows how to use `GradientDescentOptimizer.minimize`

###2-linear-bias
Linear regression, but with a weight and bias.
`GradientDescentOptimizer.minimize` apparently finds all variables to create gradients for automatically.
use `gradients = optimizer.compute_gradients(lossfunction, <list of variables>)` to select vars manually.
then `optimizer.apply_gradients(gradients)` to update the vars.

###3-logistic-regression
Train MNIST by representing images as an array of 784 RGB.
