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
Use `gradients = optimizer.compute_gradients(lossfunction, <list of variables>)` to select vars manually.
Then `optimizer.apply_gradients(gradients)` to update the vars.

###3-logistic-regression
Train MNIST by representing images as an array of 784 RGB.
Weights are a 784 x 10 matrix.
The model output is just a matmul of the weights and input images.
The cost is then softmax difference between the correct labels and the output.
GradientDescent is then used to train, resulting in ~92% accuraccy.
