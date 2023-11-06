import numpy as np
from matplotlib import pyplot as pp


def sigmoid(x):
    """ Activation function for x
    This will give a value between 0 (for large negative values)
    and 1 (large positive values).
    It is 0.5 for x = 0
    This is to say how 'activated' a neuron connection is
    1 is fully activated 0 is not activated at all"""
    return 1/(1+np.exp(-x))


def rate(s):
    """The rate of the sigmoid function.
    Calculus will show that the sigmoid function s(x) satisfies the equation
    ds/dx = s(1-s).  We use the rate to minimise the error."""
    return s*(1-s)


# The hidden layer is an intermediate layer of data between input and output
# It is based on a certain combination of outputs, based on the weight values, of input data
# In this project we will have only one intermediate, or hidden, layer for simplicity
HIDDEN_SIZE = 6
# The input data which, like the way data is stored on a computer, are sets of binary values
data_in = np.array([[1, 0, 1, 0, 1], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 1, 1, 1]])
# target gives the expected value of the output
# we will want to make the output be as close as possible to the target
# as with input it is a set of binary data
# in many neural nets the targets are a single column of values
# each value corresponds to the correct identification and is 1 for that answer, 0 if not
# This is not like that to better show the learning mechanism
target = np.array([[1, 1, 1, 1, 1], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]])
# The weights of each layer say how much of each layer should be added to the next
# We do this by matrix multiplication of the previous layer to get the next one
# weights1 say how much of each input value makes each hidden layer value
# Use randn to get probabilities based on a normal distribution
# That is values that are most likely to be near 0
# The weights must be dimensionally consistent for matrix multiplication
# That is the number of columns must match the number of input rows
# The number of rows will be the number of rows in the answer for the next layer (hidden)
weights1 = np.random.randn(HIDDEN_SIZE, np.size(data_in, 0))
# weights 2 is how much each hidden value is used to make the output
# For matrix multiplication the number of rows must be the number of output rows
# The number of columns must be the number of rows in the hidden layer
# Most neural nets also have a bias value to add on but for simplicity we don't have it here
weights2 = np.random.randn(np.size(target, 0), HIDDEN_SIZE)
# The error is the square of the difference of the target from the output of the neural net
# We sum up all these values for the total error
# The aim is to minimise the error
# This will take the error from each iteration of the loop
# We use this to monitor the performance of the learning mechanism to reduce the total error
total_error = []

# With each iteration we should be reducing the error
# That is the neural net will return values closer to what should be expected
# If working well, after several iterations it should get close to the optimal values
# This could be after several hundred iterations
for i in range(200):
    # Start with propagation.
    # Multiply weights to get the hidden and again to get the output
    # Use the sigmoid function on each result to keep values between 0 and 1
    hidden = np.matmul(weights1, data_in)
    hidden = sigmoid(hidden)
    output = np.matmul(weights2, hidden)
    output = sigmoid(output)

    # Find the error as the difference of the target and output
    # And the total error as the sum of these
    # Monitor this in graphs to see how many iterations are needed so the error is low enough
    error = (target-output)
    total_error.append(np.sum(error**2))

    # Backpropagation
    # Using calculus we see that the rate of change of the total error is
    # the error * sigmoid rate * the result of the weights applied to the previous layer
    # for matrix multiplication in backprop we need to swap rows and columns to get the
    # right matrix dimensions, that is use the transpose of the weights
    # We then add the changes to the weights to find values that reduce the error
    # we make changes to one layer and calculate the error
    # we then use this to find the error in the previous layer and change that layer
    change1 = error * rate(output)
    error2 = np.matmul(weights2.T, change1)
    weights2 += np.matmul(change1, hidden.T)
    change2 = error2 * rate(hidden)
    weights1 += np.matmul(change2, data_in.T)

# print the final output and see how close it is to the target
print(output)
# We can see how the total error changes with iterations
# but if this is a large number may take up too much screen space, so stick with graph
# print(total_error)

# basic line plot, but feel free to use many of matplotlibs features to improve the graph
pp.plot(total_error)
pp.show()