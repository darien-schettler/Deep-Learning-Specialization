import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

"""
Convolutional Neural Networks: Step by Step
Implement convolutional (CONV) and pooling (POOL) layers in numpy, including both forward prop and back prop

------------------------
Notation:

Superscript  [l]  denotes an object of the  lth  layer
    --> Example:  a[4]a[4]  is the  4th4th  layer activation.  W[5]  and  b[5]  are the  5th  layer parameters

Superscript  (i)  denotes an object from the  ith  example
    --> Example:  x(i)  is the  ith  training example input

Lowerscript  i  denotes the  ith  entry of a vector.
    --> Example:  a[l][i]  denotes the  ith  entry of the activations in layer  l
    ------> Assuming this is a fully connected (FC) layer

n[H],  n[W],  and  n[C]  denote respectively the height, width and number of channels of a given layer
--> If you want to reference a specific layer  l, you can also write  n[l][H],  n[l][W],  n[l][C]

n[Hprev],  n[Wprev],  and  n[Cprev]  denote respectively the height, width and number of channels of the previous layer
--> If referencing a specific layer  l, you can also write  n[l−1][H],  n[l−1][W],  n[l−1][C]

We assume that you are already familiar with numpy
Let's get started!
------------------------

"""

'''
*************** Packages ***************
Let's first import all the packages that you will need during this assignment.

1. numpy --                 is the fundamental package for scientific computing with Python.
2. matplotlib --            is a library to plot graphs in Python.
3. np.random.seed(1) --     is used to keep all the random function calls consistent. It will help us grade your work.

--DONE ABOVE--
'''

'''
-------------- OUTLINE OF WHAT WE ARE ABOUT TO DO --------------
You will be implementing the building blocks of a convolutional neural network! 
Each function you will implement will have detailed instructions that will walk you through the steps needed...

1. Convolution functions, including:
--> Zero Padding
--> Convolve window
--> Convolution forward
--> Convolution backward

2. Pooling functions, including:
--> Pooling forward
--> Create mask
--> Distribute value
--> Pooling backward

This program will ask you to implement these functions from scratch in numpy
In the next program, you will use the TensorFlow equivalents of these functions to build the following model...
              -----------------------------------------
---------     | ---------     ---------     --------- |        ---------     -----------
| INPUT | --> | | CONV. | --> | RELU  | --> | POOL. | | x2 --> |   FC  | --> | SOFTMAX | 
---------     | ---------     ---------     --------- |        ---------     -----------
              ----------------------------------------- 

Note:
-- For every forward function, there is its corresponding backward equivalent
-- Hence, at every step of your forward module you will store some parameters in a cache
-- These parameters are used to compute gradients during backpropagation

'''

'''
ZERO-PADDING
--> Zero Padding means you add zeros around the border of an image (not NO PADDING)

                       0 0 0 0 0 0 0 0 0
                       0 0 0 0 0 0 0 0 0
    1 2 3 4 5  ----->  0 0 1 2 3 4 5 0 0
    1 2 3 4 5  ----->  0 0 1 2 3 4 5 0 0
    1 2 3 4 5  ----->  0 0 1 2 3 4 5 0 0
    1 2 3 4 5  ----->  0 0 1 2 3 4 5 0 0
                       0 0 0 0 0 0 0 0 0
                       0 0 0 0 0 0 0 0 0

------------------------------------------------
The main benefits of padding are the following:
------------------------------------------------

1. It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes
----> This is important for building deeper networks, since otherwise the height/width would shrink as you go deeper
----> An important special case is the "same" convolution, in which the height/width is exactly preserved after 1 layer

2. It helps us keep more of the information at the border of an image
----> Without padding, very few values at the next layer would be affected by pixels as the edges of an image

------------------------------------------------

Exercise: Implement the following function, which pads all the images of a batch of examples X with zeros
---> Use np.pad
------> If you want to pad the array "a" of shape  (5,5,5,5,5)  
----------> with pad = 1 for the 2nd dimension
----------> with pad = 3 for the 4th dimension 
----------> with pad = 0 for the rest, you would do:
---------------> a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), 'constant', constant_values = (..,..))

'''


# implement a function to create zero_pad

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    # This is a complicated line coming below utilizing np.pad:

    # The first value, X is the array/matrix we wish to pad
    # The second value, (tuple of 0s - no pad) is how much padding we want on the FIRST dim of the array (Number of Ex.)
    # The third value, (tuple of pads) is how much padding we want on the SECOND dim of the array (Height)
    # The fourth value, (tuple of pads) is how much padding we want on the THIRD dim of the array (WIDTH)
    # The fourth value, (tuple of 0s - no pad) is how much padding we want on the FOURTH dim of the array (Channels)
    # The fifth value, 'constant' refers to what type of padding we want (constant value padding in this case)
    # The last value, constant_values=(0,0) refers to the constant we want to use, left is 1st val & right is 2nd val

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))

    return X_pad


# Now we will see it in action...
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print("\nx.shape =", x.shape)
print("x_pad.shape =", x_pad.shape)
print("\nx[1,1] =\n", x[1, 1])
print("\nx_pad[1,1] =\n", x_pad[1, 1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
plt.show()

'''
SINGLE STEP OF CONVOLUTION

In this part, implement a single step of convolution, in which you apply the filter to a single position of the input

This will be used to build a convolutional unit, which:

1. Takes an input volume
2. Applies a filter at every position of the input
3. Outputs another volume (usually of different size)

In a computer vision application, each value in the matrix on the left corresponds to a single pixel value
To convolve the image we apply a  3x3 filter using the following steps 
-- 1. multiplying its values element-wise with the original matrix
-- 2. summing them up
-- 3. adding a bias

In this first step of the exercise, you will implement a single step of convolution
----> corresponding to applying a filter to just one of the positions to get a single real-valued output

Later in this program you'll apply this function to multiple positions of the input to implement the full conv operation

Exercise: Implement conv_single_step()
'''


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = a_slice_prev * W

    # Sum over all entries of the volume s.
    Z = np.sum(s)

    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z += b

    return Z


'''
CNN FORWARD PASS

In the forward pass, you will take many filters and convolve them on the input
--> Each 'convolution' gives you a 2D matrix output
--> You will then stack these outputs to get a 3D volume:

Exercise: Implement the function below to convolve the filters W on an input activation A_prev
--> This function takes as input.. 
-----> A_prev
-----> the activations output by the previous layer (for a batch of m inputs)
-----> F filters/weights denoted by W
-----> and a bias vector denoted by b
----------> where each filter has its own (single) bias

Finally you also have access to the hyperparameters dictionary which contains the stride and the padding

Hint:

1. To select a 2x2 slice at the upper left corner of a matrix "a_prev" (shape (5,5,3)), you would do:
------> a_slice_prev = a_prev[0:2,0:2,:]
------> This will be useful when you will define a_slice_prev below, using the start/end indexes you will define

2. To define a_slice you will need to first define its corners vert_start, vert_end, horiz_start and horiz_end

Reminder: The formulas relating the output shape of the convolution to the input shape is:
------------------------------------------------------------------------
1.      n[H] = floor { ( n[Hprev] − f + 2 × pad ) / ( stride ) }+1
----------
2.      n[W] = floor { ( n[Wprev] − f + 2 × pad ) / ( stride ) }+1 
----------
Note:   n[C] = number of filters used in the convolution
------------------------------------------------------------------------
 
For this exercise, we won't worry about vectorization, and will just implement everything with for-loops.

'''


# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Compute the dimensions of the CONV output volume using the formula given above
    # Hint: use int() to floor. (≈2 lines) -- probably only need the int on the outside ...
    n_H = int((n_H_prev - f + (2 * pad)) / stride) + 1
    n_W = int((n_W_prev - f + (2 * pad)) / stride) + 1

    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev calling function zero_pad(X, pad)
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # loop over the batch of training examples (i is 1 for 1st example)
        a_prev_pad = A_prev_pad[i]  # Select ith training example's padded activation
        for h in range(n_H):  # loop over vertical axis of the output volume (h is 1 for 1st row of pixels)
            for w in range(n_W):  # loop over horizontal axis of the output volume (w is 1 for 1st col of pixels)
                for c in range(n_C):  # loop over channels (= #filters) of the output volume (c is 1 for 1st channel)

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = (h * stride) + f  # Could use vert_start instead of (h*stride)
                    horiz_start = w * stride
                    horiz_end = (w * stride) + f  # Could use horiz_start instead of (w*stride)

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


# Test one step of convolving forward

# ---------------------------------------------------------------------------------------------------------------------

A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad": 2,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("\n------------------------------------------------------------------------------------------------------------"
      "\nCONV 1 STEP FORWARD"
      "\n------------------------------------------------------------------------------------------------------------"
      "\n\nZ's mean =", np.mean(Z))
print("\nZ[3,2,1] =\n", Z[3, 2, 1])
print("\ncache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])

'''

***NOTE***

The CONV layer should also contain an activation, in which case we would add the following line of code:
---> Convolve the window to get back one output neuron
------> Z[i, h, w, c] = Applied Activation... (shown on next line)
------> A[i, h, w, c] = activation(Z[i, h, w, c])

You don't need to do it here.
'''

# ---------------------------------------------------------------------------------------------------------------------

'''
POOLING LAYER

The pooling (POOL) layer reduces the height and width of the input
It helps reduce computation, as well as helps make feature detectors more invariant to its position in the input

The two types of pooling layers are:

1. Max-pooling layer        : slides an ( f,f ) window over the input & stores the max value of the window as output
2. Average-pooling layer    : slides an ( f,f ) window over the input & stores the average value of the window as output

These pooling layers have no parameters for backpropagation to train
However, they have hyperparameters such as the window size  f
---> ** f ** ---> This specifies the height and width of the fxf window you would compute a max or average over

*******************
  FORWARD POOLING
*******************

Now, you are going to implement MAX-POOL and AVG-POOL, in the same function

Exercise: Implement the forward pass of the pooling layer

Reminder: As there's no padding, the formulas binding the output shape of the pooling to the input shape is:

1. n[H] = floor{ ( n[Hprev] − f ) / ( stride ) } + 1 
2. n[W] = floor{ ( n[Wprev] − f ) / ( stride ) } + 1 
3. n[C] = n[Cprev]

'''


# GRADED FUNCTION: pool_forward

def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """

    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyper-parameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):  # loop over the training examples
        for h in range(n_H):  # loop on the vertical axis of the output volume
            for w in range(n_W):  # loop on the horizontal axis of the output volume
                for c in range(n_C):  # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = (h * stride) + f  # Could use vert_start instead of (h*stride)
                    horiz_start = w * stride
                    horiz_end = (w * stride) + f  # Could use horiz_start instead of (w*stride)

                    # Use the corners to define the current slice on the ith training example of A_prev, channel c
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache


# Test one step of pooling forward

# ---------------------------------------------------------------------------------------------------------------------

A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride": 2, "f": 3}

A, cache = pool_forward(A_prev, hparameters)

print("\n------------------------------------------------------------------------------------------------------------"
      "\nPOOLING 1 STEP FORWARD"
      "\n------------------------------------------------------------------------------------------------------------")

print("\n---------------")
print("mode = max")
print("---------------")
print("A =\n", A)

A, cache = pool_forward(A_prev, hparameters, mode="average")
print("\n---------------")
print("mode = average")
print("---------------")
print("A =\n", A)

# ---------------------------------------------------------------------------------------------------------------------

'''
****************************************************
  BACKPROPOGATION IN CONVOLUTIONAL NEURAL NETWORKS
****************************************************

In modern deep learning frameworks, you only have to implement the forward pass...
.. the framework takes care of the backward pass... 
.. so most deep learning engineers don't need to bother with the details of the backward pass

The backward pass for convolutional networks is complicated

If you wish however, you can work through this part of the program to get a sense of what backprop in a CNN looks like

Before, in a simple (FC) NN, you used backprop to compute the derivatives w.r.t the cost to update the parameters
Similarly, in CNNs you can to calculate the derivatives with respect to the cost in order to update the parameters

The backprop equations are not trivial and we did not derive them in lecture, but we briefly presented them below

Let's start by implementing the backward pass for a CONV layer

*****************
  Computing dA:
*****************

This is the formula for computing  dA  w.r.t the cost for a certain filter  Wc  and a given training example (m) :

First Sum    <<< FROM: h=0 ~~ TO: n[H] >>>
Second Sum   <<< FROM: w=0 ~~ TO: n[W] >>>

~~~~ dA += ∑∑ W[c] × dZ[h][w] ~~~~
 
Where:  
--> Wc  is a filter
--> Zhw  is a scalar corresponding to the gradient of the cost w.r.t the output of the conv layer...
-->.. Z at the hth row & wth column (corresponding to the dot product taken at the ith stride left and jth stride down
-----> NOTE:
--------> That at each time, we multiply the the same filter  Wc  by a different dZ when updating dA
--------> We do so because when computing the forward prop, each filter is dotted and summed by a different a_slice
--------> Therefore when computing the backprop for dA, we are just adding the gradients of all the a_slices

In code, inside the appropriate for-loops, this formula translates into:

da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]

*****************
  Computing dW:
*****************

This is the formula for computing  dWc  ( dWc  is the derivative of one filter ) w.r.t the loss:

First Sum    <<< FROM: h=0 ~~ TO: n[H] >>>
Second Sum   <<< FROM: w=0 ~~ TO: n[W] >>>

~~~~ dWc += ∑∑ a[slice] × dZ[h][w] ~~~~
 
Where:
--> a[slice]  corresponds to the slice which was used to generate the activation  Z[i][j]

Hence, this ends up giving us the gradient for  W  w.r.t that slice
Since it is the same  W, we will just add up all such gradients to get  dW

In code, inside the appropriate for-loops, this formula translates into:

dW[:,:,:,c] += a_slice * dZ[i, h, w, c]

*****************
  Computing db:
*****************

This is the formula for computing  db  w.r.t the cost for a certain filter Wc :


First Sum    <<< FROM: h=0 ~~ TO: n[H] >>>
Second Sum   <<< FROM: w=0 ~~ TO: n[W] >>>

~~~~ db = ∑∑ dZ[h][w] ~~~~
 
As you have previously seen in basic neural networks, db is computed by summing  dZ
In this case, you are just summing over all the gradients of the conv output (Z) w.r.t the cost

In code, inside the appropriate for-loops, this formula translates into:

db[:,:,:,c] += dZ[i, h, w, c]

Exercise: Implement the conv_backward function below

1. You should sum over all the training examples, filters, heights, and widths
2. You should then compute the derivatives using formulas above

'''


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):  # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):  # loop over vertical axis of the output volume
            for w in range(n_W):  # loop over horizontal axis of the output volume
                for c in range(n_C):  # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the un-padded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    # Making sure your output shape is correct
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db


# TEST BACKWARD PASS OF CONV LAYER

# --------------------------------------------------------------------------------------------------------------------

print("\n-------------------------------------------------------------------------------------------")
print("BACKWARD PASS OF CONV LAYER TEST")
print("-------------------------------------------------------------------------------------------\n")

dA, dW, db = conv_backward(Z, cache_conv)

print("dA_mean =\n", np.mean(dA))
print("\ndW_mean =\n", np.mean(dW))
print("\ndb_mean =\n", np.mean(db))
print("\n-------------------------------------------------------------------------------------------")

# --------------------------------------------------------------------------------------------------------------------

'''
BACKWARD POOLING PASS

Next, let's implement the backward pass for the pooling layer, starting with the MAX-POOL layer

A pooling layer has no parameters for backprop to update
However, you backprop the gradient through the pooling layer to compute gradients for layers that came before it

Before jumping into the backpro of the pooling layer, you are going to build a helper function
--> The helper function is called create_mask_from_window() which does the following:

                  [ 1 3 4 ]               [ 0 0 0 ]
            X  =  [ 2 2 7 ]     ----->    [ 0 0 1 ]
                  [ 1 4 0 ]               [ 0 0 0 ]
 
As you can see, this function creates a "mask" matrix which keeps track of where the maximum of the matrix is
--> True    (1) indicates the position of the maximum in X, the other entries are...
--> False   (0) 

You'll see later that the backward pass for average pooling will be similar to this but using a different mask

Exercise: Implement create_mask_from_window()

This function will be helpful for pooling backward

------------
   Hints:
------------

1. np.max() may be helpful... It computes the maximum of an array

2. If you have a matrix X and a scalar x: A = (X == x) will return a matrix A of the same size as X such that:
-----> A[i,j] = True    if X[i,j] == x
-----> A[i,j] = False   if X[i,j] != x

3. Here, you don't need to consider cases where there are several maxima in a matrix

------------

'''


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """

    mask = x == np.max(x)

    return mask


# TEST MASK

x = np.random.randn(2, 3)
mask = create_mask_from_window(x)

print('\n---MASK TEST---\n\n'
      'x =\n',
      x)

print("\nmask =\n",
      mask)

# Looks Good !

# ------ ??? NOTE ------
# Why do we keep track of the position of the max?

# It's because this is the input value that ultimately influenced the output, and therefore the cost
# Backprop is computing grads w.r.t the cost, so anything that influences the ultimate cost should have a non-zero grad
# So, backprop will "propagate" the gradient back to this particular input value that had influenced the cost

# ------ END NOTE ------

'''
AVERAGE POOLING BACKWARD PASS

In max pooling, for each input window, all the "influence" on the output came from a single input value--the max
In average pooling, every element of the input window has equal influence on the output
So to implement backprop, you will now implement a helper function that reflects this

i.e. If we did average pooling in the forward pass using a 3x3 filter, then the mask for the backward pass will be:
 
                                   [ 1/9  1/9  1/9 ]
    dZ  =  1    -->    dZ    =     [ 1/9  1/9  1/9 ]
                                   [ 1/9  1/9  1/9 ]

This implies that each position in the  dZ  matrix contributes equally to output (since forward pass used an average)

Exercise: Implement the function below to equally distribute a value dz through a matrix of dimension shape

'''


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape

    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """

    # Retrieve dimensions from shape
    (n_H, n_W) = shape

    # Compute the value to distribute on the matrix
    average = dz / (n_H * n_W)

    # Create a matrix where every entry is the "average" value
    a = average * np.ones(shape)

    return a


# TEST DISTRIBUTED VALUES FOR AVERAGE POOLING

a = distribute_value(2, (2, 2))
print('\n\n---DISTRIBUTE TEST---\n\n'
      'distributed value =\n',
      a)

'''
PUT IT ALL TOGETHER -- BACKWARD POOLING

You now have everything you need to compute backward propagation on a pooling layer.

Exercise: Implement the pool_backward function in both modes ("max" and "average")

You will once again use 4 for-loops (iterating over training examples, height, width, and channels)

You should use an if/elif statement to see if the mode is equal to 'max' or 'average'
--> If equal to 'average' use the distribute_value() fn shown above to create a matrix of the same shape as a_slice
--> If the mode is equal to 'max'... use fn create_mask_from_window() & multiply it by the corresponding value of dZ


'''


def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """

    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache

    # Retrieve hyper-parameters from "hparameters"
    stride = hparameters["stride"]
    f = hparameters["f"]

    # Retrieve dimensions from A_prev's shape and dA's shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):  # loop over the training examples

        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]

        for h in range(n_H):  # loop on the vertical axis
            for w in range(n_W):  # loop on the horizontal axis
                for c in range(n_C):  # loop over the channels (depth)

                    # Find the corners of the current "slice"

                    vert_start = h * stride
                    vert_end = vert_start + f

                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Compute the backward propagation in both modes
                    if mode == "max":

                        # Use the corners and "c" to define the current slice from a_prev
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                        # Create the mask from a_prev_slice
                        mask = create_mask_from_window(a_prev_slice)

                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                        dA_prev[i, vert_start:vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]

                    elif mode == "average":

                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]

                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)

                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    # Making sure your output shape is correct
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


# TEST POOLING

np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride": 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)

dA_prev = pool_backward(dA, cache, mode="max")
print("\n\n-----FULL TEST-----")

print("\n--------------------\n  mode = max  \n--------------------")
print('\nmean of dA =\n',
      np.mean(dA))

print('\ndA_prev[1,1] =\n',
      dA_prev[1, 1])

dA_prev = pool_backward(dA, cache, mode="average")
print("\n--------------------\n  mode = average  \n--------------------")
print('\nmean of dA =\n',
      np.mean(dA))
print('\ndA_prev[1,1] =\n',
      dA_prev[1, 1])

'''

ALL DONE!!

THIS WAS HOW TO COMPUTE CNNS USING ONLY NUMPY THE NEXT PROGRAM WILL FOCUS ON SIMILAR APPLICATIONS USING TENSORFLOW

'''
