#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    H1 = np.dot(data, W1) + b1
    A1 = sigmoid(H1)

    H2 = np.dot(A1, W2) + b2
    A2 = H2

    probs = softmax(A2)
    cost = -np.sum(labels * np.log(probs)) / labels.shape[0]
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    dcost_dA2 = probs.copy()
    dcost_dA2 -= labels
    dcost_dA2 /= labels.shape[0]

    dcost_dH2 = dcost_dA2

    dH2_dA1 = W2
    dH2_dW2 = A1
    dH2_db2 = np.ones(shape = (A1.shape[0], 1))

    gradW2 = np.dot(dcost_dH2.T, dH2_dW2).T
    gradb2 = np.dot(dcost_dH2.T, dH2_db2).T

    dcost_dA1 = np.dot(dcost_dH2, dH2_dA1.T)
    assert dcost_dA1.shape == A1.shape, str(dcost_dA1.shape) + " != " + str(dH2_dA1.shape)

    dA1_dH1 = sigmoid_grad(A1)
    dcost_dH1 = dcost_dA1 * dA1_dH1
    assert dcost_dH1.shape == H1.shape, str(dcost_dH1.shape) + " != " + str(H1.shape)

    dH1_ddata = W1
    dH1_dW1 = data
    dH1_db1 = np.ones(shape = (data.shape[0], 1))

    gradW1 = np.dot(dcost_dH1.T, dH1_dW1).T
    gradb1 = np.dot(dcost_dH1.T, dH1_db1).T

    assert gradW1.shape == W1.shape, str(gradW1.shape) + " != " + str(W1.shape)
    assert gradb1.shape == b1.shape, str(gradb1.shape) + " != " + str(b1.shape)
    assert gradW2.shape == W2.shape, str(gradW2.shape) + " != " + str(W2.shape)
    assert gradb2.shape == b2.shape, str(gradb2.shape) + " != " + str(b2.shape)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    for i in range(10):
       data = np.random.randn(1, 1)
       labels = np.array([[1, 0]])
       dims = [1, 1, 2]
       params = np.random.randn((dims[0] + 1) * dims[1] + (dims[1] + 1) * dims[2])
       gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dims), params)
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
