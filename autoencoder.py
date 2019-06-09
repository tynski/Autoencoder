import numpy as np
from layer import *

class Autoencoder(object):
    """
    The architecure of autoencoder looks like this:
    input -> relu -> sigmoid -> relu -> ouput

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=4, hidden_dim=2, num_classes=4,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.no_dims = len(hidden_dim)

        for idx, val in enumerate(hidden_dim):
            self.params['W'+str(idx)] = np.random.normal(0, weight_scale, [input_dim, val])
            self.params['b'+str(idx)] = np.zeros([val])
            input_dim = val
      
        self.params['W' + str(self.no_dims)] = np.random.normal(0, weight_scale, [input_dim, num_classes])
        self.params['b' + str(self.no_dims)] = np.zeros([num_classes])


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        loss, grads = 0, {}

        # encoder
        encode1, encode1_cache = forward_pass(X, self.params['W0'], self.params['b0']) 
        relu1, relu1_cache = relu_forward(encode1)

        #encode2, encode2_cache = forward_pass(relu1, self.params['W1'], self.params['b1']) 
        #relu2, relu2_cache = relu_forward(encode2)

        #latent view
        latent, latent_cache = forward_pass(relu1, self.params['W1'], self.params['b1'])
        sigmoid = sigmoid_forward(latent)

        #decoder
        decode1, decode1_cache = forward_pass(sigmoid, self.params['W2'], self.params['b2'])
        relu2, relu2_cache = relu_forward(decode1)

        ##output layer
        scores, output_cache = forward_pass(relu2, self.params['W3'], self.params['b3'])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        
        loss, dMSE = MSE_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.square(self.params['W3'])))

        ##gradient descend
        ##output layer
        dx, dw, db = backward_pass(dMSE, output_cache)

        #Store gradients of the output
        grads['W3'] = dw + self.reg * self.params['W3']
        grads['b3'] = db
        
        #Decoder gradients
        drelu2 = relu_backward(dx, relu2_cache)
        dx, dw, db = backward_pass(drelu2, decode1_cache)

        grads['W2'] = dw + self.reg * self.params['W2']
        grads['b2'] = db

        #latent gradients
        dsigmoid = sigmoid_backward(dx, sigmoid)
        dx, dw, db = backward_pass(dsigmoid, latent_cache)

        grads['W1'] = dw + self.reg * self.params['W1']
        grads['b1'] = db

        #Encoder gradients
        drelu1 = relu_backward(dx, relu1_cache)
        dx, dw, db = backward_pass(drelu1, encode1_cache)

        grads['W0'] = dw + self.reg * self.params['W0']
        grads['b0'] = db

        return loss, grads
