import numpy as np
from sklearn.metrics import mean_squared_error


def forward_pass(x, w, b):
    out = None
    out = np.dot(x, w) + b
    cache = (x, w, b)
    return out, cache


def backward_pass(dout, cache):
    x, w, _ = cache
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)
    dw = np.dot(x.T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx = None
    x = cache

    mask = (x >= 0)
    dx = dout * mask

    return dx

def sigmoid_forward(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid

def sigmoid_backward(dout, cache):
    sigmoid = cache
    dx = (sigmoid * (1. - sigmoid)) * dout 
    return dx

def sigmoid_loss(x, y):
    h = sigmoid_forward(x)[0]
    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    gradient = h * (1. - h)
    return loss, gradient


def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def MSE_loss(scores, y):
    error = (y - scores)
    gradient = -1.0 * error / y.shape[0]
    loss = np.mean(np.square(error)) * 1/2
    return loss, gradient


