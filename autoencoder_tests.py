from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import pandas as pd

from NN import FullyConnectedNet
from autoencoder import Autoencoder
from solver import Solver
from layer import *

train = pd.read_csv('sample_mnist.csv')
train_x = train[list(train.columns)[1:]].values
train_y = train['label'].values

## normalize and reshape the predictors  
train_x = train_x / 255

## create train and validation datasets
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)

## reshape the inputs
train_x = train_x.reshape(-1, 784)
val_x = val_x.reshape(-1, 784)

##create autoencoder instance instace
data = {"X_train":train_x, "y_train":train_x}
model = Autoencoder(input_dim=784, hidden_dim=[1564, 392, 1564], num_classes=784)

##check layers
s= 'W0 : ' + str(model.params['W0'].shape) \
 + '\nW1 : ' + str(model.params['W1'].shape) \
 + '\nW2 : ' + str(model.params['W2'].shape) \
 + '\nW3 : ' + str(model.params['W3'].shape)
print(s)

##encoder layer
encode1, encode1_cache = forward_pass(train_x, model.params['W0'], model.params['b0'])
print("encode : " + str(encode1.shape))
relu1, relu1_cache = relu_forward(encode1)

#latent view
latent, latent_cache = forward_pass(relu1, model.params['W1'], model.params['b1'])
print("latent : " + str(latent.shape))
sigmoid = sigmoid_forward(latent) 

##decoder layer
decode1, decode1_cache = forward_pass(sigmoid, model.params['W2'], model.params['b2'])
print("decode : " + str(decode1.shape))
relu2, relu2_cache = relu_forward(decode1)

##output layer
scores, output_cache = forward_pass(relu2, model.params['W3'], model.params['b3'])
print("output : " + str(scores.shape))

##MSE test
loss, dMSE = MSE_loss(scores, train_x)
print("dMSE" + str(dMSE.shape))

##gradient descend
##output layer
dx, dw, db = backward_pass(dMSE, output_cache)
print("dw3" + str(dw.shape) + ", db3" + str(db.shape))

#Decoder gradients
drelu2 = relu_backward(dx, relu2_cache)
dx, dw, db = backward_pass(drelu2, decode1_cache)
print("dw2" + str(dw.shape) + ", db2" + str(db.shape))

#latent gradients
dsigmoid = sigmoid_backward(dx, sigmoid)
dx, dw, db = backward_pass(dsigmoid, latent_cache)
print("dw1" + str(dw.shape) + ", db1" + str(db.shape))

#Encoder gradients
drelu1 = relu_backward(dx, relu1_cache)
dx, dw, db = backward_pass(drelu1, encode1_cache)
print("dx" + str(dx.shape) + ", dw0" + str(dw.shape) + ", db0" + str(db.shape))
