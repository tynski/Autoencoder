import numpy as np
from sklearn.metrics import mean_squared_error

class Solver(object):
    def __init__(self, model, data, **kwargs):
        """
        Required arguments:
        model - object to be executed
        data - dictionary of training data containing:
        'X_train'
        'y_train'

        Optional arguments:

        update_rule - how the weights are updated (default is sgd)
        learing_rate - value of learing_rate (default is 1e-2)
        num_epochs - number of epochs to run for during train (default is 10)
        """

        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.learning_rate = kwargs.pop('learning_rate', 1e-2)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        self.reset()

    def reset(self):
        self.train_acc_history = []
        self.train_error_history = []
        self.loss_history = []
        self.best_error = np.inf
        self.best_acc = 0.0
        self.best_params = {}

    def check_accuracy(self, X, y):
        score = self.model.loss(X)
        y_pred = np.argmax(score, axis=1)
        y_pred = np.hstack(y_pred)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    def check_MSE(self, X, y):
        output = self.model.loss(X)
        error = mean_squared_error(y, output)
        return error

    def sgd(self, w, dw, learning_rate):
        """
        Performs vanilla stochastic gradient descent.
        """
        w += (-1) * learning_rate * dw
        return w

    def train_step(self):
        # compute loss and gradient
        loss, grads = self.model.loss(self.X_train, self.y_train)
        self.loss_history.append(loss)

        # perform a parameter update
        for key, val in self.model.params.items():
            dw = grads[key]
            w = val
            next_w = self.sgd(w, dw, self.learning_rate)
            self.model.params[key] = next_w

    def train(self):
        """
        Train the model.
        """
        num_train = self.X_train.shape[0]
        num_iterations = self.num_epochs * num_train

        for i in range(num_iterations):
            self.train_step()

            epoch_end = (i + 1) % num_train == 0

            first_it = (i == 0)
            last_it = (i == num_iterations - 1)

            if first_it or last_it or epoch_end:
                #train_acc = self.check_accuracy(self.X_train, self.y_train)
                #self.train_acc_history.append(train_acc)
                train_error = self.check_MSE(self.X_train, self.y_train)
                self.train_error_history.append(train_error)
            """
            # keep track of best model
            if train_acc > self.best_acc:
                self.best_acc = train_acc
                for k, v in self.model.params.items():
                    self.best_params[k] = v.copy()
            """
            if train_error < self.best_error:
                self.best_error = train_error
                for k, v in self.model.params.items():
                    self.best_params[k] = v.copy()

        # at the end save the best model
        self.model.params = self.best_params
