import os
from time import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import logging
logging.getLogger('tensorflow').disabled = True

import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from util import*
from variables import*

np.random.seed(seed)
tf.random.set_seed(seed)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Layer2NeuralNetwork(object):
    def __init__(self):
        train_ds, test_ds = prepare_dataset(map_fn_Q1, CIFAR10_data)
        self.train_ds = train_ds
        self.test_ds  = test_ds
        self.learning_rate = learning_rate

    def initialize_weights(self):
        self.w1 = np.sqrt(2./n_features) * np.random.randn(n_features, hidden_dim)
        self.w2 = np.sqrt(2./hidden_dim) * np.random.randn(hidden_dim, n_classes)
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(n_classes)

    @staticmethod
    def sigmoid(z):
        return 1/(1 + tf.math.exp(-z))

    @staticmethod
    def relu(z):
        return z * tf.cast(z > 0, tf.float32)

    @staticmethod
    def softmax(z):
        return tf.math.exp(z) / (tf.math.reduce_sum(tf.math.exp(z), axis=-1))

    @staticmethod
    def derivative_of_sigmoid(z):
        d1 = Layer2NeuralNetwork.sigmoid(z)
        d2 = Layer2NeuralNetwork.sigmoid(z) ** 2
        return  d1 - d2 

    @staticmethod
    def derivative_of_relu(z):
        return tf.cast(z > 0, tf.float32)

    def forward_propogation(self, X):
        n1 = tf.matmul(X, self.w1) + self.b1
        A = Layer2NeuralNetwork.relu(n1)
        Y = tf.matmul(A, self.w2) + self.b2
        return Y, A

    def Loss(self, Y, T):
        loss_term = (1./len(Y))*np.square(Y - T).sum()
        reg_term = reg*(np.sum(self.w2*self.w2) + np.sum(self.w1*self.w1))
        return loss_term + reg_term

    def backward_propogation(self, X, Y, A, T):        
        dY = 1./batch_size*2.0*(Y - T)
        self.dw2 = tf.matmul(tf.transpose(A), dY) + reg*self.w2   
        self.db2 = tf.math.reduce_sum(dY, axis=0)
        dA = tf.matmul(dY, tf.transpose(self.w2))
        self.dw1 = tf.matmul(tf.transpose(X), dA*Layer2NeuralNetwork.derivative_of_relu(A)) + reg*self.w1
        self.db1 = tf.math.reduce_sum(dA*Layer2NeuralNetwork.derivative_of_relu(A), axis=0)

    def gradient_descent(self):
        self.w2 = self.w2 - learning_rate * self.dw2
        self.w1 = self.w1 - learning_rate * self.dw1
        self.b2 = self.b2 - learning_rate * self.db2
        self.b1 = self.b1 - learning_rate * self.db1

    def lr_decay(self):
        self.learning_rate = self.learning_rate* learning_rate_decay

    def cosine_lr_decay(self, step):
        self.learning_rate = 0.5 * learning_rate * (1 + np.cos(step * np.pi / n_epoches))

    def evaluation(self, X, T):
        Y, A = self.forward_propogation(X)
        Y = np.argmax(Y, axis=-1)
        T = np.argmax(T, axis=-1)
        return np.mean(Y==T)

    def train_model(self):
        self.initialize_weights()
        Train_losses = []
        Train_accuracy = []
        Test_losses = []
        Test_accuracy = []

        train_batches = len(list(self.train_ds))
        test_batches = len(list(self.test_ds))

        for epoch in range(1, n_epoches+1):
            train_loss = 0
            train_acc = 0

            test_loss = 0
            test_acc = 0           

            t0 = time()
            for X, T in self.train_ds:

                X = X.numpy()
                T = T.numpy()

                Y, A = self.forward_propogation(X)
                loss = self.Loss(Y, T)
                acc = self.evaluation(X, T)
                train_loss += loss
                train_acc += acc

                self.backward_propogation(X, Y, A, T)
                self.gradient_descent()

            # self.lr_decay()
            # self.cosine_lr_decay(epoch+1)

            for X, T in self.test_ds:

                X = X.numpy()
                T = T.numpy()

                Y, A = self.forward_propogation(X)
                loss = self.Loss(Y, T)
                acc = self.evaluation(X, T)
                test_loss += loss
                test_acc += acc

            train_loss = train_loss/train_batches
            test_loss = test_loss/test_batches
            train_acc = train_acc/train_batches
            test_acc = test_acc/test_batches

            Train_losses.append(train_loss)
            Test_losses.append(test_loss)
            Train_accuracy.append(train_acc)
            Test_accuracy.append(test_acc)

            t1 = time()
            epoch_time = t1-t0
            print(" Epoch : {} , Epoch Time : {} , Train Loss : {} , Train Acc : {} Test Loss : {} Test Acc : {}".format(
                                                                                                                    epoch, 
                                                                                                                    round(epoch_time,3),
                                                                                                                    round(train_loss,3), 
                                                                                                                    round(train_acc,3), 
                                                                                                                    round(test_loss,3), 
                                                                                                                    round(test_acc,3)
                                                                                                                    ))

        visualize_cum_metrices(Train_losses,Test_losses,  Train_accuracy,Test_accuracy, 'q1')
if __name__ == "__main__":
    model = Layer2NeuralNetwork()
    model.train_model()
