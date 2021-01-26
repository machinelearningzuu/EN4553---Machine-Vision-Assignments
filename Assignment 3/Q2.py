import os
from time import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import logging
logging.getLogger('tensorflow').disabled = True

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from util import*
from variables import*

np.random.seed(seed)
tf.random.set_seed(seed)

class GradientTapeLayer2Model:
    def __init__(self):Assignment 3
        train_ds, test_ds = prepare_dataset(map_fn_Q2, CIFAR10_data)
        self.train_ds = train_ds
        self.test_ds  = test_ds
        self.learning_rate = tf.constant(learning_rate)
        self.TrainLoss = tf.keras.losses.CategoricalCrossentropy()
        self.TestLoss = tf.keras.losses.CategoricalCrossentropy()
        self.TrainEvaluation = tf.keras.metrics.SparseCategoricalAccuracy()
        self.TestEvaluation = tf.keras.metrics.SparseCategoricalAccuracy()

    def initialize_weights(self):
        # self.w1 = tf.Variable(
        #                     tf.random.normal(
        #                                 [n_features, hidden_dim], 
        #                                 stddev=std, 
        #                                 seed=seed,
        #                                 dtype=tf.float32)
        #                         )
                            
        # self.w2 = tf.Variable(
        #                     tf.random.normal(
        #                                 [hidden_dim, n_classes], 
        #                                 stddev=std, 
        #                                 seed=seed,
        #                                 dtype=tf.float32
        #                                 )
        #                         )

        # self.b1 = tf.Variable(
        #                 tf.zeros(
        #                         [hidden_dim], 
        #                         tf.float32
        #                         )
        #                     )

        # self.b2 = tf.Variable(
        #                 tf.zeros(
        #                         [n_classes], 
        #                         tf.float32
        #                         )
        #                     )

        w1 = np.sqrt(2./n_features) * np.random.randn(n_features, hidden_dim)
        w2 = np.sqrt(2./hidden_dim) * np.random.randn(hidden_dim, n_classes)
        b1 = np.zeros(hidden_dim)
        b2 = np.zeros(n_classes)

        self.w1 = tf.Variable(tf.convert_to_tensor(w1, dtype=tf.float32))
        self.w2 = tf.Variable(tf.convert_to_tensor(w2, dtype=tf.float32))
        self.b1 = tf.Variable(tf.convert_to_tensor(b1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.convert_to_tensor(b2, dtype=tf.float32))

    def forward_propogation(self, X):
        n1 = tf.matmul(X, self.w1) + self.b1
        A = tf.nn.relu(n1)
        Y = tf.matmul(A, self.w2) + self.b2
        return Y

    def Loss(self, T, Y):
        loss_term = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(tf.math.subtract(Y,T)), axis=-1))
        reg_term = reg * (tf.math.reduce_sum(self.w2 * self.w2) + tf.math.reduce_sum(self.w1 * self.w1))
        return loss_term + reg_term

    def train_autoGrad(self):
        self.initialize_weights()

        Train_losses = []
        Train_accuracy = []
        Test_losses = []
        Test_accuracy = []

        train_batches = len(list(self.train_ds))
        test_batches = len(list(self.test_ds))

        for epoch in range(1, n_epoches+1):
            train_loss = []
            train_acc = []

            test_loss = []
            test_acc = []    

            t0 = time()
            for X, T in self.train_ds:

                with tf.GradientTape() as tape:
                    Y = self.forward_propogation(X)
                    loss = self.Loss(T, Y)

                [dw1, dw2, db1, db2] = tape.gradient(
                                        loss, 
                                        [self.w1, self.w2, self.b1, self.b2])
                self.w1.assign_sub(dw1 * self.learning_rate)
                self.w2.assign_sub(dw2 * self.learning_rate)
                self.b1.assign_sub(db1 * self.learning_rate)
                self.b2.assign_sub(db2 * self.learning_rate)

                acc = self.evaluation(Y, T)
                train_loss.append(loss.numpy())
                train_acc.append(acc)
                # self.TrainEvaluation.update_state(T, Y)

            for X, T in self.test_ds:

                Y = self.forward_propogation(X)
                loss = self.Loss(T, Y)
                acc = self.evaluation(Y, T)
                test_loss.append(loss.numpy())
                test_acc.append(acc)
                # self.TestEvaluation.update_state(T, Y)

            train_loss = sum(train_loss)/len(train_loss)
            test_loss = sum(test_loss)/len(test_loss)
            train_acc = sum(train_acc)/len(train_acc)
            test_acc = sum(test_acc)/len(test_acc)
            # train_acc = self.TrainEvaluation.result().numpy()
            # test_acc = self.TestEvaluation.result().numpy()
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
                                                                                                                    round(test_acc,3)))
                                                               
            # self.TrainEvaluation.reset_states()
            # self.TestEvaluation.reset_states()
        visualize_cum_metrices(Train_losses,Test_losses,  Train_accuracy,Test_accuracy, 'q2')

    def evaluation(self, Y, T):
        Y = np.argmax(Y.numpy(), axis=-1)
        T = np.argmax(T.numpy(), axis=-1)
        return np.mean(Y==T)

if __name__ == "__main__":
    model = GradientTapeLayer2Model()
    model.train_autoGrad()
