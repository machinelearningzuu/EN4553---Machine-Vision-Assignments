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

class CustomGradientUpdate(object):
    def __init__(self):
        train_ds, test_ds = prepare_dataset(map_fn_Q2, CIFAR10_data)
        self.train_ds = train_ds
        self.test_ds  = test_ds
        self.learning_rate = tf.constant(learning_rate)

    def initialize_weights(self):
        w1 = np.sqrt(2./n_features) * np.random.randn(n_features, hidden_dim)
        w2 = np.sqrt(2./hidden_dim) * np.random.randn(hidden_dim, n_classes)
        b1 = np.zeros(hidden_dim)
        b2 = np.zeros(n_classes)

        self.w1 = tf.Variable(tf.convert_to_tensor(w1, dtype=tf.float32))
        self.w2 = tf.Variable(tf.convert_to_tensor(w2, dtype=tf.float32))
        self.b1 = tf.Variable(tf.convert_to_tensor(b1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.convert_to_tensor(b2, dtype=tf.float32))

        prev_w1 = np.zeros((n_features, hidden_dim))
        prev_w2 = np.zeros((hidden_dim, n_classes))
        prev_b1 = np.zeros(hidden_dim)
        prev_b2 = np.zeros(n_classes)

        self.prev_w1 = tf.Variable(tf.convert_to_tensor(prev_w1, dtype=tf.float32))
        self.prev_w2 = tf.Variable(tf.convert_to_tensor(prev_w2, dtype=tf.float32))
        self.prev_b1 = tf.Variable(tf.convert_to_tensor(prev_b1, dtype=tf.float32))
        self.prev_b2 = tf.Variable(tf.convert_to_tensor(prev_b2, dtype=tf.float32))

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
        alphas = np.arange(0.1, 1, 0.1)
        fig = plt.figure(figsize =(12, 12)) 

        for i, alpha in enumerate(alphas):
            print("alpha : {}".format(round(float(alpha), 1)))
            self.alpha = tf.constant(round(float(alpha), 1))

            self.initialize_weights()

            Train_losses = []
            Test_losses = []
            train_batches = len(list(self.train_ds))
            test_batches = len(list(self.test_ds))

            for epoch in range(1, n_epoches+1):
                train_loss = []
                test_loss = []
                
                t0 = time()
                for X, T in self.train_ds:

                    self.w1.assign(tf.subtract(self.w1, (self.w1 - self.prev_w1) * self.alpha))
                    self.w2.assign(tf.subtract(self.w2, (self.w2 - self.prev_w2) * self.alpha))
                    self.b1.assign(tf.subtract(self.b1, (self.b1 - self.prev_b1) * self.alpha))
                    self.b2.assign(tf.subtract(self.b2, (self.b2 - self.prev_b2) * self.alpha))
                    self.prev_w1 = self.w1
                    self.prev_w2 = self.w2
                    self.prev_b1 = self.b1
                    self.prev_b2 = self.b2

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

                    train_loss.append(loss.numpy())

                for X, T in self.test_ds:

                    Y = self.forward_propogation(X)
                    loss = self.Loss(T, Y)
                    test_loss.append(loss.numpy())

                train_loss = sum(train_loss)/len(train_loss)
                test_loss = sum(test_loss)/len(test_loss)

                Train_losses.append(train_loss)
                Test_losses.append(test_loss)

                t1 = time()
                epoch_time = t1-t0
                print(" Epoch : {} , Epoch Time : {} , Train Loss : {} , Test Loss : {}".format(
                                                                                            epoch, 
                                                                                            round(epoch_time,3),
                                                                                            round(train_loss,3), 
                                                                                            round(test_loss,3)))
            
            Train_losses,Test_losses = np.array(Train_losses),np.array(Test_losses)
            # cum_train_loss = np.cumsum(Train_losses) / np.arange(1,n_epoches+1)
            # cum_test_loss = np.cumsum(Test_losses) / np.arange(1,n_epoches+1)

            plot_size = int(len(alphas) ** 0.5)

            fig.add_subplot(plot_size, plot_size, i+1)
            plt.plot(Train_losses, 'r', label='Training Loss')
            plt.plot(Test_losses, 'b', label='Test Loss')
            plt.title('alpha : {}'.format(round(float(alpha), 1)))
            # plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
        plt.savefig(loss_img.format('q3'))
        plt.show()

if __name__ == "__main__":
    model = CustomGradientUpdate()
    model.train_autoGrad()
