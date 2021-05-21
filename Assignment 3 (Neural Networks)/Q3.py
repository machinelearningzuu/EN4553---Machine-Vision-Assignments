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

    def forward_propogation(self, X):
        n1 = tf.matmul(X, self.w1) + self.b1
        A = tf.nn.relu(n1)
        Y = tf.matmul(A, self.w2) + self.b2
        return Y

    def Loss(self, T, Y):
        loss_term = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(tf.math.subtract(Y,T)), axis=-1))
        reg_term = reg * (tf.math.reduce_sum(self.w2 * self.w2) + tf.math.reduce_sum(self.w1 * self.w1))
        return loss_term + reg_term

    def initialize_weights(self):
        w1 = np.sqrt(2./n_features) * np.random.randn(n_features, hidden_dim)
        w2 = np.sqrt(2./hidden_dim) * np.random.randn(hidden_dim, n_classes)
        b1 = np.zeros(hidden_dim)
        b2 = np.zeros(n_classes)

        self.w1_t1 = tf.Variable(tf.convert_to_tensor(w1, dtype=tf.float32))
        self.w2_t1 = tf.Variable(tf.convert_to_tensor(w2, dtype=tf.float32))
        self.b1_t1 = tf.Variable(tf.convert_to_tensor(b1, dtype=tf.float32))
        self.b2_t1 = tf.Variable(tf.convert_to_tensor(b2, dtype=tf.float32))

    def train_autoGrad(self):
        alphas = np.arange(0.1, 1, 0.1)
        fig = plt.figure(figsize =(12, 12)) 

        for i, alpha in enumerate(alphas):
            print("alpha : {}".format(round(float(alpha), 1)))
            self.alpha = tf.constant(round(float(alpha), 1))
            self.initialize_weights()

            losses = []
            X, T = self.train_ds
            for epoch in range(1, n_epoches+1):
                self.w1_t = self.w1_t1
                self.w2_t = self.w2_t1
                self.b1_t = self.b1_t1
                self.b2_t = self.b2_t1

                with tf.GradientTape() as tape:
                    Y = self.forward_propogation(X)
                    train_loss = self.Loss(T, Y)

                [dw1, dw2, db1, db2] = tape.gradient(
                                        train_loss, 
                                    [self.w1_t1, self.w2_t1, self.b1_t1, self.b2_t1])

                self.w1_t1.assign_sub(dw1 * self.learning_rate)
                self.w2_t1.assign_sub(dw2 * self.learning_rate)
                self.b1_t1.assign_sub(db1 * self.learning_rate)
                self.b2_t1.assign_sub(db2 * self.learning_rate)

                if epoch>=1:    
                    self.w1 = (1-alpha)*self.w1_t + alpha*self.w1_t1
                    self.w2 = (1-alpha)*self.w2_t + alpha*self.w2_t1
                    self.b1 = (1-alpha)*self.b1_t + alpha*self.b1_t1
                    self.b2 = (1-alpha)*self.b2_t + alpha*self.b2_t1

                    y_pred = self.forward_propogation(X)
                    loss = self.Loss(T, Y)
                    losses.append(loss)
                    print('epoch {} / {}: loss {}'.format(epoch-1, n_epoches, loss))

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