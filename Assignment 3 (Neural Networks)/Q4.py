import os
import itertools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import logging
logging.getLogger('tensorflow').disabled = True

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input, Dropout, BatchNormalization
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from util import*
from variables import*

np.random.seed(seed)
tf.random.set_seed(seed)

class MnistCNN:
    def __init__(self):
        train_ds, test_ds = prepare_dataset(map_fn_Q4, MNIST_data)
        self.train_ds = train_ds
        self.test_ds  = test_ds

    def classifier(self):
        inputs = Input(shape=input_shape)
        x = Conv2D(32, (3,3), activation='relu')(inputs)
        x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Flatten()(x)
        outputs = Dense(10, activation='softmax')(x)
        self.model = Model(inputs, outputs)
        self.model.summary()

        # self.model.compile(
        #     loss='sparse_categorical_crossentropy',
        #     optimizer=Adam(),
        #     metrics=['accuracy'],
        #                  )

        # self.history = self.model.fit(
        #                     self.train_ds,
        #                     epochs=2,
        #                     validation_data=self.test_ds
        #                     )

    def TFlearn_classifier(self):
        Xtrain , Ytrain, Xval, Yval, Xtest , Ytest = TFlearn_dataset(map_fn_Q4, MNIST_data)

        # self.model.trainable = False
        inputs = self.model.input
        x = self.model.layers[-2].output
        outputs = Dense(1, activation='sigmoid', name='sigmoid_output')(x)
        self.model2 = Model(inputs, outputs)
        self.model2.summary()

        self.model2.compile(
            loss='binary_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy'],
                        )

        self.model2.fit(Xtrain,Ytrain,
                        epochs=2,
                        batch_size=32,
                        validation_data=[Xval, Yval]
                       )

        self.plot_confusion_matrix(Xtest , Ytest)
        self.model2.evaluate(Xtest , Ytest)

    def train(self):
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy'],
        )

        self.history = self.model.fit(
                            self.train_ds,
                            epochs=2,
                            validation_data=self.test_ds
                            )

    def save_model(self):
        model_json = self.model.to_json()
        with open(model_architecure, "w") as json_file:
            json_file.write(model_json)
        self.model.save(model_weights)

    def load_model(self):
        json_file = open(model_architecure, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(model_weights)

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy'],
        )

    def plot_confusion_matrix(self, X, Y, cmap=None, normalize=True):
    # def plot_confusion_matrix(self, eval_data, cmap=None, normalize=True):
        # X = []
        # Y = []
        # eval_data = tfds.as_numpy(eval_data)  # Convert `tf.data.Dataset` to Python generator
        # for ex in eval_data:
        #     X.extend(ex[0].tolist())
        #     Y.extend(ex[1].tolist())
        # X = np.array(X)
        # Y = np.array(Y)

        P = self.model2.predict(X) > 0.5
        cm = confusion_matrix(Y, P)

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('CM for Fine Tuning')
        plt.colorbar()

        class_names = list(set(Y))

        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=0)
            plt.yticks(tick_marks, class_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        plt.savefig('visualization/cm_D.png')

    def run(self):
        if os.path.exists(model_weights):
            self.load_model()
        else:
            self.classifier()
            self.train()
            self.save_model()

        # eval_data = self.test_ds.shuffle(buffer_size = 2)
        # eval_data = eval_data.take(100)
        # self.plot_confusion_matrix(eval_data)
        # self.model.evaluate(eval_data)

if __name__ == "__main__":
    model = MnistCNN()
    model.run()
    model.TFlearn_classifier()