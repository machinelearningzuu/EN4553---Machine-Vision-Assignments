import os
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

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input, Dropout, BatchNormalization
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from util import*
from variables import*

np.random.seed(seed)
tf.random.set_seed(seed)

class MnistCNN:
    def __init__(self):
        if not os.path.exists(model_weights):
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
        x = Dense(64, activation='relu')(x)
        outputs = Dense(10, activation='softmax')(x)
        self.model = Model(inputs, outputs)
        self.model.summary()

        tf.keras.utils.plot_model(
                        self.model, 
                        to_file=Q4_a_file, 
                        show_shapes=True
                        )

    def TFlearn_classifier(self):
        Xtrain , Ytrain, Xtest , Ytest = TFlearn_dataset(map_fn_Q4, MNIST_data)

        inputs = self.model.input
        x = self.model.layers[-2].output
        # x = Dense(64, activation='relu', name='finetune_dense')(x)
        # x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid', name='sigmoid_output')(x)
        self.model2 = Model(inputs, outputs)
        self.model2.summary()

        tf.keras.utils.plot_model(
                        self.model2, 
                        to_file=Q4_d_file, 
                        show_shapes=True
                        )

        self.model2.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy'],
        )

        self.model2.fit(
                    Xtrain,
                    Ytrain,
                    epochs=2,
                    batch_size=128,
                    validation_data=[Xtest, Ytest]
                    )

    def train(self):
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=Adam(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
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
        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights(model_weights)
        loaded_model.compile(
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    optimizer=Adam(),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                        )
        self.model = loaded_model

    def run(self):
        if os.path.exists(model_weights):
            self.load_model()
        else:
            self.classifier()
            self.train()
            self.save_model()

if __name__ == "__main__":
    model = MnistCNN()
    model.run()
    # model.TFlearn_classifier()
