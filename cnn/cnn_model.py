"""
Define the CNN used for predictions
Source: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
"""

from keras.models import Sequential
from parameters import SIDE_LENGTH, NUM_CHANNELS, NUM_CLASSES
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D


def get_cnn_model(optimizer, loss="categorical_crossentropy", metrics=["accuracy"]):
    """
    :return: keras.Sequential model
    """
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=(SIDE_LENGTH, SIDE_LENGTH, NUM_CHANNELS)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
