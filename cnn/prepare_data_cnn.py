"""
Data Preparation Methods for the CNN
"""
import pandas as pd
from parameters import SIDE_LENGTH, PIXEL_SCALE, RANDOM_SEED, NUM_CLASSES, NUM_CHANNELS
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from sklearn.model_selection import train_test_split


def pre_process_features_for_cnn(data_df: pd.DataFrame):
    """
    :param data_df: contains the features (i.e. the pixels)
    :return: rescaled data for use by Keras
    """
    # Normalize data on scale [0, 1]
    data_df = data_df / PIXEL_SCALE

    # Reshape image in 3 dimensions (height: px, width: px , canal: num colors)
    data_df = data_df.values.reshape(-1, SIDE_LENGTH, SIDE_LENGTH, NUM_CHANNELS)
    return data_df


def prepare_split_cnn(x_train_df: pd.DataFrame, y_train_df: pd.DataFrame,
                      x_test_df: pd.DataFrame, proportion_train: float = 0.1):
    """
    Given data as imported in dataloader, return formatted & split data for CNN model
    :param proportion_train: how much to put in the training set
    :param x_train_df: features in training
    :param y_train_df: label in training
    :param x_test_df: features in test
    :return:
    """
    # Pre-Process the features
    x_train = pre_process_features_for_cnn(x_train_df)
    x_test = pre_process_features_for_cnn(x_test_df)

    # Make dummy's with the labels
    y_train = to_categorical(y_train_df, num_classes=NUM_CLASSES)

    # Split the data into train / validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=proportion_train,
                                                      random_state=RANDOM_SEED)

    return x_train, x_val, y_train, y_val, x_test

