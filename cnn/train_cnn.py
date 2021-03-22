"""
Train the CNN
"""
from keras.preprocessing.image import ImageDataGenerator
from dataloader import X_train, Y_train, test
from cnn.prepare_data_cnn import prepare_split_cnn
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import pickle
from cnn.cnn_model import get_cnn_model

PICKLED_MODEL_PATH = 'cnn/trained_cnn.pickle'
PICKLED_HISTORY_PATH = "cnn/history.pickle"
EPOCHS = 1
BATCH_SIZE = 86


def train_model(X_train=X_train, Y_train=Y_train, test=test, path_save_model=PICKLED_MODEL_PATH,
                path_save_history=PICKLED_HISTORY_PATH):
    """
    Build and train the CNN model from scratch
    :return: Tre trained model, and The History
    """
    # Data split; Making training set with data augmentation
    X_train, X_val, Y_train, Y_val, test = prepare_split_cnn(x_train_df=X_train, y_train_df=Y_train, x_test_df=test)
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model = get_cnn_model(optimizer=optimizer)

    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  epochs=EPOCHS, validation_data=(X_val, Y_val),
                                  verbose=2, steps_per_epoch=X_train.shape[0] // BATCH_SIZE
                                  , callbacks=[learning_rate_reduction])

    pickle.dump(model, open(path_save_model, 'wb'))
    pickle.dump(history, open(path_save_history, 'wb'))
    return model, history


def load_pretrained_model(path_load_model: str = PICKLED_MODEL_PATH,
                          path_load_history: str = PICKLED_HISTORY_PATH):
    """
    Load a previously trained model, which we had pickled
    :param path_load_model:
    :return: The trained model, The history of the training
    """
    loaded_model = pickle.load(open(path_load_model, 'rb'))
    loaded_history = pickle.load(open(path_load_history, 'rb'))
    return loaded_model, loaded_history
