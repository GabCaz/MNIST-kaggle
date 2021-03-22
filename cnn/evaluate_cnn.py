"""
Evaluate the performance of our model
"""

from cnn.train_cnn import load_pretrained_model, train_model
from matplotlib import pyplot as plt
from dataloader import X_train, Y_train, test
import numpy as np
from sklearn.metrics import confusion_matrix
from cnn.prepare_data_cnn import prepare_split_cnn
import itertools

# %%
_RELOAD = False
if _RELOAD:
    model, history = train_model()
else:
    model, history = load_pretrained_model()

# %% Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss")
ax[0].legend()
ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
ax[1].legend()
plt.show()

# %%
X_train, X_val, Y_train, Y_val, test = prepare_split_cnn(x_train_df=X_train, y_train_df=Y_train, x_test_df=test)
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)


# %% Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    :param cm: 2d numpy array, as returned by sklearn.metrics.confusion_matrix
    :param classes: iterable with the class labels in them
    :param normalize: whether to apply normalization to the plot
    :param title: plot title
    :param cmap: color map to use
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(10))
plt.show()

# %% Looking at biggest mistakes (probabilities of real value - predicted)

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(a=Y_pred_errors, indices=Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors
most_important_errors = sorted_dela_errors[-6:]


#%%
def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """
    This function shows 6 images with their predicted and real labels
    :param errors_index: iterable with the indices of most important errors in img_errors
    :param img_errors: (nerrors, tuple_image) contains the observations which resulted in a error,
        with a dimension formatted for keras: tuple_image = (side, side, channels)
    :param pred_errors: iterable of same len errors_index, what was predicted
    :param obs_errors: iterable of same len errors_index, what is the true label
    :return:
    """
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
            n += 1

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
plt.show()
