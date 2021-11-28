import numpy as np
import pickle as pck
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import matplotlib.pyplot as plt

def load_file(file):
    """
    file = data file to load from
    """

    with open(file, 'rb') as f:
        data = pck.load(f, encoding='bytes')

    return data

def load_training(folder):
    """

    loads all batch image training data into numpy array

    folder = directory containing all the batch files
    """

    batches = [ f"data_batch_{i}" for i in range(1, 6)]

    path = os.path.dirname(__file__)

    folder = os.path.join(path, folder)

    X = np.empty(shape=(10000, 3072))
    y = np.empty(shape=10000)

    for i, filename in enumerate(batches):
        
        data = load_file(folder + filename)
        print(f"LOADED {data[b'batch_label']}")

        if i == 0:
            X = data[b'data']
            y = data[b'labels']
            continue
        X = np.concatenate((X, data[b'data']), axis=0)
        y = np.concatenate((y, data[b'labels']), axis=0)

    return X, y

def load_test(file):

    path = os.path.dirname(__file__)

    file = os.path.join(path, file)

    X_tst = np.empty(shape=(10000, 3072))
    y_tst = np.empty(shape=10000)

    data = load_file(file)
    X_tst = data[b'data']
    y_tst = np.array(data[b'labels'])

    return X_tst, y_tst

def load_cifar10():
    """
    Returns: Tuple
        X_train,  X_test = training and test image data
        y_train, y_test = training and test labels
        classes = classifcation names
    """

    X_trn, y_trn = load_training('./data/cifar-10-batches-py/')
    X_tst, y_tst = load_test('./data/cifar-10-batches-py/test_batch')

    dir = os.path.dirname(__file__)

    path = os.path.join(dir, "./data/cifar-10-batches-py/batches.meta")

    metadata = load_file(path)

    label_names = np.ravel(metadata[b'label_names'])

    return X_trn, X_tst, y_trn, y_tst, label_names

def scale_data(X):
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X)
    return scaler.transform(X)

def save_model(model):
    filename = './out/cifar10.pth' 
    torch.save(model.state_dict(), filename)
    
def load_model(model, file):
    load_dict = torch.load(file)

    model.load_state_dict(load_dict)

    return model

def plot_confusion_matrix(y_pred, y_true, classes = [], display_accs=True):

    y_pred_names = [classes[i] for i in y_pred]
    y_true_names = [classes[i] for i in y_true]

    cm = confusion_matrix(y_true_names, y_pred_names, labels=classes)

    if display_accs:
        accuracies = cm.diagonal() / cm.sum(axis=1)
        
        for i, label in enumerate(classes):
            print("accuracy of {}: {:.2%}".format(label.decode('ascii'), accuracies[i]))

    d = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    d.plot(xticks_rotation='vertical')
    plt.show()

    # ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=classes, xticks_rotation='vertical')


