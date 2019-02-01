#! /usr/bin/python3

import os
import numpy as np

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MNIST_DATA_PATH = BASE_PATH + '/Data/mnist/'
MNIST_TRAIN_BATCH = MNIST_DATA_PATH + 'mnist_train.csv'
MNIST_TEST_BATCH = MNIST_DATA_PATH + 'mnist_test.csv'

def mnist_data():
    """Retrieve and format the MNIST dataset
    Available at https://www.python-course.eu/neural_network_mnist.php
    
    Returns:
        tuple(numpy.ndarray, numpy.ndarray)-- The images formatted as 28x28x1 and the labels
    """

    print('Getting data ...')
    data = _get_mnist_csv()
    labels = data[:, 0]
    imgs = (data[:, 1:].reshape([-1, 1, 28, 28])) / 255

    return imgs[:20000].astype(np.float32), labels[:20000].astype(np.float32)

def _get_mnist_csv():
    """[INTERNAL] Reads the MNIST train and testing csv files
    
    Returns:
        numpy.ndarray -- The labels and flattened images as a single matrix
    """

    data_train = np.loadtxt(MNIST_TRAIN_BATCH, delimiter=',')
    data_test = np.loadtxt(MNIST_TEST_BATCH, delimiter=',')
    data = np.vstack((data_train, data_test))

    return data
