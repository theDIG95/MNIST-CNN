# MNIST-CNN

A Convolutional neural network classifier made using Theano frame work in python. The problem solved using this is the classification of the MNIST handwritten digit dataset available [here](https://www.python-course.eu/neural_network_mnist.php).

## Usage

### Same dataset

Download the data and adjust `MNIST_DATA_PATH`, `MNIST_TRAIN_BATCH` and `MNIST_TEST_BATCH` in `data_importer.py` accordingly.

### Other datasets

The `CNNClassifier` can be used for any classification problem. The parameters can be passed in the constructor and `fit`/`k_fold_x_validation` method for training.  
Predictions can be made via the `predict` method.  
For further details please view the `cnn.py` file, it contains docstrings and comments explaining each step of the process.

## Model properties

A sequential convolutional neural network with the following implementations.

- LeNet model.
- Batched gradient descent.
- Nestrov's momentum to improve learning with batches.
- RMSProp for adaptive learning rate.
- Dropout regularization.
- K-Fold cross validation.

## Model Parameters

For the 'MNIST handwritten digit dataset'

- First ConvPool layer with filter size (4, 4), feature maps (50), down sampling (2, 2)
- Second ConvPool layer with filter size (4, 4), feature maps (25), down sampling (2, 2)
- One hidden layer with depth (100)
- Learning rate (0.0005)
- Hidden and final layer's probability of keeping (0.8, 0.8)
- RMSProp cache decay (0.9)
- Nestrov's momentum (0.8)
- Epochs (500)
- Test/validation split (20) for 1 part validation (19,000 vs 1,000)
- Batch size (1,000)

## Results

Output of training

**NOTE** The training was done over 19,000 samples and testing on 1,000 samples due to limitations imposed by hardware resources.

```(bash)
100  iterations | 4.058 % train error | 3.6 % testing error ...
200  iterations | 1.926 % train error | 2.1 % testing error ...
300  iterations | 1.326 % train error | 1.9 % testing error ...
400  iterations | 1.126 % train error | 1.5 % testing error ...
Training final error:  0.8263157894736842 %
Testing final error:  0.8999999999999999 %
```