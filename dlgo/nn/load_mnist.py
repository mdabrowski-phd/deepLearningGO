#%% Listing 5.1 One-hot encoding of MNIST labels
import numpy as np

def encode_label(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#%% Listing 5.2 Reshaping MNIST data and loading training and test data
def shape_data(data):
    features = [np.reshape(x, (784, 1)) for x in data[0]]
    labels = [encode_label(y) for y in data[1]]
    return list(zip(features, labels))

def load_data_impl():
    path = 'E:/MACHINE_LEARNING/DeepLearningGoGame/dlgo_MD/dlgo/nn/mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train']/255, f['y_train']
    x_test, y_test = f['x_test']/255, f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def load_data():
    train_data, test_data = load_data_impl()
    return shape_data(train_data), shape_data(test_data)