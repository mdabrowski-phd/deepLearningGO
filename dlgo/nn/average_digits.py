#%% Listing 5.3 Computing the average value for images representing the same digit
import numpy as np
from dlgo.nn.load_mnist import load_data
from dlgo.nn.layers import sigmoid_double

def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)

train, test = load_data()
avg_eight = average_digit(train, 8)

#%% Listing 5.4 Computing and displaying the average 8 in your training set
from matplotlib import pyplot as plt

img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.colorbar()
plt.show()

#%% Listing 5.5 Computing how close a digit is to your wieghts by using the dot product
x_3 = train[2][0]    # Training sample at index 2 is a "4".
x_18 = train[17][0]  # Training sample at index 17 is an "8"

W = np.transpose(avg_eight)

print(np.dot(W, x_3))
print(np.dot(W, x_18))

#%% Listing 5.7 Computing predictions from weights and bias with dot product and sigmoid
def predict(x, W, b):  # <1>
    return sigmoid_double(np.dot(W, x) + b)

b = -45  # Based on the examples computed so far we set the bias term to -45

print(predict(x_3, W, b))
print(predict(x_18, W, b))

#%% Listing 5.8 Evaluating predictions of your model with a decision threshold
def evaluate(data, digit, threshold, W, b):
    total_samples = 1.0 * len(data)
    correct_predictions = 0
    for x in data:
        if predict(x[0], W, b) > threshold and np.argmax(x[1]) == digit:
            correct_predictions += 1
        if predict(x[0], W, b) <= threshold and np.argmax(x[1]) != digit:
            correct_predictions += 1
    return correct_predictions / total_samples

#%% Listing 5.9 Calculating prediction accuracy for three data sets
print(evaluate(data=train, digit=8, threshold=0.5, W=W, b=b))
print(evaluate(data=test, digit=8, threshold=0.5, W=W, b=b))

eight_test = [x for x in test if np.argmax(x[1]) == 8]
print(evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b))