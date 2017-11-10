import numpy as np
import pandas as pd
import csv

from sklearn.preprocessing import normalize
from sklearn import linear_model

class Network(object):
    '''
        Neural Network class
    '''

    def __init__(self, *args, **kwargs):
        self.weights = np.zeros([10, 1])
        self.biases = np.ones([10, 1])

        pass

    def sigmoid(self, a, deriv=False):
        if deriv:
            return a * (1 - a)
        else:
            return 1/(1 + np.exp(-a))


    def feed_forward(self, a):
        '''
            Output of NN for an input vector a
        :param a: input vector
        :return: output
        '''

        for b, w in zip(self.biases, self.weights):
            pass


    def loss(self, predictions, labels):

        return np.average(np.sum(predictions - labels, axis=1))
        ''' Cross-Entropy Loss'''
        #return - np.sum(np.sum(labels * np.log(predictions) , axos=1)) / len(predictions)



def loss(predictions, labels):
    ''' Negative log-likelihood to go with the softmax activation'''
    return  np.mean(np.log(predictions))

#Activation functions
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1.0 - x)
    else:
        return 1.0 / (1.0 + np.exp(-x))

def relu(x, deriv=False):
    if deriv:
        return 1.0 - np.exp(-x)
    else:
        return np.log(1.0 + np.exp(x))

def softmax(x, deriv=False):
    if deriv:
        return x * (1.0 - x)
    else:
        return np.exp(x) / sum(np.exp(x))


x = pd.read_csv("data/train_x.csv", delimiter=",").values[:1000]
y = pd.read_csv("data/train_y.csv", delimiter=",").values[:1000]
test_x = pd.read_csv("data/test_x.csv", delimiter=",").values

x = x / 255.0
# y = y.reshape(-1, 1)
test_x = test_x / 255.0

train_data_size = len(x)

output_encoding = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10,
                 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18,
                 20:19, 21:20, 24:21, 25:22, 27:23, 28:24, 30:25, 32:26,
                 35:27, 36:28, 40:29, 42:30, 45:31, 48:32, 49:33, 54:34,
                 56:35, 63:36, 64:37, 72:38, 81:39}

new_y = np.zeros([len(y), 40])
for i, value in enumerate(y):
    new_y[i][output_encoding[value]] = 1

y = new_y

print(' Loaded Data')

# Parameters
alpha = 0.001
nb_minibatch = 1000
nb_updates = np.round(train_data_size/nb_minibatch) * 500

# layer sizes
input_layer_size = 64^2
layer2_size = 500
layer3_size = 500
layer4_size = 500
output_layer_size = 40

#dropout = 0.5
#momentum = 0.75

np.random.seed(7)

# Init weights
bias_in = 2 * np.random.random([input_layer_size]) - 1
weights_in_2 = 2 * np.random.random([input_layer_size, layer2_size]) - 1
bias_2 = 2 * np.random.random([layer2_size]) - 1
weights_2_3 = 2 * np.random.random([layer2_size, layer3_size]) - 1
bias_3 = 2 * np.random.random([layer3_size]) - 1
weights_3_4 = 2 * np.random.random([layer3_size, layer4_size]) - 1
bias_4 = 2 * np.random.random([layer4_size]) - 1
weights_4_out = 2 * np.random.random([layer4_size, output_layer_size]) - 1
bias_out = 2 * np.random.random([output_layer_size]) - 1


#Init weight updates
db_in = np.zeros(bias_in.shape)
dW_in_2 = np.zeros(weights_in_2.shape)
db_2 = np.zeros(bias_2.shape)
dW_2_3 = np.zeros(weights_2_3.shape)
db_3 = np.zeros(bias_3.shape)
dW_3_4 = np.zeros(weights_3_4.shape)
db_4 = np.zeros(bias_4.shape)
dW_4_out = np.zeros(weights_4_out.shape)
db_out = np.zeros(bias_out.shape)

## Train
corrects = 0
tries = 0

for i in range(nb_updates):

    for j in range(nb_minibatch):
        #pick an image randomly
        i_row = np.random.randint(0, train_data_size)
        x_row = x[i_row]
        y_row = y[i_row]

        #feed-forward (activation -> a)
        a1 = sigmoid(x_row+bias_in)
        a2 = relu(np.dot(weights_in_2, a1) + bias_2)
        a3 = relu(np.dot(weights_2_3, a2) + bias_3)
        a4 = relu(np.dot(weights_3_4, a3) + bias_4)
        a5 = softmax(np.dot(weights_4_out, a4) + bias_out)


        #backpropagation
        b5 = y_row - a5
        b4 = np.dot(weights_4_out, b5) * relu(a4, True)
        b3 = np.dot(weights_3_4, b4) * relu(a3, True)
        b2 = np.dot(weights_2_3, b3) * relu(a2, True)
        b1 = np.dot(weights_in_2, b2) * sigmoid(a1, True)

        #weight adjustments
        db_in += b1
        dW_in_2 += np.dot(a1, b2)
        db_2 += b2
        dW_2_3 += np.dot(a2, b3)
        db_3 += b3
        dW_3_4 += np.dot(a3, b4)
        db_4 += b4
        dW_4_out += np.dot(a4, b5)
        db_out += b5

        #check for correct guess
        if np.argmax(a5) == np.argmax(y_row):
            corrects += 1

        tries += 1

    # update parameters per SGD
    bias_in = alpha * db_in
    weights_in_2 = alpha * dW_in_2
    bias_2 = alpha * db_2
    weights_2_3 = alpha * dW_2_3
    bias_3 = alpha * db_3
    weights_3_4 = alpha * dW_3_4
    bias_4 = alpha * db_4
    weights_4_out = alpha * dW_4_out
    bias_out = alpha * db_out

    #decrease alpha
    alpha = alpha * ((nb_updates - i)/(nb_updates - i + 1))

    #print progress
    if i % 100 == 0:

        print('PROGRESS:')
        print ('    Batch = ', i)
        print ('    Alpha = ', np.round(alpha, 8))
        print ('    Accuracy = ', np.round((100.0 * corrects/tries), 8))

        tries = 0
        corrects = 0

