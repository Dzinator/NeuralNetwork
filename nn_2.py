import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

'''
    Simpler version of the NN with 2 hidden layers
    
    Reads data from data/ folder where script is
'''

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
        return np.maximum(x, 0)

        # return np.log(1.0 + np.exp(x))

def softmax(x, deriv=False):
    if deriv:
        return x * (1.0 - x)
    else:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x), axis=0)


x = pd.read_csv("data/train_x.csv", delimiter=",", memory_map=True).values
y = pd.read_csv("data/train_y.csv", delimiter=",", memory_map=True).values

#binarize input
x = (x > 240.0)

# test_x = pd.read_csv("data/test_x.csv", delimiter=",", memory_map=True).values
# test_x = (x > 240)

output_encoding = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10,
                 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18,
                 20:19, 21:20, 24:21, 25:22, 27:23, 28:24, 30:25, 32:26,
                 35:27, 36:28, 40:29, 42:30, 45:31, 48:32, 49:33, 54:34,
                 56:35, 63:36, 64:37, 72:38, 81:39}

new_y = np.zeros([len(y), 40])

for i, value in np.ndenumerate(y):
    new_y[i, output_encoding[value]] = 1

y = new_y

#cross validation 80 %split
x, x_test, y, y_test = train_test_split(x, y, train_size=0.8)

print(' Loaded Data')

# Parameters
train_data_size = len(x)
alpha = 0.01
nb_minibatch = 1000
nb_updates = int(train_data_size/nb_minibatch) * 500

# layer sizes
input_layer_size = len(x[0])
layer2_size = 1000
layer3_size = 1000
output_layer_size = 40

np.random.seed(7)

# Init weights
try:
    with open('weights_2l.bin', 'r') as fp:
        tweights_in_2,
        bias_2,
        weights_2_3,
        bias_3,
        weights_3_out,
        bias_out = pickle.loads(fp.read())
    print '-Loaded saved weights'
except:
    bias_in = 2 * np.random.random([input_layer_size]) - 1
    weights_in_2 = 2 * np.random.random([input_layer_size, layer2_size]) - 1
    bias_2 = 2 * np.random.random([layer2_size]) - 1
    weights_2_3 = 2 * np.random.random([layer2_size, layer3_size]) - 1
    bias_3 = 2 * np.random.random([layer3_size]) - 1
    weights_3_out = 2 * np.random.random([layer3_size, output_layer_size]) - 1
    bias_out = 2 * np.random.random([output_layer_size]) - 1


#Init weight updates
db_in = np.zeros(bias_in.shape)
dW_in_2 = np.zeros(weights_in_2.shape)
db_2 = np.zeros(bias_2.shape)
dW_2_3 = np.zeros(weights_2_3.shape)
db_3 = np.zeros(bias_3.shape)
dW_3_out = np.zeros(weights_3_out.shape)
db_out = np.zeros(bias_out.shape)

# Train

corrects = 0
tries = 0

for i in range(nb_updates):

    for j in range(nb_minibatch):

        #pick an image randomly
        i_row = np.random.randint(0, train_data_size)
        x_row = x[i_row]
        y_row = y[i_row]

        #feed-forward (activations : a)
        a2 = relu(np.dot(x_row, weights_in_2) + bias_2)
        a3 = relu(np.dot(a2, weights_2_3) + bias_3)
        o = softmax(np.dot(a3, weights_3_out) + bias_out)


        #backpropagation
        bo = y_row - o
        b3 = np.dot(weights_3_out, bo) * relu(a3, True)
        b2 = np.dot(weights_2_3, b3) * relu(a2, True)

        #weight adjustments
        dW_in_2 += np.dot(x_row.reshape(-1, 1), b2.reshape(1, -1))
        db_2 += b2
        dW_2_3 += np.dot(a2.reshape(-1, 1), b3.reshape(1, -1))
        db_3 += b3
        dW_3_out += np.dot(a3.reshape(-1, 1), bo.reshape(1, -1))
        db_out += bo

        #check for correct guess
        if np.argmax(o) == np.argmax(y_row):
            corrects += 1

        tries += 1

    # update parameters per SGD
    weights_in_2 = alpha * dW_in_2
    bias_2 = alpha * db_2
    weights_2_3 = alpha * dW_2_3
    bias_3 = alpha * db_3
    weights_3_out = alpha * dW_3_out
    bias_out = alpha * db_out

    #decrease alpha over training
    alpha = alpha * ((nb_updates - i)/(nb_updates - i + 1))

    #print progress
    if i % 100 == 0:

        print('PROGRESS:')
        print('    Batch = %d' % i)
        print('    Alpha = %0.6f' % alpha)
        print('    Accuracy = %0.4f' % (100.0 * corrects/tries))

        test_correct = 0
        test_tries = 0

        for x_t, y_t in zip(x_test, y_test):

            a2 = relu(np.dot(x_t, weights_in_2) + bias_2)
            a3 = relu(np.dot(a2, weights_2_3) + bias_3)
            o = softmax(np.dot(a3, weights_3_out) + bias_out)

            if np.argmax(o) == np.argmax(y_t):

                test_correct += 1

            test_tries += 1

        print ('    Test Accuracy = %0.4f' % (100.0 * test_correct / test_tries))

        #saving weights
        with open('weights_2l.bin', 'w') as fp:
            fp.write(pickle.dumps([
                weights_in_2,
                bias_2,
                weights_2_3,
                bias_3,
                weights_3_out,
                bias_out]))

        tries = 0
        corrects = 0

