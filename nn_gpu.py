import numpy as np
import pandas as pd


import pycuda.autoinit
import pycuda.gpuarray as gpu
import pycuda.cumath as cm
from pycuda.elementwise import ElementwiseKernel

# Activation functions
gpu_sigmoid = ElementwiseKernel(
    "float x",
    "x[i] = x[i]*(1.0-x[i])",
    "gpu_sigmoid"
)
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1.0 - x)
    else:
        return 1.0 / (1.0 + np.exp(-x))

def relu(x, deriv=False):
    if deriv:
        return 1.0 - cm.exp(-x)
    else:
        return gpu.maximum(x, 0)
        # vf = np.vectorize(lambda x : x if x > 30 else np.log(1.0 + np.exp(x)))
        # return vf(x)
        # return cm.log(1.0 + cm.exp(x))

def softmax(x, deriv=False):
    if deriv:
        return x * (1.0 - x)
    else:
        x = x - gpu.max(x)
        return cm.exp(x) / gpu.sum(cm.exp(x))


x = pd.read_csv("data/train_x.csv", delimiter=",", nrows=100,  memory_map=True).values
y = pd.read_csv("data/train_y.csv", delimiter=",", nrows=100, memory_map=True).values


x = x / 255.0
# x = gpu.to_gpu(x)
# y = y.reshape(-1, 1)

# test_x = pd.read_csv("data/test_x.csv", delimiter=",", memory_map=True).values
# test_x = test_x / 255.0

train_data_size = len(x)

output_encoding = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10,
                 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18,
                 20:19, 21:20, 24:21, 25:22, 27:23, 28:24, 30:25, 32:26,
                 35:27, 36:28, 40:29, 42:30, 45:31, 48:32, 49:33, 54:34,
                 56:35, 63:36, 64:37, 72:38, 81:39}

new_y = np.zeros([len(y), 40])

for i, value in np.ndenumerate(y):
    new_y[i, output_encoding[value]] = 1

# y = gpu.to_gpu(new_y)

print(' Loaded Data')

# Parameters
alpha = 0.01
nb_minibatch = 1000
nb_updates = np.round(train_data_size/nb_minibatch) * 500

# layer sizes
input_layer_size = 4096
layer2_size = 500
layer3_size = 500
layer4_size = 500
output_layer_size = 40

#dropout = 0.5
#momentum = 0.75

np.random.seed(7)

# Init weights
bias_in       = gpu.to_gpu(2 * np.random.random([input_layer_size]) - 1)
weights_in_2  = gpu.to_gpu(2 * np.random.random([input_layer_size, layer2_size]) - 1)
bias_2        = gpu.to_gpu(2 * np.random.random([layer2_size]) - 1)
weights_2_3   = gpu.to_gpu(2 * np.random.random([layer2_size, layer3_size]) - 1)
bias_3        = gpu.to_gpu(2 * np.random.random([layer3_size]) - 1)
weights_3_4   = gpu.to_gpu(2 * np.random.random([layer3_size, layer4_size]) - 1)
bias_4        = gpu.to_gpu(2 * np.random.random([layer4_size]) - 1)
weights_4_out = gpu.to_gpu(2 * np.random.random([layer4_size, output_layer_size]) - 1)
bias_out      = gpu.to_gpu(2 * np.random.random([output_layer_size]) - 1)

gpu_sigmoid(weights_in_2)


#Init weight updates
db_in    = gpu.to_gpu(np.zeros(bias_in.shape))
dW_in_2  = gpu.to_gpu(np.zeros(weights_in_2.shape))
db_2     = gpu.to_gpu(np.zeros(bias_2.shape))
dW_2_3   = gpu.to_gpu(np.zeros(weights_2_3.shape))
db_3     = gpu.to_gpu(np.zeros(bias_3.shape))
dW_3_4   = gpu.to_gpu(np.zeros(weights_3_4.shape))
db_4     = gpu.to_gpu(np.zeros(bias_4.shape))
dW_4_out = gpu.to_gpu(np.zeros(weights_4_out.shape))
db_out   = gpu.to_gpu(np.zeros(bias_out.shape))

## Train
corrects = 0
tries = 0


for i in range(nb_updates):

    for j in range(nb_minibatch):
        #pick an image randomly
        i_row = np.random.randint(0, train_data_size)
        x_row = gpu.to_gpu(x[i_row])
        y_row = gpu.to_gpu(y[i_row])

        #feed-forward (activation -> a)
        a1 = sigmoid(x_row+bias_in)
        a2 = relu(gpu.dot(weights_in_2.transpose(), a1) + bias_2)
        a3 = relu(gpu.dot(weights_2_3.transpose(), a2) + bias_3)
        a4 = relu(gpu.dot(weights_3_4.transpose(), a3) + bias_4)
        a5 = softmax(gpu.dot(weights_4_out.transpose(), a4) + bias_out)


        #backpropagation
        b5 = y_row - a5
        b4 = gpu.dot(weights_4_out, b5) * relu(a4, True)
        b3 = gpu.dot(weights_3_4, b4) * relu(a3, True)
        b2 = gpu.dot(weights_2_3, b3) * relu(a2, True)
        b1 = gpu.dot(weights_in_2, b2) * sigmoid(a1, True)

        #weight adjustments
        db_in += b1
        dW_in_2 += gpu.dot(a1.reshape(-1, 1), b2.reshape(1, -1))
        db_2 += b2
        dW_2_3 += gpu.dot(a2, b3)
        db_3 += b3
        dW_3_4 += gpu.dot(a3, b4)
        db_4 += b4
        dW_4_out += gpu.dot(a4.reshape(-1, 1), b5.reshape(1, -1))
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
        print(weights_in_2, weights_2_3, bias_2)

        print('PROGRESS:')
        print ('    Batch = ', i)
        print ('    Alpha = ', alpha)
        print ('    Accuracy = ', (100.0 * corrects/tries))

        tries = 0
        corrects = 0

