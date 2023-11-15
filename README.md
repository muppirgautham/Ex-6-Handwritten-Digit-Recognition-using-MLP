# Skill Assisessment-Handwritten Digit Recognition using MLP
## Aim:
       To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:
I implemented a simple two-layer neural network and trained it on the MNIST digit recognizer dataset. It's meant to be an instructional example, through which you can understand the underlying math of neural networks better.

## Algorithm :
1.Import the libraries and load the dataset.

2.Preprocess the data. *The image data cannot be fed directly into the model so we need to perform some operations and process the data to make it ready for our neural network.

3.Create the model.Now we will create our CNN model in Python data science project. A CNN model generally consists of convolutional and pooling layers.

4.Train the model.The model.fit() function we have implemented MLP with backpropagation using ReLU activation function. will start the training of the model. It takes the training data, validation data, epochs, and batch size.

5.Evaluate the model. We have 10,000 images in our dataset which will be used to evaluate how good our model works.

6.Create GUI to predict digits.Now for the GUI, we have created a new file in which we build an interactive window to draw digits on canvas and with a button, we can recognize the digit.

## Program:
```
Developed by: M GAUTHAM
RegisterNumber:  212221230027
```
```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
```
```
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
```
```
Y_train
```
```

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
```
```
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
```
```
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
```
```
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
```
```
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
```
```
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
```
```
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
```


## Output :

![image](https://github.com/muppirgautham/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94810884/fc92e90d-864f-48b1-8c46-23444a1a8f25)

![image](https://github.com/muppirgautham/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94810884/5f1e4dcc-92bb-4201-9fed-44ca32263bcf)

![image](https://github.com/muppirgautham/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94810884/9afef96b-8c67-4e69-9698-6654af250889)

![image](https://github.com/muppirgautham/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94810884/d9015886-1fd3-4890-bbbf-bd172862c16e)

![image](https://github.com/muppirgautham/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94810884/8dbc787e-0fb2-44fa-82f3-915b821d43e9)

![image](https://github.com/muppirgautham/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94810884/aa57e12d-f5d2-401a-996e-30f3f247e747)

![image](https://github.com/muppirgautham/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94810884/a9cc3dcd-ec63-4ae6-9285-5fa9a05768e3)

![image](https://github.com/muppirgautham/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94810884/cd4652e2-e49f-4143-befe-2d0714ee3113)

![image](https://github.com/muppirgautham/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/94810884/f03d9b03-af80-4cc5-a88c-3733226315ab)



## Result:
Thus The Implementation of Handwritten Digit Recognition using MLP Is Executed Successfully.


