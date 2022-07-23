import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def init_parameter():
    W1 = np.random.rand(10, 784) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    W3 = np.random.rand(10, 10) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3


def ReLu(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, W2, W3, b1, b2, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(Z1) + b2
    A2 = ReLu(Z2)
    Z3 = W3.dot(Z2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def ReLU_deriv(Z):
    return Z > 0


def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ3) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ2.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3


def get_predictions(A3):
    return np.argmax(A3, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_parameter()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, W2, W3, b1, b2, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3


def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, W2, W3, b1, b2, b3, X)
    predictions = get_predictions(A3)
    return predictions


def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    df_train = pd.read_csv("Data/train.csv")
    df_test = pd.read_csv("Data/test.csv")
    y_df = pd.read_csv("Data/sample_submission.csv")
    # get test data set
    X_test = df_test.values.T
    X_test = X_test / 255
    y_test = y_df["Label"].values.T
    # get training set
    data = np.array(df_train)
    m, n = data.shape
    np.random.shuffle(data)
    # get validation set
    data_val = data[0:1000]
    X_val = data_val[:, 1:n].T
    X_val = X_val / 255
    y_val = data_val[:, 0].T
    # get test set
    data_train = data[1000:m]
    X_train = data_train[:, 1:n].T
    X_train = X_train / 255
    y_train = data_train[:, 0].T

    W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, y_train, 0.05, 1000)

    test_prediction(0, W1, b1, W2, b2, W3, b3)
    test_prediction(10, W1, b1, W2, b2, W3, b3)
    test_prediction(20, W1, b1, W2, b2, W3, b3)
    test_prediction(30, W1, b1, W2, b2, W3, b3)

    print("finish")
