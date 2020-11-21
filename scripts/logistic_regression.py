
import numpy as np
from sklearn.linear_model import LogisticRegression
from load_feature_time_series import load_all_data_train

def train():
    training_array, training_array_label, testing_array, testing_array_label = load_all_data_train()
    print(training_array)
    print(training_array_label)
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(training_array, training_array_label)
    predictions = model.predict(testing_array)
    #print(predictions)
    return


def sigmoid(theta, X):
    score = np.dot(theta, X.T)
    return 1/(1+np.exp(-score)) - 0.0000001


def cost(X, y, theta):
    y1 = sigmoid(X, theta)
    loss = np.sum(y*np.log(y1) + (1-y)*np.log(1-y1))
    train_loss = -(1/len(X)) * loss
    return train_loss


def gradient_descent(X, y, theta, alpha, epochs):
    m = len(X)
    for i in range(0, epochs):
        for j in range(0, 10):
            h = sigmoid(theta[:,j], X)
            for k in range(0, len(theta)):
                theta[k,j] -= (alpha/m) * np.sum((h-y[:, j])*X[:, k])
    return theta


def train_scratch():
    training_array, training_array_label, testing_array, testing_array_label = load_all_data_train()
    

if __name__ == "__main__":
    train()