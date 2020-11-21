import numpy as np
import os
import sys
from Log import Log

topdir = os.path.split(os.path.split(os.path.realpath(sys.argv[0]))[0])[0]
logfile = os.path.join(topdir, 'log/logistic_training_mock.log')
logging = Log(logfile)

#Training
training_features = os.path.join(topdir, 'log/X.csv')
training_labels = os.path.join(topdir, 'log/y.csv')

#Testing
testing_features = os.path.join(topdir, 'log/mfcc_testing_value_less.csv')
testing_labels = os.path.join(topdir, 'log/mfcc_testing_target_less.csv')

# Training
df_numpy = np.genfromtxt(training_features, delimiter=",")
y_numpy = np.genfromtxt(training_labels, delimiter=",")

# Testing 
df_numpy_testing = np.genfromtxt(testing_features, delimiter=",")
y_numpy_testing = np.genfromtxt(testing_labels, delimiter=",")

# df_numpy = (df_numpy * 10000000)
# df_numpy_testing = (df_numpy_testing * 10000000)

def hypothesis(theta, X):
    logging.writelog(f"Computing the hypothesis with theta and X")
    logging.write_numpy_array(theta)

    dot_product = np.dot(theta, X.T)
    logging.write_newline()
    logging.writelog(f"The dot product is:")
    logging.write_numpy_array(dot_product)

    n_exp = np.exp(-dot_product)
    logging.write_newline()
    logging.writelog(f"The log of dot product is:")
    logging.write_numpy_array(n_exp)

    h = 1 / (1 + n_exp) - 0.0000001
    logging.write_newline()
    logging.writelog(f"The hypothesis is :")
    logging.write_numpy_array(h)
    return h

def cost(X, y, theta):
    y1 = hypothesis(X, theta)
    return -(1/len(X)) * np.sum(y*np.log(y1) + (1-y)*np.log(1-y1))

add_array = np.ones((df_numpy.shape[0], 1))
X = np.hstack((add_array, df_numpy))
add_array_testing = np.ones((df_numpy_testing.shape[0], 1))
X_testing = np.hstack((add_array_testing, df_numpy_testing))

y1 = np.zeros([df_numpy.shape[0], len(np.unique(y_numpy))])
y1_testing = np.zeros([df_numpy_testing.shape[0], len(np.unique(y_numpy_testing))])

for i in range(0, len(np.unique(y_numpy))):
    for j in range(0, len(y1)):
        if y_numpy[j] == np.unique(y_numpy)[i]:
            y1[j, i] = 1
        else: 
            y1[j, i] = 0

def gradient_descent(X, y, theta, alpha, epochs):
    logging.write_newline()

    logging.writelog(f"X size: {X.shape}")
    logging.write_numpy_array(X)
    logging.write_newline()
    logging.writelog(f"y size: {y.shape}")
    logging.write_numpy_array(y)
    logging.write_newline()
    logging.writelog(f"theta size: {theta.shape}")
    m = len(X)
    for i in range(0, epochs):
        logging.write_newline()
        logging.writelog(f"Training the {i}th epoch")
        logging.write_numpy_array(theta)
        for j in range(0, len(np.unique(y_numpy))):
            logging.write_newline()
            logging.writelog(f"hypothesis from theta[:,{j}] and X")
            h = hypothesis(theta[:,j], X)
            for k in range(0, theta.shape[0]):
                theta[k, j] -= (alpha/m) * np.sum((h-y[:, j])*X[:, k])
    return theta, cost


theta = np.ones([df_numpy.shape[1]+1, y1.shape[1]])
theta = gradient_descent(X, y1, theta, 0.001, 3000)

logging.write_newline()
logging.writelog("INFERENCE")

output = []
for i in range(0, len(np.unique(y_numpy))):
    theta1 = theta[0]
    logging.write_newline()
    logging.writelog(f"theta1 size: {theta1.shape}")
    logging.write_numpy_array(theta1[:,i])
    h = hypothesis(theta1[:,i], X)
    logging.write_newline()
    logging.writelog(f"hypothesis value: {h}")
    output.append(h)
output_cat = np.vstack(output)

# # TESTING
# output_test = []
# for i in range(0, len(np.unique(y_numpy_testing))):
#     theta1 = theta[0]
#     h = hypothesis(theta1[:,i], X_testing)
#     logging.write_newline()
#     logging.writelog(f"hypothesis value: {h}")
#     output_test.append(h)
# output_cat_test = np.vstack(output_test)


accuracy = 0
for col in range(0, len(np.unique(y_numpy))):
    for row in range(len(y1)):
        if y1[row, col] == 1 and output_cat[col, row] >= 0.5:
            accuracy += 1
accuracy = accuracy/len(X)
print(accuracy)


# accuracy = 0
# for col in range(0, len(np.unique(y_numpy_testing))):
#     for row in range(len(y1_testing)):
#         if y1_testing[row, col] == 1 and output_cat_test[col, row] >= 0.5:
#             accuracy += 1
# accuracy = accuracy/len(X_testing)
# print(accuracy)