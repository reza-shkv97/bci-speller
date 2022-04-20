import numpy as np
from scipy.linalg import pinv
import mat73
from utils.metrics import performance_measure


X_train_PCA = np.loadtxt('../Data/X_train_A_pca.npy.gz')
Y_train = np.loadtxt('../Data/y_train_A.npy.gz')
X_test_PCA = np.loadtxt('../Data/X_test_A_pca.npy.gz')

test_dataset = mat73.loadmat('../Data/Test_A.mat')
event = np.array(test_dataset['event'])
test_str_A = test_dataset['test_str_A']
input_size = X_train_PCA.shape[1]

hidden_size = 900

W_input = np.random.normal(size=[input_size, hidden_size])
bias = np.random.normal(size=[hidden_size])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


G_tr = (np.dot(X_train_PCA, W_input)) + bias
H_tr = sigmoid(G_tr)

W_output = np.dot(pinv(H_tr), Y_train)

G_ts = (np.dot(X_test_PCA, W_input)) + bias
H_ts = sigmoid(G_ts)

prediction = np.dot(H_ts, W_output)

test_accuracy = performance_measure(prediction, test_str_A, event)

print('acc_test: {}'.format(test_accuracy))