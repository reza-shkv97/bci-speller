import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mat73
from utils.metrics import accuracy_score

X_train_PCA = np.loadtxt('../Data/X_train_A_pca.npy.gz')
Y_train = np.loadtxt('../Data/y_train_A.npy.gz')
X_test_PCA = np.loadtxt('../Data/X_test_A_pca.npy.gz')

test_dataset = mat73.loadmat('../Data/Test_A.mat')
event = np.array(test_dataset['event'])
test_str_A = test_dataset['test_str_A']

LDA_model = LinearDiscriminantAnalysis()
LDA_model.fit(X_train_PCA, Y_train)

prediction = LDA_model.predict_proba(X_test_PCA)


test_accuracy = accuracy_score(prediction, test_str_A, event)

print('acc_test: {}'.format(test_accuracy))