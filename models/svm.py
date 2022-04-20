import mat73
import numpy as np
from sklearn.svm import SVC
from utils.metrics import performance_measure

X_train_PCA = np.loadtxt('../Data/X_train_A_pca.npy.gz')
Y_train = np.loadtxt('../Data/y_train_A.npy.gz')
X_test_PCA = np.loadtxt('../Data/X_test_A_pca.npy.gz')

test_dataset = mat73.loadmat('../Data/Test_A.mat')
event = np.array(test_dataset['event'])
test_str_A = test_dataset['test_str_A']


clf = SVC(kernel='rbf', probability=True, C=0.5, class_weight={1: 5})

clf.fit(X_train_PCA, Y_train)

prediction = clf.predict_proba(X_test_PCA)
accuracy, precision, recall, f1, support = performance_measure(prediction, test_str_A, event)
print('acc_test: {}, precision: {}, recall: {}, f1:{}'.format(accuracy, precision, recall, f1))

