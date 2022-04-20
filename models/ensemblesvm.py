import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import mat73
from utils.metrics import performance_measure

X_train = np.loadtxt('../Data/X_train_A_pca.npy.gz')
y_train = np.loadtxt('../Data/y_train_A.npy.gz')
X_test = np.loadtxt('../Data/X_test_A_pca.npy.gz')

test_dataset = mat73.loadmat('../Data/Test_A.mat')
event = np.array(test_dataset['event'])
test_str_A = test_dataset['test_str_A']

clf = BaggingClassifier(SVC(kernel='rbf', class_weight={1: 5}, C=0.5), n_estimators=20)

clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

accuracy, precision, recall, f1, support = performance_measure(y_pred, test_str_A, event)
print('acc_test: {}, precision: {}, recall: {}, f1:{}'.format(accuracy, precision, recall, f1))

