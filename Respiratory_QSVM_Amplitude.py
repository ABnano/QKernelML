from pylab import cm
from sklearn import metrics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.visualization import circuit_drawer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel, FidelityQuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.metrics import classification_report
from sklearn.svm import SVC

seed = 123
np.random.seed(seed)

adhoc_dimension = 2
X_train, y_train, X_test, y_test, adhoc_total = ad_hoc_data(
    training_size=50,
    test_size=10,
    n=adhoc_dimension,
    gap=0.5,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)

train_features, train_labels, test_features, test_labels = X_train, y_train, X_test, y_test

# plt.figure(figsize=(5, 5))
# plt.ylim(0, 2 * np.pi)
# plt.xlim(0, 2 * np.pi)
# plt.imshow(
#     np.asmatrix(adhoc_total).T,
#     interpolation="nearest",
#     origin="lower",
#     cmap="RdBu",
#     extent=[0, 2 * np.pi, 0, 2 * np.pi],
# )
#
# plt.scatter(
#     X_train[np.where(y_train[:] == 0), 0],
#     X_train[np.where(y_train[:] == 0), 1],
#     marker="s",
#     facecolors="w",
#     edgecolors="b",
#     label="A train",
# )
# plt.scatter(
#     X_train[np.where(y_train[:] == 1), 0],
#     X_train[np.where(y_train[:] == 1), 1],
#     marker="o",
#     facecolors="w",
#     edgecolors="r",
#     label="B train",
# )
# plt.scatter(
#     X_test[np.where(y_test[:] == 0), 0],
#     X_test[np.where(y_test[:] == 0), 1],
#     marker="s",
#     facecolors="b",
#     edgecolors="w",
#     label="A test",
# )
# plt.scatter(
#     X_test[np.where(y_test[:] == 1), 0],
#     X_test[np.where(y_test[:] == 1), 1],
#     marker="o",
#     facecolors="r",
#     edgecolors="w",
#     label="B test",
# )
#
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
# plt.title("Ad hoc dataset for classification")
#
# plt.show()

num_features = train_features.shape[1]
print("num_features", num_features)

# feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
# print(feature_map)


# from qiskit_machine_learning.kernels import QuantumKernel
ansatz = RealAmplitudes(1, reps=1)


# quant_kernel = FidelityQuantumKernel(feature_map=feature_map)

#with ansatz
quant_kernel = TrainableFidelityQuantumKernel(feature_map=ansatz)

QSVC_quantum = QSVC(
    quantum_kernel=quant_kernel,
)

# print shape of train_features
# print(train_features.shape)
# print(train_labels.shape)
#
# print(test_labels.shape)
# print(test_features.shape)

print("fitting")
QSVC_quantum.fit(train_features, train_labels)
print("fitted")
train_score_qsvc = QSVC_quantum.score(train_features, train_labels)
test_score_qsvc = QSVC_quantum.score(test_features, test_labels)
#
print(f"Quantum SVC on the training dataset: {train_score_qsvc:.2f}")
print(f"Quantum SVC on the test dataset:     {test_score_qsvc:.2f}")
print(classification_report(test_labels, QSVC_quantum.predict(test_features)))

svc = SVC()
_ = svc.fit(train_features, train_labels)
train_score_c4 = svc.score(train_features, train_labels)
test_score_c4 = svc.score(test_features, test_labels)

print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")
print(classification_report(test_labels, svc.predict(test_features)))


"""
Quantum SVC on the training dataset: 0.70
Quantum SVC on the test dataset:     0.60
              precision    recall  f1-score   support

           0       0.62      0.50      0.56        10
           1       0.58      0.70      0.64        10

    accuracy                           0.60        20
   macro avg       0.60      0.60      0.60        20
weighted avg       0.60      0.60      0.60        20

Classical SVC on the training dataset: 0.75
Classical SVC on the test dataset:     0.65
              precision    recall  f1-score   support

           0       0.80      0.40      0.53        10
           1       0.60      0.90      0.72        10

    accuracy                           0.65        20
   macro avg       0.70      0.65      0.63        20
weighted avg       0.70      0.65      0.63        20
"""