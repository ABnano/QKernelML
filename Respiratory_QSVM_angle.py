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
import pennylane as qml

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

# dump adhoc dataset to excel
import pandas as pd






num_features = train_features.shape[1]
print("num_features", num_features)

# feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
# print(feature_map)

nqubits = 2

dev = qml.device("default.qubit", wires=nqubits)
# @qml.qnode(dev)
# def circuit(features):
#     qml.IQPEmbedding(features, wires=range(3))
#     return [qml.expval(qml.PauliZ(w)) for w in range(3)]

@qml.qnode(dev)
def circuit(weights, f=None):
    qml.QAOAEmbedding(features=f, weights=weights, wires=range(2))
    return qml.expval(qml.PauliZ(0))

features = [1., 2.]
layer1 = [0.1, -0.3, 1.5]
layer2 = [3.1, 0.2, -2.8]
weights = [layer1, layer2]

print(circuit(weights, f=features))

@qml.qnode(dev)
def kernel_circ(a, b):
    qml.AngleEmbedding(

        a, wires=range(nqubits))

    qml.adjoint(qml.AngleEmbedding(

        b, wires=range(nqubits)))

    return qml.probs(wires=range(nqubits))


def qkernel(A, B):
    return np.array([[circuit(a, b)[0] for b in B] for a in A])


QSVC_quantum = SVC(kernel=qkernel)

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
print(QSVC_quantum.support_vectors_)
#
print(f"Quantum SVC on the training dataset: {train_score_qsvc:.2f}")
print(f"Quantum SVC on the test dataset:     {test_score_qsvc:.2f}")
print(classification_report(test_labels, QSVC_quantum.predict(test_features)))

svc = SVC()
_ = svc.fit(train_features, train_labels)
train_score_c4 = svc.score(train_features, train_labels)
test_score_c4 = svc.score(test_features, test_labels)
print(svc.support_vectors_)
print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")
print(classification_report(test_labels, svc.predict(test_features)))


"""
Quantum SVC on the training dataset: 0.56
Quantum SVC on the test dataset:     0.55
              precision    recall  f1-score   support

           0       0.54      0.70      0.61        10
           1       0.57      0.40      0.47        10

    accuracy                           0.55        20
   macro avg       0.55      0.55      0.54        20
weighted avg       0.55      0.55      0.54        20

Classical SVC on the training dataset: 0.63
Classical SVC on the test dataset:     0.65
              precision    recall  f1-score   support

           0       0.64      0.70      0.67        10
           1       0.67      0.60      0.63        10

    accuracy                           0.65        20
   macro avg       0.65      0.65      0.65        20
weighted avg       0.65      0.65      0.65        20
"""