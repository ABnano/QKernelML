import pandas as pd
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel, FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


def plot_classification_dataset(X_train, X_test, y_train, y_test):
    """
    Plot a classification dataset.

    Parameters:
    X: array-like, shape (n_samples, n_features)
        Feature matrix of the dataset.
    y: array-like, shape (n_samples,)
        Labels of the dataset.
    test_size: float, optional (default=0.3)
        The proportion of the dataset to include in the test split.
    random_state: int, optional (default=0)
        Controls the shuffling applied to the data before applying the split.
    """
    # Splitting the dataset into train and test sets

    # Creating a figure
    plt.figure(figsize=(15, 15))

    # Plotting the training and testing data
    classes = np.unique(y_train)
    print(classes)
    markers = ['s', 'o', '^', 'v', '*', '+']  # Add more markers if needed
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']  # Add more colors if needed

    for i, cls in enumerate(classes):
        plt.scatter(X_train[y_train == cls, 0], X_train[y_train == cls, 1], alpha=0.8, color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], label=f'Class {cls} train')
        plt.scatter(X_test[y_test == cls, 0], X_test[y_test == cls, 1], alpha=0.8, color=colors[i % len(colors)],
                    marker=markers[(i + len(classes)) % len(markers)], label=f'Class {cls} test')

    # Adding legend and title
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=1.0)
    plt.title("Dataset for classification (Iris)")

    # Showing the plot
    # plt.show()
    plt.savefig("iris.png")
    # plt.close()
    quit()


data = load_iris()
features = data.data
labels = data.target
#save as csv
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df["class"] = pd.Series(data.target)
# df.to_csv("iris.csv")
# quit()
features = MinMaxScaler().fit_transform(features)

df = pd.DataFrame(data.data, columns=data.feature_names)
df["class"] = pd.Series(data.target)

algorithm_globals.random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=algorithm_globals.random_seed
)

# plot_classification_dataset(train_features, test_features, train_labels, test_labels)

num_features = features.shape[1]

for i in range(1, 10):
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=i)
    # feature_map.decompose().draw(output="mpl", fold=-1)
    # plt.title("ZZFeatureMap for Iris Dataset: reps=1")
    # plt.savefig("iris_zzfeaturemap.png")
    # plt.show()
    # quit()

# from qiskit.circuit.library import RealAmplitudes
#
# ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
# ansatz.decompose().draw(output="mpl", fold=20)

    quant_kernel = TrainableFidelityQuantumKernel(feature_map=feature_map)

    # from qiskit_machine_learning.kernels import QuantumKernel

    QSVC_quantum = QSVC(
        quantum_kernel=quant_kernel,
        gamma=0.5,

    )
#
    _ = QSVC_quantum.fit(train_features, train_labels)
    train_score_qsvc = QSVC_quantum.score(train_features, train_labels)
    test_score_qsvc = QSVC_quantum.score(test_features, test_labels)

    print(i, train_score_qsvc, test_score_qsvc,sep=",")

# predictions = QSVC_quantum.predict(test_features)
# cm = confusion_matrix(test_labels, predictions)
# print(confusion_matrix)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=QSVC_quantum.classes_)
# disp.plot()
# plt.title("Confusion Matrix Quantum SVC on the Iris Dataset")
# plt.suptitle("reps=2, gamma=0.5")
# plt.savefig("iris_confusion_matrix_2.png")
# plt.show()
# # print(f"Quantum SVC on the training dataset: {train_score_qsvc:.2f}")
# print(f"Quantum SVC on the test dataset:     {test_score_qsvc:.2f}")
# print(classification_report(test_labels, QSVC_quantum.predict(test_features)))


svc = SVC()
_ = svc.fit(train_features, train_labels)
train_score_c4 = svc.score(train_features, train_labels)
# test_score_c4 = svc.score(test_features, test_labels)
# predictions = svc.predict(test_features)
# cm = confusion_matrix(test_labels, predictions)
# print(confusion_matrix)
#
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
# disp.plot()
# plt.title("Confusion Matrix for Iris Dataset (Classical)")
# plt.savefig("iris_confusion_matrix_classical.png")
# plt.show()

## plot support vectors
# plt.scatter(train_features[:, 0], train_features[:, 1], c=train_labels, s=50, cmap='autumn')  # , alpha=0.5)
# plt.scatter(QSVC_quantum.support_vectors_[:, 0], QSVC_quantum.support_vectors_[:, 1], s=50,
#             linewidth=1, facecolors='none', edgecolors='k')
# plt.show()

# print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
# print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")
# print(classification_report(test_labels, svc.predict(test_features)))

"""
Quantum SVC on the training dataset: 0.99
Quantum SVC on the test dataset:     0.97
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       0.86      1.00      0.92         6
           2       1.00      0.91      0.95        11

    accuracy                           0.97        30
   macro avg       0.95      0.97      0.96        30
weighted avg       0.97      0.97      0.97        30

Classical SVC on the training dataset: 0.99
Classical SVC on the test dataset:     0.97
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       0.86      1.00      0.92         6
           2       1.00      0.91      0.95        11

    accuracy                           0.97        30
   macro avg       0.95      0.97      0.96        30
weighted avg       0.97      0.97      0.97        30

"""
