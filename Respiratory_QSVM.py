from pylab import cm
from sklearn import metrics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.visualization import circuit_drawer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel, FidelityQuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC


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
    plt.title("Dataset for classification (Ad hoc)")

    # Showing the plot
    # plt.show()
    print("saving adhoc.png")
    plt.savefig("adhoc.png")
    # plt.close()
    quit()

adhoc_dimension = 2
X_train, y_train, X_test, y_test, adhoc_total = ad_hoc_data(
    training_size=30,
    test_size=5,
    n=adhoc_dimension,
    gap=0.5,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)

#save adhoc data to csv
#concatenate X_train, X_test
X = np.concatenate((X_train, X_test), axis=0)
#concatenate y_train, y_test
y = np.concatenate((y_train, y_test), axis=0)


# data = np.concatenate((X, y.reshape(-1,1)), axis=1)
# print(data)

#save as csv
# np.savetxt("adhoc1.csv", data, delimiter=",")
# #
# quit()








# np.savetxt("adhoc.csv", adhoc_total, delimiter=",")
# quit()

train_features, train_labels, test_features, test_labels = X_train, y_train, X_test, y_test
# plot_classification_dataset(X_train, X_test, y_train, y_test)

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

# for i in range(1, 10):
#
#     feature_map = ZZFeatureMap(feature_dimension=num_features, reps=i)
#     feature_map.decompose().draw(output="mpl", fold=50)
#
#     quant_kernel = FidelityQuantumKernel(feature_map=feature_map)
#
#     # from qiskit_machine_learning.kernels import QuantumKernel
#
#     QSVC_quantum = QSVC(
#         quantum_kernel=quant_kernel,
#     )
#
#     QSVC_quantum.fit(train_features, train_labels)
#     train_score_qsvc = QSVC_quantum.score(train_features, train_labels)
#     test_score_qsvc = QSVC_quantum.score(test_features, test_labels)
#     # predictions = QSVC_quantum.predict(test_features)
#     # cm = confusion_matrix(test_labels, predictions)
#     # print(confusion_matrix)
#     print(i, train_score_qsvc, test_score_qsvc)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=QSVC_quantum.classes_)
# disp.plot()
# plt.title("Confusion Matrix Quantum SVC on the Ad hoc Dataset")
# plt.suptitle("reps=5, gamma=0.5")
# plt.savefig("adhoc_confusion_5.png")
# plt.close()
#
# #
# print(f"Quantum SVC on the training dataset: {train_score_qsvc:.2f}")
# print(f"Quantum SVC on the test dataset:     {test_score_qsvc:.2f}")
# print(classification_report(test_labels, QSVC_quantum.predict(test_features)))

kernels = ["linear", "poly", "rbf", "sigmoid"]
gammas = [0.5, 0.1, 0.01, 0.001, 0.0001]
print("kernel, gamma, train_score, test_score")
for kernel in kernels:
    for gamma in gammas:
        svc = SVC(kernel=kernel, gamma=gamma)
        _ = svc.fit(train_features, train_labels)
        train_score_c4 = svc.score(train_features, train_labels)
        test_score_c4 = svc.score(test_features, test_labels)
        predictions = svc.predict(test_features)
        cm = confusion_matrix(test_labels, predictions)
        report = classification_report(test_labels, svc.predict(test_features))
        print(kernel, gamma, train_score_c4*100, test_score_c4*100, sep=",")
        # print(report)
    # print("*"*10)
#
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
# disp.plot()
# plt.title("Confusion Matrix for Ad hoc Dataset (Classical)")
# plt.savefig("adhoc_confusion_classical.png")
# plt.close()
# print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
# print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")
# print(classification_report(test_labels, svc.predict(test_features)))


"""
Quantum SVC on the training dataset: 0.90
Quantum SVC on the test dataset:     0.90
              precision    recall  f1-score   support

           0       1.00      0.80      0.89         5
           1       0.83      1.00      0.91         5

    accuracy                           0.90        10
   macro avg       0.92      0.90      0.90        10
weighted avg       0.92      0.90      0.90        10

Classical SVC on the training dataset: 0.68
Classical SVC on the test dataset:     0.40
              precision    recall  f1-score   support

           0       0.43      0.60      0.50         5
           1       0.33      0.20      0.25         5

    accuracy                           0.40        10
   macro avg       0.38      0.40      0.38        10
weighted avg       0.38      0.40      0.38        10
"""