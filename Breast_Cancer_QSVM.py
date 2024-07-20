import pandas as pd
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel, FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from sklearn.svm import SVC
from matplotlib import pyplot as plt
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
    markers = ['s', 'o', '^', 'v', '*', '+']  # Add more markers if needed
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']  # Add more colors if needed

    for i, cls in enumerate(classes):
        plt.scatter(X_train[y_train == cls, 0], X_train[y_train == cls, 1], alpha=0.8, color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], label=f'Class {cls} train')
        plt.scatter(X_test[y_test == cls, 0], X_test[y_test == cls, 1], alpha=0.8, color=colors[i % len(colors)],
                    marker=markers[(i + len(classes)) % len(markers)], label=f'Class {cls} test')

    # Adding legend and title
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.0)
    plt.title("Dataset for classification (Breast Cancer)")

    # Showing the plot
    # plt.show()
    plt.savefig("breast_cancer.png")
    # plt.close()
    quit()


data = load_breast_cancer()
features = data.data
labels = data.target

#save to csv
df = pd.DataFrame(data.data, columns=data.feature_names)
df["class"] = pd.Series(data.target)
df.to_csv("breast_cancer.csv")



features = MinMaxScaler().fit_transform(features)

df = pd.DataFrame(data.data, columns=data.feature_names)
df["class"] = pd.Series(data.target)

algorithm_globals.random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=algorithm_globals.random_seed
)
# plot_classification_dataset(train_features, test_features, train_labels, test_labels)

print(train_features.shape)
pc = PCA(n_components=2)
train_features = pc.fit_transform(train_features)
test_features = pc.transform(test_features)


#save to csv
df = pd.DataFrame(train_features, columns=["pc1", "pc2"])
df["class"] = pd.Series(train_labels)
df.to_csv("breast_cancer_pca.csv")

quit()

print(train_features.shape)
num_features = train_features.shape[1]
print("num_features", num_features)

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", fold=20)
plt.title("ZZFeatureMap for Breast Cancer Dataset: reps=1")
plt.savefig("breast_cancer_zzfeaturemap_1.png")
plt.show()

# from qiskit.circuit.library import RealAmplitudes
#
# ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
# ansatz.decompose().draw(output="mpl", fold=20)


quant_kernel = FidelityQuantumKernel(feature_map=feature_map)

# from qiskit_machine_learning.kernels import QuantumKernel

QSVC_quantum = QSVC(
    quantum_kernel=quant_kernel,
    gamma=0.5
)

print("fitting")
QSVC_quantum.fit(train_features, train_labels)
print("fitted")
train_score_qsvc = QSVC_quantum.score(train_features, train_labels)
test_score_qsvc = QSVC_quantum.score(test_features, test_labels)
predictions = QSVC_quantum.predict(test_features)
cm = confusion_matrix(test_labels, predictions)
# print(confusion_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=QSVC_quantum.classes_)
disp.plot()
plt.title("Confusion Matrix Quantum SVC on the Breast Cancer Dataset")
plt.suptitle("reps=1, gamma=0.5")
plt.savefig("breast_cancer_confusion_1.png")

print(f"Quantum SVC on the training dataset: {train_score_qsvc:.2f}")
print(f"Quantum SVC on the test dataset:     {test_score_qsvc:.2f}")
print(classification_report(test_labels, QSVC_quantum.predict(test_features)))

svc = SVC()
_ = svc.fit(train_features, train_labels)
train_score_c4 = svc.score(train_features, train_labels)
test_score_c4 = svc.score(test_features, test_labels)

predictions = svc.predict(test_features)
cm = confusion_matrix(test_labels, predictions)
# print(confusion_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
disp.plot()
plt.title("Confusion Matrix for Breast Cancer Dataset (Classical)")
plt.savefig("breast_cancer_confusion_matrix_classical.png")
plt.show()


print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")
print(classification_report(test_labels, svc.predict(test_features)))

"""
Quantum SVC on the training dataset: 0.93
Quantum SVC on the test dataset:     0.93
              precision    recall  f1-score   support

           0       0.97      0.83      0.89        41
           1       0.91      0.99      0.95        73

    accuracy                           0.93       114
   macro avg       0.94      0.91      0.92       114
weighted avg       0.93      0.93      0.93       114

Classical SVC on the training dataset: 0.95
Classical SVC on the test dataset:     0.96
              precision    recall  f1-score   support

           0       1.00      0.90      0.95        41
           1       0.95      1.00      0.97        73

    accuracy                           0.96       114
   macro avg       0.97      0.95      0.96       114
weighted avg       0.97      0.96      0.96       114

"""
