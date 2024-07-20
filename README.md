# QKernelML
# Quantum Machine Learning vs Classical Machine Learning

This repository contains code and resources for a comparative study between Quantum Support Vector Classification (QSVC) and classical Support Vector Classification (SVC) using an ad hoc dataset.

## Overview

This project aims to demonstrate the application of quantum machine learning algorithms and compare their performance against classical machine learning algorithms. We utilize the Qiskit framework for quantum machine learning and scikit-learn for classical machine learning.

## Setup

### Prerequisites

- Python 3.7 or higher
- Qiskit
- scikit-learn
- numpy
- matplotlib

### Installation

To set up the environment, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/quantum-vs-classical-ml.git
cd quantum-vs-classical-ml
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Code

### Quantum Machine Learning

The `quantum_ml.py` script demonstrates the quantum support vector classification. It includes the following steps:

1. Data preparation using the `respiratory data` function from Qiskit.
2. Creation of a parameterized quantum circuit for the feature map.
3. Optimization of the quantum kernel using the SPSA optimizer.
4. Training the QSVC model with the optimized kernel.
5. Evaluation of the QSVC model's performance.

To run the quantum machine learning script:

```bash
python quantum_ml.py
```

### Classical Machine Learning

The `classical_ml.py` script demonstrates classical support vector classification using scikit-learn. It includes the following steps:

1. Data preparation using the same `respiratory data` function for consistency.
2. Training the classical SVC model.
3. Evaluation of the SVC model's performance.

To run the classical machine learning script:

```bash
python classical_ml.py
```

## Conclusion

This project provides a comprehensive comparison between quantum and classical machine learning models using a synthetic dataset. It highlights the potential and current limitations of quantum machine learning algorithms.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Qiskit for providing the quantum computing framework.
- scikit-learn for the classical machine learning tools.

---

Feel free to modify the content as per your requirements. Happy coding!
