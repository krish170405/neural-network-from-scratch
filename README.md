# Neural Network from Scratch (NumPy + Math Derivation)

This project implements a deep neural network **entirely from scratch** using **NumPy and raw mathematical derivations**. The goal was to build every component of a feedforward neural network without using any machine learning libraries.

##  Mathematical Foundation

The neural network architecture and learning process were modeled from first principles using linear algebra and multivariable calculus.

All mathematical derivations are included in the accompanying PDF: **`NN_Math_Scratch.pdf`**.

###  Topics Covered in the Math Notes:

- **Modeling the network mathematically**:
  - Layers, neurons, weights, and biases using matrix/vector notation
- **Forward Propagation**:
  - Linear transformations: \( Z = W \cdot A + b \)
  - Activation functions: ReLU for hidden layers, Softmax for output layer
- **Backpropagation Derivations**:
  - Step-by-step gradient derivation for the last two layers
  - Generalized gradient formula for any layer \( k \) (index notation)
- **Vectorized Reformulation**:
  - Converting indexed equations to vectorized format for computational efficiency
  - Mapping the math directly to NumPy operations for optimized matrix algebra

> 📄 All math derivations are documented in **`NN_Math_Scratch.pdf`** in this repository.

---

## 🛠 Implementation Details

The derived math was implemented in Python using NumPy:

- Manual weight initialization and architecture setup
- ReLU and Softmax activation functions
- Cross-entropy loss function
- Forward propagation and fully vectorized backpropagation
- Gradient descent for weight updates
- Accuracy evaluation on the MNIST dataset

> Final accuracy achieved: **93.20%** on test set with 32 neurons per hidden layer, learning rate 0.1, 3500 epochs.

---

## Project Goals

- Build a working deep neural network without libraries
- Understand every component of training and optimization
- Translate pure math into performant NumPy code

---

## Files

- `MNIST_NN_Scratch.ipynb` – Full training and testing code
- `NN_Math_Scratch.pdf` – Complete mathematical derivation and explanation of the model
