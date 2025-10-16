# Support Vector Machine (SVM) from Scratch

This project is a Python implementation of a linear Support Vector Machine (SVM) for binary classification, built from scratch using NumPy. The model is trained using Stochastic Gradient Descent (SGD).

## Core Concepts

An SVM is a supervised learning model that finds the optimal **hyperplane** to separate data points into two classes. The "best" hyperplane is the one that maximizes the **margin**, which is the distance between the hyperplane and the nearest data points from each class. These closest points are called **support vectors**.

### The Hyperplane

A linear hyperplane can be defined by the equation:

$$
w \cdot x - b = 0
$$

Where:
* `w` is the weight vector, normal to the hyperplane.
* `x` is the input feature vector.
* `b` is the bias term.

### Decision Rule

For a binary classification problem with labels $y \in \{-1, 1\}$, the decision rule for any data point $x_i$ is:

* If $w \cdot x_i - b \geq 1$, predict class $y_i = 1$.
* If $w \cdot x_i - b \leq -1$, predict class $y_i = -1$.

This can be combined into a single condition for all correctly classified points:

$$
y_i(w \cdot x_i - b) \geq 1
$$

## Optimization

To find the optimal `w` and `b`, we need to minimize a cost function. The SVM cost function consists of two parts: a regularization term to maximize the margin and a loss term to penalize misclassifications.

### Cost Function (Hinge Loss + Regularization)

The cost function is defined as:

$$
J = \lambda \|w\|^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(w \cdot x_i - b))
$$

Where:
* $\lambda$ is the regularization parameter. It controls the trade-off between maximizing the margin and minimizing classification error.

*this is a README i generated with Gemini... a more proper README will take its place in the upcoming days.*
