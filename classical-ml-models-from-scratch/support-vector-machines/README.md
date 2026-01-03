# Support Vector Machine (SVM) Overview

Support Vector Machine (SVM) is a powerful and versatile **supervised machine learning algorithm** used for both classification and regression. However, it is most commonly used for classification problems.

## 🚀 How It Works
The goal of an SVM is to find the **optimal hyperplane** in an $N$-dimensional space (where $N$ is the number of features) that distinctly classifies the data points.

### Key Concepts

* **Hyperplane:** A decision boundary that separates different classes. In 2D, this is a line; in 3D, it's a plane.
* **Support Vectors:** These are the data points closest to the hyperplane. They are the "critical" elements of the dataset; if they were removed, the position of the hyperplane would change.
* **Margin:** The distance between the hyperplane and the nearest data point from either class. SVM aims to **maximize this margin** to ensure the model is robust.
* **Kernel Trick:** When data is not linearly separable, SVM uses "kernels" to project the data into a higher-dimensional space where a linear separation becomes possible.

---

## 🧠 Mathematical Intuition
The hyperplane is defined by the equation:
$$w \cdot x + b = 0$$

Where:
* $w$ is the weight vector (normal to the hyperplane).
* $x$ is the input feature vector.
* $b$ is the bias.

The algorithm solves an optimization problem to maximize the margin $M$:
$$M = \frac{2}{\|w\|}$$
])
print(f"Prediction: {predictions}")
