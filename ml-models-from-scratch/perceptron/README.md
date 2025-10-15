# Perceptron

The **Perceptron** is the simplest form of a linear classifier, serving as the foundational computational unit of a neural network. It's designed to solve **binary classification problems** where the data is linearly separable.



***

## 1. Linear Model (Net Input)

The Perceptron's computation begins by calculating a **net input** ($z$), which is a linear combination of the input features ($\mathbf{x}$) and their corresponding weights ($\mathbf{w}$), plus a bias term ($b$).

$$
z = f(\mathbf{w}, b) = \mathbf{w}^T\mathbf{x} + b
$$

-   **$\mathbf{x}$ (Input Vector)**: The vector of input features ($\mathbf{x} = [x_1, x_2, \ldots, x_n]$).
-   **$\mathbf{w}$ (Weight Vector)**: The vector of weights ($\mathbf{w} = [w_1, w_2, \ldots, w_n]$). The weights determine the importance of each input feature.
-   **$b$ (Bias)**: A constant offset that allows the decision boundary to be shifted away from the origin.

***

## 2. Activation and Approximation

The net input ($z$) is passed through a **Unit Step Activation Function** ($g(z)$) to produce the final binary prediction ($\hat{y}$), which is the approximation of the true class ($y$).

$$
\hat{y} = g(z) = g(\mathbf{w}^T\mathbf{x} + b)
$$

The **Unit Step Function** (also known as the **Heaviside step function**) defines the prediction:

$$
g(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}
$$

*Note: The threshold $\theta$ from the initial definition is typically absorbed into the bias term $b$, simplifying the comparison to $0$.*

***

## 3. The Perceptron Learning Rule

The goal is to find the optimal $\mathbf{w}$ and $b$. Unlike models that use Gradient Descent to minimize a continuous loss function, the Perceptron uses a unique, discrete update rule that only activates upon **misclassification**.

The update for the weights ($\mathbf{w}$) and bias ($b$) is performed *for each training sample* $(\mathbf{x}_i, y_i)$ that is misclassified:

### Weight Update Rule:
$$
\mathbf{w} := \mathbf{w} + \Delta\mathbf{w}
$$
$$\Delta\mathbf{w} = \alpha \cdot (y_i - \hat{y}_i) \cdot \mathbf{x}_i$$

### Bias Update Rule:
$$
b := b + \Delta b
$$
$$\Delta b = \alpha \cdot (y_i - \hat{y}_i)$$

-   **$\alpha$ (Learning Rate)**: A value in $(0, 1]$ that controls the magnitude of the correction step.
-   **$(y_i - \hat{y}_i)$**: The classification error, which is the engine of the update.

***

## 4. Error-Driven Adjustment

The core idea is to push the weight vector in the direction of the correctly classified target class. This is determined solely by the sign of the error $(y_i - \hat{y}_i)$.

| True Class ($y_i$) | Predicted Class ($\hat{y}_i$) | Error ($y_i - \hat{y}_i$) | Effect on $\mathbf{w}$ |
| :---: | :---: | :---: | :--- |
| **1** | **0** | **+1** (False Negative) | $\mathbf{w}$ is adjusted by $+\alpha \cdot \mathbf{x}_i$, **pulling** the decision boundary toward the positive sample $\mathbf{x}_i$. |
| **0** | **1** | **-1** (False Positive) | $\mathbf{w}$ is adjusted by $-\alpha \cdot \mathbf{x}_i$, **pushing** the decision boundary away from the negative sample $\mathbf{x}_i$. |
| $y_i = \hat{y}_i$ | Correct | **0** | **No change** is made to $\mathbf{w}$ or $b$. |

This process is repeated until all samples are correctly classified or a maximum number of training epochs is reached.

***

## 💡 Key Property: Convergence

The Perceptron Learning Algorithm is guaranteed to converge to a separating hyperplane (the final optimal $\mathbf{w}$ and $b$) **if and only if** the training data is **linearly separable**. If the data is not linearly separable (e.g., the XOR problem), the algorithm will oscillate and never converge.
