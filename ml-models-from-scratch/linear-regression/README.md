# Linear Regression from Scratch

In simple linear regression, we aim to model a linear relationship between a single input feature ($x$) and a continuous target variable ($y$).

<img width="3024" height="2160" alt="Image" src="https://github.com/user-attachments/assets/a863dc81-e3ad-40ab-b87f-f2f8ddc47a6b" />
*Credit to [Arjun Moti]/[https://arjun-mota.github.io/posts/linear-regression/] *

## Approximation
$$
\hat{y} = wx + b
$$

-  **$w$ (weight)**: This is the **slope** of the line, which determines the influence that the input feature $x$ has on the predicted output $\hat{y}$.
-  **$b$ (bias)**: This is the **y-intercept**, which is the value of $\hat{y}$ when $x$ is zero.

Our goal is to find the optimal values for $w$ and $b$ that result in a line that best fits the data.

***

## Cost Function

To determine how well our line fits the data, we need to measure the error.

$$
Mean Squared Error (MSE) = J(w,b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \frac{1}{N} \sum_i{i=1}^{N} (y_i - (wx_i + b))^2
$$

-  **$N$**: Total number of data points in the training set.
-  **$(y_i - \hat{y}_i)^2$**: The squared difference for a single data point. We square the error to ensure it is always positive.
-  This function calculates the **average** of the squared errors. We want to minimize this function.

## Gradient Descent

**Gradient Descent** is an iterative optimization algorithm we use to find the minimum of a function. In this case, we use it to find the values of $w$ and $b$ that minimize the MSE cost function.

<img width="898" height="636" alt="Image" src="https://github.com/user-attachments/assets/e4a55ddb-0c5b-49cb-9ab4-86779323f376" />
*Credit to [Naveen]/[https://www.nomidl.com/machine-learning/gradient-descent-for-linear-regression/] *

To implement this, we need to calculate the partial derivatives of the cost function.

-  **Partial Derivative with respect to $w$ ($dw$):**
   $$
   \frac{\partial J}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} -x_i(y_i - (wx_i + b))
   $$

-  **Partial Derivative with respect to $b$ ($db$):**
   $$
   \frac{\partial J}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} -(y_i - (wx_i + b))
   $$

Using these gradients, we iteratively update our parameters:

$$
w := w - \alpha \cdot \frac{\partial J}{\partial w}
$$
$$
b := b - \alpha \cdot \frac{\partial J}{\partial b}
$$

-  **$\alpha$ (alpha)** is the **learning rate**, a hyperparameter that controls the step size in each iteration.

## The Learning Rate ($\alpha$)

<img width="791" height="394" alt="Image" src="https://github.com/user-attachments/assets/9bbbbc20-0617-4c4d-921d-4bf6f6aa3411" />
*Credit to [Naveen]/[https://www.nomidl.com/machine-learning/gradient-descent-for-linear-regression/] *

-  **If the learning rate is too low**: The model will learn very slowly, requiring many iterations to converge to the minimum.
-  **If the learning rate is too fast**: The algorithm can overshoot the minimum, causing the loss to possibly diverge, failing to find the minimum.

Choosing an appropriate learning rate is a key part of training an ML model.
