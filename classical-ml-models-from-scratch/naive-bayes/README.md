# Naive Bayes from Scratch

In Naive Bayes, we aim to classify data points by applying **Bayes' Theorem** with a strong (naive) independence assumption between the features. It is a probabilistic classifier that predicts the class with the highest probability given the input features.

## Bayes' Theorem

The core of the algorithm is Bayes' Theorem, which describes the probability of an event, based on prior knowledge of conditions that might be related to the event.

$$
P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}
$$

- **$P(y|X)$ (Posterior)**: The probability of class $y$ being the correct output given the input features $X$. This is what we want to calculate.
- **$P(X|y)$ (Likelihood)**: The probability of seeing the features $X$ given that the class is $y$.
- **$P(y)$ (Prior)**: The initial probability of class $y$ occurring (frequency of class $y$ in the training set).
- **$P(X)$ (Evidence)**: The total probability of the features $X$ occurring. (Since this is constant for all classes, we often ignore it during comparison).

***

## The "Naive" Assumption

The algorithm is called "Naive" because it assumes that all features in $X$ are **mutually independent** given the class label. This allows us to simplify the Likelihood calculation:

$$
P(X|y) = P(x_1|y) \cdot P(x_2|y) \cdot ... \cdot P(x_n|y)
$$

Instead of calculating complex joint probabilities, we just multiply the individual probabilities of each feature.

***

## Gaussian Probability Density Function (PDF)

To calculate the likelihood $P(x_i|y)$ for continuous data, we assume the features follow a **Normal (Gaussian) Distribution**. We calculate the probability density using the mean ($\mu$) and variance ($\sigma^2$) of each feature for each class.

$$
P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \cdot \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)
$$

- **$\mu_y$**: The mean of feature $x_i$ for class $y$.
- **$\sigma_y^2$**: The variance of feature $x_i$ for class $y$.
- **$\exp$**: The exponential function ($e^x$).

***

## The Algorithm

The implementation is split into two phases: Training (Fit) and Prediction.

### 1. Training (Fit)
The "learning" phase involves calculating the statistics for each class in the dataset:
- **Calculate Priors**: The frequency of each class $y$ in the training data ($\frac{Count(y)}{Total Samples}$).
- **Calculate Statistics**: Compute the **Mean** ($\mu$) and **Variance** ($\sigma^2$) for every feature within each class.

### 2. Prediction
For a new data point $X_{new}$:
- **Calculate Likelihoods**: Use the Gaussian PDF equation to find $P(x_i|y)$ for every feature.
- **Calculate Posterior**: Multiply the Likelihoods by the Prior $P(y)$. (In practice, we often use **log-probabilities** to prevent numerical underflow).
- **Select Best Class**:
   $$
   y_{pred} = \arg\max_y (P(y) \prod_{i=1}^{n} P(x_i|y))
   $$
   The class with the highest resulting probability is our prediction.
