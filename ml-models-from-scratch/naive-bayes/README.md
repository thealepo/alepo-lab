# Naive Bayes Classifier from Scratch

---

## Theory

This ML model is based on **Bayes' Theorem**, which provides a way to calculate the posterior probability of a class ($y$) given a set of features ($X$). (I also worked using Bayes' Theorem and Posteriors a lot this past summer!)

<img width="1344" height="960" alt="Image" src="https://github.com/user-attachments/assets/d47d72e8-6c6c-4a76-ac88-958e82c1c485" />

$$P(y | X) = \frac{P(X | y) \cdot P(y)}{P(X)}$$

Where
* $P(y | X)$ is the **posterior probability**.
* $P(X | y)$ is the **likelihood**.
* $P(y)$ is the **prior probability**.

### The "Naive" Assumption
The 'naive' part of the algorithm is that **it assumes that all the features are mutually independent.**

$$P(X | y) = P(x_1|y) \cdot P(x_2|y) \cdot ... \cdot P(x_n|y)$$

### Making a Prediction
To make a prediciton, we calculate the posterior probability for each possible class and choose the one with the highest probability. Since the denominator $P(X)$ is constant for all classes, we can simplify the decision to finding the `argmax` of the numerator:

$$ \hat{y} = \underset{y}{\mathrm{argmax}} \left( P(y) \prod_{i=1}^{n} P(x_i|y) \right) $$

For numerical stability (to avoid underflow from multiplying many small probabilities), we work with the sum of log-probabilities instead:

$$ \hat{y} = \underset{y}{\mathrm{argmax}} \left( \log(P(y)) + \sum_{i=1}^{n} \log(P(x_i|y)) \right) $$

### Calculating Probabilities
* **Prior Probability $P(y)$**: This is the frequency of each class in the training data.
* **Likelihood $P(x_i|y)$**: We assume that the features for each class follow a normal distribution. We calculate this from the Probability Density Function.

![Image](https://github.com/user-attachments/assets/63d5f9e4-4850-486f-ab76-cc9ff0949b61)

$$ P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} e^{-\frac{(x_i-\mu_y)^2}{2\sigma_y^2}} $$