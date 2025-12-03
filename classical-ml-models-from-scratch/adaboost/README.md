# AdaBoost Implementation

This repository contains an implementation of **AdaBoost** (Adaptive Boosting) from scratch.

## 1. Weak Learner (Decision Stump)
The core component is a **Decision Stump**, which is a 1-level Decision Tree. It classifies samples based on a single feature and a threshold.

$$
h(x) = \begin{cases} 
1 & \text{if } x_i < \text{threshold} \\
-1 & \text{otherwise}
\end{cases}
$$

* **Classes:** $\{-1, 1\}$
* **Method:** Greedy search for the best feature and threshold that minimizes weighted error.

## 2. Error Calculation
The error is calculated differently depending on the iteration phase:

1.  **First Iteration:** $\frac{\text{misclassifications}}{N}$
2.  **Subsequent Iterations:** The sum of weights of misclassified samples.

$$
\text{Error}_t = \sum_{i=1}^{N} w_i \cdot \mathbb{I}(y_{true} \neq y_{pred})
$$

> **Note:** If $\text{Error}_t > 0.5$, the classifier is worse than random guessing. We **flip the decision** (invert predictions) and update the error:
> $$\text{Error}_t = 1 - \text{Error}_t$$

## 3. Performance (Alpha)
The weight of the stump's say ($\alpha$) in the final decision is based on its error. Lower error = higher alpha.

$$
\alpha = 0.5 \cdot \log\left(\frac{1 - \text{Error}_t}{\text{Error}_t}\right)
$$

## 4. Weights Update
Weights are updated to penalize misclassified samples (giving them higher impact in the next round) and reward correctly classified ones.

**Initialization:**
$$
w_0 = \frac{1}{N} \quad \text{for each sample}
$$

**Update Rule:**
$$
w_{new} = \frac{w_{old} \cdot \exp(-\alpha \cdot y \cdot h(x))}{\sum w_{new}}
$$

* Where $h(x)$ is the prediction of the current stump.
* Weights are normalized (divided by their sum) to ensure they sum to 1.

## 5. Final Prediction
The final strong classifier aggregates the votes of all weak learners, weighted by their alpha.

$$
Y = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \cdot h_t(X)\right)
$$

---

## 6. Training Algorithm

The full training loop proceeds as follows:

1.  **Initialize** weights $w = \frac{1}{N}$.
2.  **For** $t$ in $T$ (number of estimators):
    1.  **Train** weak classifier:
        * Find best feature and threshold.
    2.  **Calculate Error**:
        * Sum of weights of misclassified samples.
        * If `error > 0.5`: Flip decision and set `error = 1 - error`.
    3.  **Calculate Alpha**:
        * $\alpha = 0.5 \cdot \log(\frac{1-error}{error})$
    4.  **Update Weights**:
        * Apply update formula.
        * Normalize weights.
