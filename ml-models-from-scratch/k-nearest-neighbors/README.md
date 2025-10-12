# K-Nearest Neighbors (KNN) from Scratch

In K-Nearest Neighbors, we aim to classify a new data point based on the labels of its closest neighbors in the training set. It's a simple, non-parametric algorithm that makes predictions by proximity.

<img width="498" height="450" alt="Image" src="https://github.com/user-attachments/assets/a9dd5d1b-b818-4914-bac3-6b832fbaaa2c" />
*Credit to [Tavish Srivastava]/[https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/] *

## Prediction by Majority

KNN doesn't "learn" a model. Instead, it memorizes the entire training dataset (a process known as **lazy learning**). To predict the class of a new data point, it performs the following steps:

1.  **Calculates** the distance from the new point to every point in the training data.
2.  **Identifies** the '$K$' nearest data points (the "neighbors").
3.  **Assigns** the new data point the class label that is most common among its $K$ neighbors (a **majority vote**).

***

## Distance Metrics: Measuring Closeness

To find the "nearest" neighbors, we need a function to measure the distance between two points. While several metrics exist, I used the following three:

### Minkowski Distance
The generalized form of distance metric.

$$d(p, q) = \left(\sum_{i=1}^{n} |p_i - q_i|^p\right)^{1/p}$$

### Manhattan Distance
Also known as "city block" distance, this is the distance you would travel between two points if you could only move along grid lines.

$$d(p, q) = \sum_{i=1}^{n} |p_i - q_i|$$

### Euclidean Distance
This is the most common metric, representing the straight-line "as the crow flies" distance between two points.

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

***

## The Prediction Algorithm

For each new point, we:
1.  **Compute Distances**: Iterate through all points in the training set (`X_train`) and compute the distance to the new point `x`.
2.  **Get Neighbors**: Sort the distances in ascending order and select the indices of the top `K` smallest distances.
3.  **Vote for Label**: Retrieve the labels (`y_train`) corresponding to these `K` indices and determine the most frequent label. This becomes the final prediction.

***

## The Number of Neighbors ($K$)

The choice of $K$ is a critical hyperparameter that influences the model's behavior and the shape of the decision boundary.

- **If $K$ is too low**: The model will be very sensitive to noise and outliers in the data. The decision boundary will be highly irregular, a condition known as **high variance**.
- **If $K$ is too high**: The model may consider neighbors that are too far away and from different classes, leading to an overly smooth decision boundary. This can result in misclassification, a condition known as **high bias**.