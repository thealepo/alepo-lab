# Decision Tree from Scratch

A Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It models decisions and their possible consequences as a tree-like structure.

## Structure
The tree consists of three main components:
- **Root Node**: Represents the entire population or sample data.
- **Decision Nodes**: Sub-nodes that split into further sub-nodes based on a specific feature threshold.
- **Leaf Nodes (Terminal Nodes)**: Nodes that do not split; they hold the final prediction (class label or value).

Our goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

***

## Impurity Measures

To determine the best way to split the data, we need a metric to measure the "purity" of a split. In classification, we often use **Entropy** or **Gini Impurity**.

### Entropy
Entropy measures the amount of uncertainty or disorder in a dataset.

$$
H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)
$$

- **$S$**: The current set of data samples.
- **$c$**: The number of unique classes in the target variable.
- **$p_i$**: The proportion of samples belonging to class $i$.

### Gini Impurity
Alternatively, Gini Impurity measures the probability of incorrectly classifying a randomly chosen element.

$$
Gini(S) = 1 - \sum_{i=1}^{c} (p_i)^2
$$

We want to **minimize** impurity. A node with an entropy or Gini of 0 is considered "pure" (all samples belong to the same class).

***

## Information Gain

We select the best feature to split on by calculating the **Information Gain**. This measures the reduction in entropy (or impurity) achieved by splitting the dataset $S$ on an attribute $A$.

$$
IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
$$

- **$H(S)$**: The entropy of the parent node (before the split).
- **$S_v$**: The subset of examples where attribute $A$ has value $v$.
- **$|S_v|$**: The number of samples in the subset.
- **$|S|$**: The total number of samples in the parent node.

The algorithm iterates through all features and thresholds, calculates the Information Gain for each, and chooses the split that yields the **highest** gain.

***

## Recursive Splitting

The tree is built using a **Greedy Approach** (Recursive Partitioning):

1. **Start at the Root**: Calculate the impurity of the current dataset.
2. **Find Best Split**: Iterate through every feature and every possible threshold to find the one with the highest Information Gain.
3. **Split**: Divide the dataset into left and right branches based on that threshold.
4. **Repeat**: Recursively apply steps 1-3 to the child nodes.

The recursion stops when a stopping criterion (Hyperparameter) is met.

***

## Hyperparameters (Stopping Criteria)

If we allow the tree to grow indefinitely, it will memorize the training data, leading to **Overfitting**. To prevent this, we implement stopping criteria.

### Max Depth
- **Description**: The maximum length of the path from the root to a leaf.
- **Effect**: Limits the complexity of the model. A lower depth prevents the model from learning highly specific patterns that might be noise.

### Min Samples Split
- **Description**: The minimum number of samples required to split an internal node.
- **Effect**: Prevents the algorithm from splitting nodes that contain very few samples, which helps in generalizing the model.

Controlling these hyperparameters is essential for creating a generalized model that performs well on unseen data.
