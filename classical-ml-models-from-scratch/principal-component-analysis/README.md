# Principle Component Analysis (PCA) from Scratch

**Principle Component Analysis (PCA)** is an unsupervised dimensionality reduction technique, used to find  a new set of dimensions called **principle components** that are linearly independent and ordered by amount of variance from original data.

This can help us reduce the number of features in a dataset while retaining as much information (variance) as possible, projecting the data into a smaller dimension.


## Goal: Maximizing Variance

The core idea of PCA is to find a new set of axes for the data.
-   The **1st Principal Component ($PC_1$)** is the axis that captures the *largest possible variance* in the data.
-   The **2nd Principal Component ($PC_2$)** is the axis that captures the *next largest variance*, with the constraint that it must be **orthogonal** (at a 90-degree angle) to $PC_1$.
-   This continues for $d$ components, with each subsequent component capturing the maximum remaining variance while being orthogonal to all previous components.

By choosing the top $k$ components, we create a compressed representation of the data that is uncorrelated.

***

## Key Concepts: Covariance & Eigenvectors

We rely on:

### 1. The Covariance Matrix ($\Sigma$)

$$
\Sigma = \frac{1}{N} \sum_{i=1}^{N} (X_i - \bar{X})(X_i - \bar{X})^T
$$

Or, more simply, using mean-centered data $X_c = X - \bar{X}$:
$$
\Sigma = \frac{1}{N} X_c^T X_c
$$

-   **$N$**: Total number of data points.
-   **$X_c$**: The $N \times d$ data matrix, where the mean of each feature (column) has been subtracted out.
-   The **diagonal** of $\Sigma$ (e.g., $Cov(X_j, X_j)$) contains the **variance** of each feature.
-   The **off-diagonal** (e.g., $Cov(X_j, X_k)$) contains the **covariance** between pairs of features, indicating how they move together.

### 2. Eigendecomposition ($Av = \lambda v$)

Once we have the covariance matrix $\Sigma$, we can "solve" it using eigendecomposition. This process breaks the matrix down into its fundamental directions (eigenvectors) and their magnitudes (eigenvalues).

$$
\Sigma v = \lambda v
$$

-   **$v$ (Eigenvector)**: These are the **directions** of the new axes. The eigenvectors of the covariance matrix *are* the **principal components**. They point in the directions of maximum variance.
-   **$\lambda$ (Eigenvalue)**: This is a scalar value that represents the **magnitude** associated with its eigenvector. It tells us *how much variance* is captured by that principal component.

***

## The PCA Algorithm

1.  **Standardize the Data**
    Subtract the mean from each feature (column) so the data is centered at the origin.
    $$
    X_c = X - \bar{X}
    $$

2.  **Calculate the Covariance Matrix ($\Sigma$)**
    Find the $d \times d$ covariance matrix from the mean-centered data.
    $$
    \Sigma = \frac{1}{N} X_c^T X_c
    $$

3.  **Compute Eigenvectors & Eigenvalues**
    Find the eigenvectors ($v$) and eigenvalues ($\lambda$) of the covariance matrix $\Sigma$.

4.  **Sort Components**
    Sort the eigenvector-eigenvalue pairs in **descending order** based on the eigenvalues ($\lambda$). The eigenvector with the largest eigenvalue is $PC_1$.

5.  **Choose $k$ Components**
    Decide how many dimensions ($k$) to keep.
    We often choose $k$ such that we keep 95% or 99% of the total variance. These top $k$ eigenvectors form our **projection matrix $W$**.

6.  **Transform the Data**
    Project the original $N \times d$ data ($X_c$) into the new $k$-dimensional space by multiplying it with the projection matrix $W$ (which is $d \times k$).
    $$
    Z = X_c W
    $$

***

## The New Feature Space

The resulting matrix $Z$ is our new $N \times k$ dataset. This new data has several important properties:
-   It has **$k$ dimensions** instead of the original $d$.
-   The new features (the columns of $Z$) are **linearly independent** (uncorrected).
-   This makes the data much more efficient for machine learning algorithms, as it removes multicollinearity and reduces the "curse of dimensionality."







