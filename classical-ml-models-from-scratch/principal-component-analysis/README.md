*tentative*

pca finds a new set of dimensions such that all the dimensions are orthogonal.
to find these principle components, we need the variance of a sample X (1/n * sum(Xi - X_mean)^2)

covariance matrix
Cov(X,Y) , Cov(X,X)

eigenvector , eigenvalues
Av = lambda v

approach:
- subtract mean from X
- calculate Cov(X,X)
- calculate eigenvectors and eigenvalues
- sort them in decreasing order according to eigenvalues
- choose first k eigenvectors and that will be the new k dimensions
- transform the original n dimensional data points into k dimensions

**new data is linearly independent