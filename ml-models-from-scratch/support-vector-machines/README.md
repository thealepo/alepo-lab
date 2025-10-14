find best hyperplane that best seperates data
hyperplane

linear model
w*x - b = 0
w*xi - b >= 1  if yi=1
w8xi - b <= -1  if yi=-1
yi(w*xi - b) >= 1

cost function
hinge loss
l = max(0 , 1 - yi(w*xi - b))
l = {0 if y*f(x) >= 1 , 1-y*f(x) otherwise}

add regularization
J = lambda|w|^2 + 1/nsum(i=1,n, max(0,1-yi(wxi-b)))
if yi*f(x) >= 1:
Ji = lambda|w|^2
else:
Ji = lambda|w|^2 + 1 - yi(w*xi-b)

gradients
if yi*f(x) >= 1:
    dJi/dw_k = 2 lambda w_k
    dJi/db = 0
else:
    dJi/dw_k = 2 lambda w_k - yi * xi
    dJi/db = yi

update rule
for each training sample xi:
    w := -alpha * dw
    b := -alpha * db