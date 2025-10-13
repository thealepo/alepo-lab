simulates behavior of one cell
input -> weights -> net input function -> activation function -> output

linear model
f(w,b) = w^Tx + b
Activation function
unit step function
g(x) = {1 if x >= theta , 0 otherwise}
<show graph>

approximation
y_hat = g(f(w,b)) = g(w^Tx + b)

perceptron update rule
for each training sample xi:
w := w + deltaW
deltaW := alpha * (yi-yi_hat)*xi
alpha: learning rate in [0,1]

update rule explanation
y , y_hat , y-y_hat
1 , 1 , 0
1 , 0 , 1
0 , 0 , 0
0 , 1 , -1
- weights are pushed towards positive or negative target class in case of misclassification.