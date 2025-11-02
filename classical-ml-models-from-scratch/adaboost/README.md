adaboost
explain boosting
weak learner (decision stump)
    (x < threshold>)
       |       |
     class-1   class+1
error
    first iteration -> missclassification/num of samples
    after -> sum(miss) * weights
    if error > 0.5, flip the decision and error = 1-error
weights
    w0 = 1/N for each sample
    w = (w*exp(-alpha*y*h(X))/(sum(w))) , where h(X) is pred of t  (missclassified samples have higher impact for future)
performance
    alpha = 0.5 * log(1-et / et)
prediction
    y = sign(sum(t,T) alphat * h(X))
training
    initialize weights for each sample = 1/N
    for t in T:
        train weak classifier (greedy search for best feature and threshold)
        calculate error
            flip error and deicison error if >0.5
        calculate alpha
        update weights