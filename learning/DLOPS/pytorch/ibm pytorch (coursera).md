

# module 1

## logistic regression Cross Entropy Loss

outline
- problem with Mean Squared error
- Maximum likelihood
- Logistic regression with cross entropy loss
- pytorch


when training linear classifier 
	we want to minimize the misclassified samples
	if misclassified in training --> will mis-classify during training
	minimize the
		number of mislassified samples
		i.e. minimize the loss ----using--> cost function
	loss function $l(w , b)=\frac{1}{n}*\sum{n=1 to N}(y_{n} - \sigma(w*x_{n} + b))^2$
	