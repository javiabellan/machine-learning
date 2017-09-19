## Linear model

This function generates a linear function depending of the number of variables (k).

 * K=1: a line
 * K=2: a plane
 * K>2: a hyperplane

**Function:** `output = w0 + w1*x1 + w2*x2 + ... wk*xk`


> #### Varaibles to optimize:
> `w0`, `w1`, `w2`


gradient = (target − output) *  xi
	w0_gradient = (target − output)
	w1_gradient = (target − output) * x1
	w2_gradient = (target − output) * x2


new_w = w + learningRate*gradient
	w0 = w0 + learningRate*w0_gradient
	w1 = w1 + learningRate*w1_gradient
	w2 = w2 + learningRate*w2_gradient

---

## Sigmoid

One of the interesting properties of the Sigmoid function is that the derivative can be expresed in terms of the function itself. The reason this is appealing is because it can save computation time, as we can just reuse the same calculated values for the forward pass, and for the gradient

**Function:** `output = 1 / (1 + e^(-x))`

**Derivative:** `gradient = output * (1-output)`

![imagen](https://github.com/javiabellan/machine-learning/blob/master/reference/math/images/sigmoid-derivative.jpg)



> #### Function:
> output = 1 / (1 + e^( -(w0 + w1*x1 + w2*x2) ))

> #### Varaibles to optimize:
> `w0`, `w1`, `w2`

==============================================

## Logistic regression or perceptron

> #### Function:
> output = 1 / (1 + e^( -(w0 + w1*x1 + w2*x2) ))

Varaibles to optimize:
	w0, w1, w2


gradiente = (target – output) * output * (1 – output) * xi
gradiente = (target − output) *  xi


new_w = w + learningRate * gradiente
new_w= w  + η (target−actual) xi
