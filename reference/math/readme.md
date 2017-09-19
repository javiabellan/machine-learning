## Name???

This function generates a line (k=1), a plane (k=2), or a hyperplane (k>2)

#### Function:
> #### Function:
> `output = w0 + w1*x1 + w2*x2`

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

### Function

`output = 1 / (1 + e^(-x))`

![imagen](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png)

### Derivative




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
