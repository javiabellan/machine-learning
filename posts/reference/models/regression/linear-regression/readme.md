# Linear regresion

[Linear regresion](https://en.wikipedia.org/wiki/Linear_regression) is the most basic type of regression and is about to trace a line that predics the most probale spot of a new point.

![image](https://github.com/javiabellan/machine-learning/blob/master/reference/models/regression/linear-regression/linearRegression.png)


## One step apprach ([code](https://github.com/javiabellan/machine-learning/blob/master/reference/models/regression/linear-regression/linearRegression.py))

#### More info
 * [Video](https://www.youtube.com/watch?v=SvmueyhSkgQ&index=8&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)


## Gradient descent apprach ([code](https://github.com/javiabellan/machine-learning/blob/master/reference/models/regression/linear-regression/linearRegression-GD.py))

Another way to obtain the line is using gradient descent:

 1. First, we declare initial values for the variables we want to optimize: 'm' and 'n'.
```python
initial_m = 0
initial_n = 0
 ```
 2. Then, we define an error function (square error) to describe how well our regression performs.
 
```python
def computeError(m, n):
	totalError = 0
	for i in range(len(points)):
		x      = points[i, 0]
		target = points[i, 1]
		output = m*x + n
		totalError += (target - output)**2
	return totalError / float(len(points))
```

 3. For knowing the direction to descent in each iteration of the gradient descent, we need to compute the partial derivatives of the variables respect to the function. Because the derivative of a function says if the function is incrasing or decreasing.

```python
def stepGradient(m, n):
	m_gradient = 0
	n_gradient = 0
	N = float(len(points))

	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]

		# Partial derivative respect 'm'
		m_gradient += -(2/N) * x * (y - (m*x + n))

		#Partial derivative respect 'n'
		n_gradient += -(2/N) * (y - (m*x + n))

	new_m = m - (learningRate * m_gradient)
	new_n = n - (learningRate * n_gradient)

	return new_m, new_n
```

4. Finally we iterate the gradient descent until we obtan a good solution (a minimun) for tha variables 'm' and 'n'.

```python
def linearRegr_gradDesc():
	m = initial_m
	n = initial_n

	for i in range(iterations):
		m, n = stepGradient(m, n)
	return m, n
```

```
After  1000  iterations:
Slope (m)     =  1.47774408519
Intercept (n) =  0.0889365199374
Square error  =  112.614810116
```

![image](https://github.com/javiabellan/machine-learning/blob/master/reference/models/regression/linear-regression/linearReg-gradDesc.png)

#### More info
 * [Siraj inntroductory video](https://youtu.be/UIFMLK2nj_w?t=2m)
 * [Detailed explanation](https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/)
 * [A github repo](https://github.com/alberduris/The_Math_of_Intelligence/tree/master/Week1)

## One step vs gradient descent

 * [link 1](https://stackoverflow.com/questions/18191890/why-gradient-descent-when-we-can-solve-linear-regression-analytically)
 * [link 2](https://stats.stackexchange.com/questions/278755/why-use-gradient-descent-for-linear-regression-when-a-closed-form-math-solution)
