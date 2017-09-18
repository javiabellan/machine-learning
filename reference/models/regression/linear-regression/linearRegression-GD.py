import matplotlib.pyplot as plt
from numpy import *

# Data
points       = genfromtxt("data.csv", delimiter=",")
# x = np.array([1,2,3,4,5,6], dtype=np.float32)
# y = np.array([5,4,6,5,6,7], dtype=np.float32)

# Gradient descent parameters
iterations   = 1000
learningRate = 0.0001
initial_m    = 0
initial_n    = 0

def computeError(m, n):
	totalError = 0
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m*x + n)) ** 2
	return totalError / float(len(points))

def stepGradient(m_current, n_current):
	m_gradient = 0
	n_gradient = 0
	N = float(len(points))

	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		m_gradient += -(2/N) * x * (y - ((m_current * x) + n_current))
		n_gradient += -(2/N) * (y - ((m_current * x) + n_current))

	new_m = m_current - (learningRate * m_gradient)
	new_n = n_current - (learningRate * n_gradient)

	return new_m, new_n

def linearRegr_gradDesc():
	m = initial_m
	n = initial_n

	for i in range(iterations):
		m, n = stepGradient(m, n)
	return m, n

def main():
	# Compute linear regression
	print("Performing gradient descent...")
	m, n = linearRegr_gradDesc()
	print("\nAfter ", iterations, " iterations:")
	print("Slope (m)     = ", m)
	print("Intercept (n) = ", n)
	print("Square error  = ", computeError(m, n) )

	# Show plot
	x = points[:, 0]
	y = points[:, 1]
	regressionLine = [m*X + n for X in x]
	plt.axis('equal')
	plt.scatter(x,y)
	plt.plot(x, regressionLine)
	plt.show()


if __name__ == '__main__':
    main()