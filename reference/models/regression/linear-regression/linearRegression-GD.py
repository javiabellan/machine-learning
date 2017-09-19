from mpl_toolkits.mplot3d import Axes3D
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
		x      = points[i, 0]
		target = points[i, 1]
		output = m*x + n
		totalError += (target - output)**2
	return totalError / float(len(points))

# Plot data
m_list     = [initial_m]
n_list     = [initial_n]
error_list = [computeError(initial_m,initial_n)]

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

def linearRegr_gradDesc():
	m = initial_m
	n = initial_n

	for i in range(iterations):
		m, n = stepGradient(m, n)
		m_list.append(m) # Save for plot
		n_list.append(n) # Save for plot
		error_list.append(computeError(m,n)) # Save for plot
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
	fig = plt.figure() # Contains all the plot elements
	
	# Plot 1 (data)
	x = points[:, 0]
	y = points[:, 1]
	regressionLine = [m*X + n for X in x]

	data = plt.subplot(121)
	data.axis('equal')
	data.scatter(x,y, color='cyan')
	data.plot(x, regressionLine)
	data.set_title('Data')

	# Plot 2 (gradient descent)
	m = arange(0,3,0.1)
	n = arange(-10,10,1)
	M, N = meshgrid(m, n)
	error = array([computeError(x,y) for x,y in zip(ravel(M), ravel(N))])
	Z = error.reshape(M.shape)

	grad = plt.subplot(122, projection='3d')
	grad.plot_surface(M, N, Z,cmap='Blues_r') # Plane
	grad.plot(m_list, n_list, error_list) # Line
	grad.set_title('Gradient Descent')
	grad.set_xlabel('slope (m)')
	grad.set_ylabel('y-intercept (n)')
	grad.set_zlabel('Error')


	plt.show()
	
if __name__ == '__main__':
    main()