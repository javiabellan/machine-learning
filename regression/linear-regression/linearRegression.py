import matplotlib.pyplot as plt
import numpy as np # from numpy import * (para ahorrar usar np.)

# y = mx + b
# m is slope, b is y-intercept
def computeError(m, n, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m*x + n)) ** 2
    return totalError / float(len(points))

def linearRegr(x, y):
    n      = len(x)
    xmean  = np.mean(x)
    ymean  = np.mean(y)

    slope     = (np.mean(x*y) - xmean*ymean) / (np.mean(x**2) - xmean**2) # cov(x,y) / var(x)
    intercept = ymean - slope*xmean

    return slope, intercept



# Points
points = np.genfromtxt("data.csv", delimiter=",")
x = points[:,0] # np.array([1,2,3,4,5,6], dtype=np.float32)
y = points[:,1] # np.array([5,4,6,5,6,7], dtype=np.float32)

# Compute linear regression
m, n = linearRegr(x, y)
print("After compute normal linear regression (y=mx+n) :")
print("Slope (m)     = ", m)
print("Intercept (n) = ", n)
print("Square error  = ", computeError(m, n, points) )

# Show plot
regressionLine = [m*X + n for X in x]
plt.axis('equal')
plt.scatter(x,y)
plt.plot(x, regressionLine)
plt.show()
