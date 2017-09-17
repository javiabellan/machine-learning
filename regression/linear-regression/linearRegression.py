import matplotlib.pyplot as plt
import numpy as np


def linearReg(x, y):
    n      = len(x)
    xmean  = np.mean(x)
    ymean  = np.mean(y)

    slope     = (np.mean(x*y) - xmean*ymean) / (np.mean(x**2) - xmean**2) # cov(x,y) / var(x)
    intercept = ymean - slope*xmean

    return slope, intercept


x = np.array([1,2,3,4,5,6], dtype=np.float32)
y = np.array([5,4,6,5,6,7], dtype=np.float32)

m, n = linearReg(x, y)
print("m=", m, " n=",n)

regressionLine = [m*X + n for X in x]

plt.scatter(x,y)
plt.plot(x, regressionLine)
plt.show()
