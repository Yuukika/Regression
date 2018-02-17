import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.linalg import inv

def featureNormalize(X):
    mu = X.mean(0)
    sigma = X.std(0)
    X_norm = (X - mu)/ sigma
    return mu,sigma,X_norm

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters,1))
    for i in np.arange(num_iters):
        theta = theta - alpha / m * (X.T).dot(X.dot(theta) - y)
        J_history[i] = computeCostMulti(X, y, theta)
    return theta,J_history

def computeCostMulti(X, y, theta):
    m = y.size
    return 1/2/m*sum((X.dot(theta) - y)** 2)


data = np.loadtxt('ex1data2.txt',delimiter = ',')

X = data[:, 0:2]
y = data[:, 2]
m = y.size

#print(X[0:10,:])
#print(y[0:10])

mu,sigma,X = featureNormalize(X)
X = np.c_[np.ones((m,1)),X]
#print(X[0:10,:])
y = np.transpose([y])

alpha = 0.01
num_iters = 400

theta = np.zeros((3,1))

theta,J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
print(theta)
price =np.append(np.ones(1), (([1650, 3] - mu) / sigma)).dot(theta)
print(price)
plt.plot(J_history,'-')
plt.show()


theta_normalEqn  = ((inv(X.T.dot(X))).dot(X.T)).dot(y)
print(theta_normalEqn)