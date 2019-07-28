import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def hypothesisLogistic(X, theta):
    gamma = np.dot(X, theta)
    ans = 1/(1+ np.exp(-gamma))
    return ans

def costFunctionLogistic(h, y):
    m = h.shape[0]
    h1 = np.log(h)
    h2 = np.log(1-h)
    y1 = y
    y2 = 1-y
    p1 = np.dot(y1.transpose(), h1)
    p2 = np.dot(y2.transpose(), h2)
    ans = -(p1+p2)/m
    return ans[0]

def GradDescLogisticReg(X, y, alpha = 0.1, niter= 10000):
    jHist = np.zeros(shape= (1,niter))
    X = np.insert(X, 0, values = 1, axis = 1)
    theta = np.zeros(shape = (X.shape[1], 1))
    h= hypothesisLogistic(X, theta)
    m = X.shape[0]
    
    for i in range(0, niter):
        jHist[0][i] = costFunctionLogistic(h, y)
        theta = theta - alpha*np.dot(X.transpose(),h-y)/m
        h= hypothesisLogistic(X, theta)
        
    return (theta, jHist, h)

#X = np.array([1,2,3,4]).reshape(4,1)
#y = np.array([0,0,1,1]).reshape(4,1)

#(theta, jHist, h) = GradDescLogisticReg(X, y)
#print(h) 
#plt.scatter(X, y)
#plt.scatter(X,h)
#plt.plot([i for i in range(0,10000)], jHist[0])