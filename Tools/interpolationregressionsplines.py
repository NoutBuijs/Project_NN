import numpy as np

def oneD_polyregression(x_array,y_array,order=3):
    X = np.ones((np.size(y_array),order))

    for i in range(1,order):
        X[:,i] = x_array**i
    a = np.dot(np.dot(np.linalg.inv((np.dot(X.T,X))),X.T),y_array)
    def f(x):
        X = np.ones((np.size(x),np.size(a)))
        for i in range(1,np.size(a)):
            X[:,i] = x**i
        return np.dot(X,a)
    return f,a
