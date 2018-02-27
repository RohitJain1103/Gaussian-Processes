import random
import numpy as np
from numpy.linalg import inv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from itertools import chain
import math
from sklearn.preprocessing import normalize
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pandas as pd

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

df1 = pd.read_csv("diesel_prop.csv")
df2 = pd.read_csv("diesel_spec.csv")

result = pd.concat([df1,df2],axis=1)

def kernel(x1,x2):
    sigma = 1
    l = 1
    prod = np.dot((x1-x2),(x1-x2).T)
    return (sigma**2)*math.exp(-prod/(2.0*(l**2)))

def kernel_matrix(x1,x2):
    k_matrix = np.empty([np.shape(x1)[0],np.shape(x2)[0]])
    for i in range(0,np.shape(x1)[0]):
        for j in range(0,np.shape(x2)[0]):
            k_matrix[i][j] = kernel(x1[i,:],x2[j,:])
            #print(k_matrix[i][j])
    return k_matrix

#This function returns the mean only!
#All the train and test are matrices!
def gaussian_process(x_train,y_train,x_test,train_samples,test_samples,sigma):
    mean = np.array([[0]]*train_samples)
    mean_star = np.array([[0]]*test_samples)
    diff = y_train - mean
    #print(y_train)
    k = kernel_matrix(x_train,x_train)
    k = k + (sigma**2)*np.identity(train_samples)
    inverse_k = inv(k)
    k_star_transpose = kernel_matrix(x_train,x_test).T
    dot1 = np.dot(inverse_k,diff)
    #print(inverse_k)
    return mean_star + np.dot(k_star_transpose,dot1)
##
##
##We first estimate for Viscousity!
##df = result[np.isfinite(result['VISC'])]
##
##temp = np.array(df['VISC'])
##tempm = np.mean(temp)
##temps = np.std(temp)
##
##dfn = np.matrix(df)
##
##x_train = dfn[0:300,8:]
##x_test = dfn[300:396,8:]
##y_train = (dfn[0:300,7]-tempm)/temps
##y_test = (dfn[300:396,7]-tempm)/temps
##
##train_samples = 300
##test_samples = 95
##sigma = 0.01
##
##
##g = gaussian_process(x_train,y_train,x_test,train_samples,test_samples,sigma)
##
##error = g-y_test
##sq_error = np.square(error)
##mean_sq_error = sum(sq_error)/sq_error.shape[0]
##print("Mean squared error: %.10f" % mean_sq_error)
##
##plt.plot(y_test,'r-',label='Actual Values')
##plt.plot(g,'b-',label='Predicted values')
##plt.legend()
##plt.xlabel('X')
##plt.ylabel('Viscosity_Normalized')
####plt.figure.savefig('test.jpg')
##plt.show()

##
##
#### We get MSE for viscosity as 0.1332196107.
##
##
##We estimate for BP50!
##df = result[np.isfinite(result['BP50'])]
##
##temp = np.array(df['BP50'])
##tempm = np.mean(temp)
##temps = np.std(temp)
##
##dfn = np.matrix(df)
##
##x_train = dfn[0:300,8:]
##x_test = dfn[300:396,8:]
##y_train = (dfn[0:300,1]-tempm)/temps
##y_test = (dfn[300:396,1]-tempm)/temps
##
##train_samples = 300
##test_samples = 95
##sigma = 0.01
##
##
##g = gaussian_process(x_train,y_train,x_test,train_samples,test_samples,sigma)
##
##error = g-y_test
##sq_error = np.square(error)
##mean_sq_error = sum(sq_error)/sq_error.shape[0]
##print("Mean squared error: %.10f" % mean_sq_error)
##
##
##plt.plot(y_test,'p-',label='Actual Values')
##plt.plot(g,'m-',label='Predicted values')
##plt.legend()
##plt.xlabel('X')
##plt.ylabel('BP50_Normalized')
##plt.show()


####We get MSE for BP50 as 0.1393631048.
##
####We estimate for CN!
##df = result[np.isfinite(result['CN'])]
##
##temp = np.array(df['CN'])
##tempm = np.mean(temp)
##temps = np.std(temp)
##
##dfn = np.matrix(df)
##
##x_train = dfn[0:300,8:]
##x_test = dfn[300:382,8:]
##y_train = (dfn[0:300,2]-tempm)/temps
##y_test = (dfn[300:382,2]-tempm)/temps
##
##train_samples = 300
##test_samples = 81
##sigma = 0.01
##
##
##g = gaussian_process(x_train,y_train,x_test,train_samples,test_samples,sigma)
##
##error = g-y_test
##sq_error = np.square(error)
##mean_sq_error = sum(sq_error)/sq_error.shape[0]
##print("Mean squared error: %.10f" % mean_sq_error)
##
#### We get MSE for CN as 0.5510231537.
##
##plt.plot(y_test,'g-',label='Actual Values')
##plt.plot(g,'y-',label='Predicted values')
##plt.legend()
##plt.xlabel('X')
##plt.ylabel('CN_Normalized')
####plt.figure.savefig('test.jpg')
##plt.show()
##
##
