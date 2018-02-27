import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from itertools import chain
import math
from sklearn.preprocessing import normalize
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#This is the data generation stage!
#Data generation is through a linear model with some normal noise!
alpha0 = -27.8392
alpha1 = 38.123
alpha2 = -22.001
#Both the lists are independent of each other!
list_x1 = []
for x in range(1050):
    list_x1.append(random.uniform(0,100))
list_x2 = []
for x in range(1050):
    list_x2.append(random.uniform(0,100))
#temp_list is the fianl matrix of variables x1 and x2!
temp_list = list(zip(list_x1,list_x2))
#print(list_variables)
#list_y is the list of corresponding y values for the xi's
list_y = []
for (a,b) in temp_list:
    y_temp = alpha0 + alpha1*a + alpha2*b
    list_y.append(y_temp)
#print(list_y)
#print(list_y)
#Now we will add the noise to the y values!
mu,sigma = 0,1
random.seed(1)
s = np.random.normal(mu,sigma,1050)
#This part is just for viewing the noise(normal distributed variable)
#count, bins, ignored = plt.hist(s, 100, normed=True)
#plt.plot(bins,1/(sigma*np.sqrt(2*np.pi)) *
#          np.exp(-(bins-mu)**2/(2*sigma**2)),
#          linewidth=2, color='r')
#plt.show()
#Now the y values along with normal noises are generated!
list_y = list(list_y + s)

##Now we will normalize the y values in train_dataset!

list_y = np.array(list_y)
list_y = (list_y-np.mean(list_y))/np.max(list_y)

#Here the data type of list_y is np.ndarray
final_list = list(zip(list_y,list_x1,list_x2))
#print(final_list)
#This could be used as the subsetted dataset(or the training dataset)

new_list = list(chain(final_list[0:1000]))
test_list = list(chain(final_list[1000:1050]))
train_dataset = (np.matrix(new_list))
test_dataset = (np.matrix(test_list))
#print(len(new_list))
#print(np.shape(train_dataset))
#print(test_dataset)

#This is the Gaussian Process Regression Part!

#Firstly we will define a kernel function!

def kernel(x1,x2):
    sigma = 16
    l = 1
    prod = np.dot((x1-x2),(x1-x2).T)
    return (sigma**2)*math.exp(-prod/(2.0*(l**2)))

#print(kernel(np.array(0),np.array(3)))

#print(np.shape(list_y)[0])

#Here the kernel takes argument as a matrix and returns kernel matrix!
def kernel_matrix(x1,x2):
    k_matrix = np.empty([np.shape(x1)[0],np.shape(x2)[0]])
    for i in range(0,np.shape(x1)[0]):
        for j in range(0,np.shape(x2)[0]):
            k_matrix[i][j] = kernel(x1[i,:],x2[j,:])
            #print(k_matrix[i][j])
    return k_matrix

#A = np.array([0,0.2,1,3]).reshape(4,1)
#print(kernel(A[0,:],A[1,:]))

#print(kernel_matrix(A,A))

train_samples = 1000
test_samples = 50
##mean_star = np.zeros(test_samples)
##print(mean_star.shape)

#This function returns the mean only!
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

x_train = train_dataset[:,1:3]
y_train = train_dataset[:,0]
x_test = test_dataset[:,1:3]
y_test = test_dataset[:,0]

#(kernel(x_train[0,:],x_test[0,:]))

##a = np.matrix([[0],[0.2],[1],[3]])
##print(a.shape)
##b = np.matrix([[0],[0.2],[1]])
##print(kernel_matrix(a,b))

#print(y_test)
g = gaussian_process(x_train,y_train,x_test,train_samples,test_samples,sigma)
#print(g)

##This is for fitting the linear model in python
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
y_pred = regr.predict(x_test)
print("Mean squared error for the linear model is: %.10f"
      % mean_squared_error(y_test,y_pred))

## This is for printing the accuracy of GP regression!

error = g-y_test
#print(error)
sq_error = np.square(error)
#print(sq_error)
mean_sq_error = sum(sq_error)/sq_error.shape[0]
print("Mean squared error of GPR: %.10f"
      % mean_sq_error)



