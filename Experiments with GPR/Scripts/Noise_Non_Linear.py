import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from itertools import chain
import math
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#This is the data generation stage!
#Data generation is through a linear model with some normal noise!
alpha0 = 0.57
alpha1 = 37.33
#We are making it a 1 variable plot!
alpha2 = -53.24
alpha3 = 21.15
alpha4 = -3.3929
alpha5 = 0.19166
#Both the lists are independent of each other!
list_x1 = []
for x in range(1000):
    list_x1.append(random.uniform(0,10))
list_x2 = []
for x in range(1000):
    list_x2.append(random.uniform(0,100))
#temp_list is the fianl matrix of variables x1 and x2!
temp_list = list(zip(list_x1,list_x2))
#print(list_variables)
#list_y is the list of corresponding y values for the xi's
list_y = []
for (a,b) in temp_list:
    y_temp = alpha0 + alpha1*a + alpha2*pow(a,2)+ alpha3*pow(a,3)+ alpha4*pow(a,4) + alpha5*pow(a,5)
    list_y.append(y_temp)
#print(list_y)
#print(list_y)
#Now we will add the noise to the y values!
mu,sigma = 0,10
random.seed(1)
s = np.random.normal(mu,sigma,1050)
#This part is just for viewing the noise(normal distributed variable)
#count, bins, ignored = plt.hist(s, 100, normed=True)
#plt.plot(bins,1/(sigma*np.sqrt(2*np.pi)) *
#          np.exp(-(bins-mu)**2/(2*sigma**2)),
#          linewidth=2, color='r')
#plt.show()
#Now the y values along with normal noises are generated!
list_y = list(list_y + s[0:1000])

##Now we will normalize the y values in train_dataset!

list_y = np.array(list_y)
train = list(zip(list_y,list_x1))
train = list(chain(train))
train = (np.matrix(train))
list_y = (list_y-np.mean(list_y))/np.mean(list_y)

#Here the data type of list_y is np.ndarray
#We will change it to 1 variable.
final_list = list(zip(list_y,list_x1))
#print(final_list)
#This could be used as the subsetted dataset(or the training dataset)

new_list = list(chain(final_list))
train_dataset = (np.matrix(new_list))

plt.plot(train_dataset[:,1],train_dataset[:,0],'ro')
plt.axis([0,10,-30,+10])
plt.show()

###########	Changing the range of test dataset!!

#Both the lists are independent of each other!
list1_x1 = []
for x in range(50):
    list1_x1.append(random.uniform(0,10))
#list1_x1 = list1_x1-300
list1_x2 = []
for x in range(50):
    list1_x2.append(random.uniform(0,10))
#list2_x2 = list2_x2-300
#temp_list is the final matrix of variables x1 and x2!
temp_list1 = list(zip(list1_x1,list1_x2))

list_y1 = []
for (a,b) in temp_list1:
    y_temp1 = alpha0 + alpha1*a + alpha2*pow(a,2)+ alpha3*pow(a,3)+ alpha4*pow(a,4) +alpha5*pow(a,5)
    list_y1.append(y_temp1)
#Now we will add the noise to the y values!
##mu1,sigma1 = 0,1
##random.seed(1)
##s1 = np.random.normal(mu1,sigma1,50)
list_y1 = list(list_y1 + s[1000:1050])

##Now we will normalize the y values in test_dataset!
list_y1 = np.array(list_y1)
#test and train is used in comparing with linear model!
test = list(zip(list_y1,list1_x1))
test = list(chain(test))
test = (np.matrix(test))
list_y1 = (list_y1-np.mean(list_y1))/np.mean(list_y1)

final_list1 = list(zip(list_y1,list1_x1))
new_list1 = list(chain(final_list1))
test_dataset = (np.matrix(new_list1))

plt.plot(test_dataset[:,1],test_dataset[:,0],'ro')
plt.axis([0,15,-1000,+1000])
plt.show()

############

#print(len(new_list))
#print(np.shape(train_dataset))
#print(test_dataset)

#This is the Gaussian Process Regression Part!

#Firstly we will define a kernel function!

def kernel(x1,x2):
    sigma = 1
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

x_train = train_dataset[:,1:]
y_train = train_dataset[:,0]
x_test = test_dataset[:,1:]
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
y_pred = regr.predict(test[:,1])
print("Mean squared error for the linear model is: %.10f"
      % mean_squared_error(y_test,y_pred))

poly = PolynomialFeatures(degree=4)
vector = train[:,0]
X_ = poly.fit_transform(train[:,1])
predict_ = poly.fit_transform(test[:,1])
clf = linear_model.LinearRegression()
clf.fit(X_, vector)
predictions = clf.predict(predict_)
error = predictions-y_test
sq_error = np.square(error)
mean_sq_err = sum(sq_error)/sq_error.shape[0]
print("Mean squared error for Polynomial_Regression is: %.10f"
      % mean_sq_err)

## This is for printing the accuracy of GP regression!

error = g-y_test
#print(error)
sq_error = np.square(error)
#print(sq_error)
mean_sq_error = sum(sq_error)/sq_error.shape[0]
print("Mean squared errorfor GPR is: %.10f"
      % mean_sq_error)



