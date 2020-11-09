# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 19:12:52 2020

@author: jnke2
"""

#Linear Regression 
import numpy as np;
import matplotlib.pyplot as plt;
import random 

class Linear_Regression:
    def __init__(self):
        #defining all the hyper parapemeters
        self.iters=10000  #number of iterations
        self.alpha = 0.01 #learning rate
        """
        weight matrix which is a row vector of dimension n+1 where n is the number of features
        since we have 11 features, therefore omega will be a row vector of size 12. Can also be a column vector accordingly
        """
        self.omega=np.array([random.random() for i in range(12)]) #initialize the row vector to numbers between 0 and 1
        self.Cost_to_Omega={} #mapping from cost to omega
        
    def K_Fold_validation(self,data_test,data_train): #Split the data set into 5 equal part for trainng purposes
        
        x_test=np.array(data_test[:,:data_test.shape[1]-1]) #size 200 by 11
        y_test=np.array(data_test[:,-1]) #size 200 - row vector
        x_train=np.array(data_train[:,:data_train.shape[1]-1]) #size 800 by 11
        y_train=np.array(data_train[:,-1]) #size 800 - row vector
        
        #build the Design matrix by adding a row of ones X
        X_train=np.concatenate((x_train.T,np.ones([1,x_train.shape[0]])),0) #size 12 by 800
        
        self.Gradient_Descent(X_train,y_train)
        
    
    def Gradient_Descent(self,X,y): #update the parameters to minimize the cost
        #omega by x is of size 1 by 800 as well as y
        cost=np.zeros(self.iters)
        #Cost=np.array
        for i in range(self.iters):
            hypothesis=self.omega@X
            self.omega=self.omega-((self.alpha/X.shape[1])*((self.omega@X-y.T)@X.T)) #1 by 12
            cost[i]=self.Mean_Square_Error(hypothesis,y)
        #print(self.omega[0])
        self.Cost_to_Omega[cost[-1]]=self.omega
    
    def Best_Weight(self):
        return self.Cost_to_Omega[min(self.Cost_to_Omega)]
        
    
    def Mean_Square_Error(self,hypothesis,y): #To keep track if our cost is decreasing with each iteration
        cost=np.sum(np.power((hypothesis-y.T),2))
        #print(y.shape)
        cost=cost/(2*y.shape[0]) #1 will be out if index because cost is a row vector
        return cost
    
    def Mean_Square_Error_Test(self, Best_Omega,x,y): #this function is used for the 195 samples that have not being used
        X=np.concatenate((x.T,np.ones([1,x.shape[0]])),0)
        cost=np.sum(np.power((Best_Omega@X-y.T),2))
        cost=cost/(2*y.shape[0])
        return cost
    
    def Read_CSV(self,file):
        csv_gen = (row for row in open(file)) #csv_geb is a generator 
        my_data = np.loadtxt(csv_gen, delimiter=',', skiprows=1) # np.loadtxt takes a generator as input and return a list
        #for every single string, it converts it to an integer and the separation is done when it encounter a comma. This can't load a string
        #The generator needs to be only strings which can be converted to float
        #my_data is of size (1195,12)
        
        """
        partition of my_data into design matrix X and output y
        """
        x=np.array(my_data[:,:my_data.shape[1]-1]) #x is of size (1195, 11)
        y=np.array(my_data[:,-1]) #only take the last column which is the last column-  y is of size (1195, 1)
        i=0
        
        while(i<1000):
            data_test=my_data[i:i+200] #200 by 12
            data_train=np.array(list(set(map(tuple,my_data[:1000]))-set(map(tuple,data_test)))) #size 800 by 12
            self.K_Fold_validation(data_test,data_train) #leave the last 195 samples for final testing purposes
            i+=200
        Best_Omega=self.Best_Weight()
        cost=self.Mean_Square_Error_Test(Best_Omega,x[1000:],y[1000:]) #compute the cost with the best omega
        #print(cost)
        
L=Linear_Regression ()
L.Read_CSV("housing.csv.csv")




