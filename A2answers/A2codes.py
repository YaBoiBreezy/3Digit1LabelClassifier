# COMP3105 Assignment 2
# Alexander Breeze 101 143 291
# Victor Litzanov 101143028

import numpy as np
import scipy as sp
from scipy import optimize
import cvxopt as cv
from A2helpers import *

#scipy.optimize.minimize to solve unconstrained problems
#cvxopt.solvers.qp to solve quadratic programmings

######### Question 1 #########

'''
Part a
Takes n*d input matrix X, n*1 label vector y, reg h-param lamb >0, returns d*1 vector w and intercept w0
'''
def minBinDev(X, y, lamb):
    
    def obj_func(u):
        w0 = u[-1]
        w = u[:-1]
        w = w[:, None]
        
        loss = np.sum(np.logaddexp(0.0, - y * (X @ w + w0))) #Binomial loss function
        reg = (0.5 * lamb * float(w.T @ w)) #L2 loss of weights, multiplied by lambda
    
        return loss + reg
    
    initial_u = np.ones(X.shape[1]+1)
    sol = sp.optimize.minimize(obj_func, initial_u)
    
    w = sol['x'][:-1][:,None]
    w0 = sol['x'][-1]

    return w, w0

'''
Part b
Takes n*d input matrix X, n*1 label vector y, reg h-param lamb >0, returns d*1 vector w and intercept w0
'''
def minHinge(X, y, lamb):
    n = X.shape[0]
    d = X.shape[1]

    TopHalf = np.concatenate((np.zeros((n,d)), np.zeros((n,1)), - np.identity(n)),axis=1)  #setting up constraints
    BottomHalf = np.concatenate((- (np.diagflat(y) @ X), - y, - np.identity(n)),axis=1)
    G = np.concatenate((TopHalf,BottomHalf),axis=0)
    
    h = np.concatenate((np.zeros((n,1)),- np.ones((n,1))),axis=0) #setting up constraints
    
    P1 = np.concatenate((lamb*np.identity(d), np.zeros((d,1)), np.zeros((d,n))),axis=1)   #Values to minimize
    P2 = np.concatenate((np.zeros((1,d)), np.zeros((1,1)), np.zeros((1,n))),axis=1)
    P3 = np.concatenate((np.zeros((n,d)), np.zeros((n,1)), np.zeros((n,n))),axis=1)
    P = np.concatenate((P1,P2,P3),axis=0)
    P = P + (1e-8) * np.eye(d+n+1)
    
    q = np.concatenate((np.zeros((d+1,1)),np.ones((n,1))),axis=0) #values to minimize
    
    G = G * 1.0
    h = h * 1.0
    P = P * 1.0
    q = q * 1.0
    
    G = cv.matrix(G)
    h = cv.matrix(h) #convert to cvxopt arrays
    P = cv.matrix(P)
    q = cv.matrix(q)

    cv.solvers.options['show_progress'] = False
    sol=cv.solvers.qp(P, q, G, h) #solve linear programming problem
    
    return np.array(sol['x'])[:d], np.array(sol['gap']) #return weights, intercept

'''
Part c
Takes mxd input Xtest, dx1 vector of weights/parameters w and scalar intercept w0, returns mx1 prediction vector yhat
'''
def classify(Xtest, w, w0):
    #y=sign(Xtest.T@w+w0)
    inner=(Xtest@w)+w0
    result=np.sign(inner)   #converting inout to 1 and -1 based on Xtest, w, w0 
    return result
    
'''
Part d
'''
def synExperimentsRegularize():
    n_runs = 100
    n_train = 100 #number of training data points
    n_test = 1000 #number of test data points
    lamb_list = [0.001, 0.01, 0.1, 1.] #lambda values to test
    gen_model_list = [1, 2, 3] #models to test
    
    train_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    
    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n = n_train, gen_model = gen_model)  #generate all data
                Xtest, ytest = generateData(n = n_test, gen_model = gen_model)
                w, w0 = minBinDev(Xtrain, ytrain, lamb)  #train on binomial deviation
                train_acc_bindev[i, j, r] = np.mean(ytrain == classify(Xtrain,w,w0)) # compute accuracy on training set
                test_acc_bindev[i, j, r]  = np.mean(ytest == classify(Xtest,w,w0)) # compute accuracy on test set
                w, w0 = minHinge(Xtrain, ytrain, lamb) #train on hinge deviation
                train_acc_hinge[i, j, r] = np.mean(ytrain == classify(Xtrain,w,w0)) # compute accuracy on training set
                test_acc_hinge[i, j, r]  = np.mean(ytest == classify(Xtest,w,w0)) # compute accuracy on test set
    
    train_bin = np.zeros((len(lamb_list), len(gen_model_list))) # compute the average accuracies over runs
    train_hin = np.zeros((len(lamb_list), len(gen_model_list)))
    test_bin = np.zeros((len(lamb_list), len(gen_model_list)))
    test_hin = np.zeros((len(lamb_list), len(gen_model_list)))
    
    for l in range(len(lamb_list)): #get averages
        for j in range(len(gen_model_list)):
            for r in range(n_runs):
                train_bin[l, j] += train_acc_bindev[l, j, r] / float(n_runs)
                test_bin[l, j]  += test_acc_bindev[l, j, r]  / float(n_runs)
                train_hin[l, j] += train_acc_hinge[l, j, r]  / float(n_runs)
                test_hin[l, j]  += test_acc_hinge[l, j, r]   / float(n_runs)

    train_acc = np.concatenate( (train_bin, train_hin), axis=1) # combine accuracies (bindev and hinge)
    test_acc  = np.concatenate( (test_bin, test_hin), axis=1)
    return train_acc, test_acc

######### Question 2 #########

'''
Part a
'''
def adjBinDev(X, y, lamb, kernel_func):
    K = kernel_func(X,X)
    
    def obj_func(u):
        a0 = u[-1]
        a = u[:-1]
        a = a[:, None]
        
        return np.sum(np.logaddexp(0.0, - y*(K.T @ a + a0))) + (0.5 * lamb * a.T @ K @ a) #adjoint binomial deviation equation
    
    initial_u = np.ones(K.shape[1]+1)
    sol = sp.optimize.minimize(obj_func, initial_u)
    
    a = sol['x'][:-1][:,None]
    a0 = sol['x'][-1]
    
    return a, a0
    
'''
Part b
'''
def adjHinge(X, y, lamb, kernel_func):
    n = X.shape[0]
    d = X.shape[1]

    K = kernel_func(X,X)
    #minimize ( x.T @ P @ x + q.T x ) s.t. G@x<=h

    TopHalf = np.concatenate((np.zeros((n,n)), np.zeros((n,1)), - np.identity(n)),axis=1)  #n x 2n+1
    BottomHalf = np.concatenate((- (np.diagflat(y) @ K), - y, - np.identity(n)),axis=1)    #n x 2n+1
    G = np.concatenate((TopHalf,BottomHalf),axis=0) #limiter
    
    h = np.concatenate((np.zeros((n,1)),- np.ones((n,1))),axis=0) #2n x 1   limiter
    
    P1 = np.concatenate((lamb * K, np.zeros((n,1)), np.zeros((n,n))),axis=1)
    P2 = np.concatenate((np.zeros((1,n)), np.zeros((1,1)), np.zeros((1,n))),axis=1) #minimize
    P3 = np.concatenate((np.zeros((n,n)), np.zeros((n,1)), np.zeros((n,n))),axis=1)
    P = np.concatenate((P1,P2,P3),axis=0) #2n+1 x 2n+1
    P = P + (1e-8) * np.eye(n+n+1)
    
    q = np.concatenate((np.zeros((n+1,1)),np.ones((n,1))),axis=0) #minimize
    
    G = G * 1.0
    h = h * 1.0
    P = P * 1.0
    q = q * 1.0
    
    G = cv.matrix(G)
    h = cv.matrix(h) #convert to cvxopt arrays
    P = cv.matrix(P)
    q = cv.matrix(q)

    cv.solvers.options['show_progress'] = False
    sol=cv.solvers.qp(P, q, G, h) #solve linear programming problem
    
    return np.array(sol['x'])[:n], np.array(sol['primal objective'])

'''
Part c
Takes m×d test matrix Xtest, n×1 vector of weights/parameters a, scalar intercept a0, n×d training input matrix X,
kernel function kernel func. Returns m×1 prediction vector yhat
'''
def adjClassify(Xtest, a, a0, X, kernelFunc):
    kf = kernelFunc(Xtest,X)
    return np.sign((kf@a)+a0)

'''
Part d
'''
def synExperimentsKernel():
    n_runs = 10
    n_train = 100 #number of training data points
    n_test = 1000 #number of test data points
    lamb = 0.001
    
    kernel_list = [lambda X1, X2: linearKernel(X1, X2),
        lambda X1, X2: polyKernel(X1, X2, 2),
        lambda X1, X2: polyKernel(X1, X2, 3),
        lambda X1, X2: gaussKernel(X1, X2, 1.0),
        lambda X1, X2: gaussKernel(X1, X2, 0.5),
        lambda X1, X2: gaussKernel(X1, X2, 0.1)]
        
    gen_model_list = [1, 2, 3]
    
    train_acc_bindev = np.zeros( [len(kernel_list), len(gen_model_list), n_runs] )
    test_acc_bindev  = np.zeros( [len(kernel_list), len(gen_model_list), n_runs] )
    train_acc_hinge  = np.zeros( [len(kernel_list), len(gen_model_list), n_runs] )
    test_acc_hinge   = np.zeros( [len(kernel_list), len(gen_model_list), n_runs] )
    
    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n = n_train, gen_model = gen_model) #generate all data
                Xtest,  ytest  = generateData(n = n_test,  gen_model = gen_model)
                a, a0 = adjBinDev(Xtrain, ytrain, lamb, kernel) #train on binomial deviation
                train_acc_bindev[i, j, r] = np.mean(ytrain == adjClassify(Xtrain, a, a0, Xtrain, kernel)) # compute accuracy on training set
                test_acc_bindev[i, j, r]  = np.mean(ytest  == adjClassify(Xtest,  a, a0, Xtrain, kernel)) # compute accuracy on test set
                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel) #train on hinge deviation
                train_acc_hinge[i, j, r] = np.mean(ytrain == adjClassify(Xtrain, a, a0, Xtrain, kernel)) # compute accuracy on training set
                test_acc_hinge[i, j, r]  = np.mean(ytest  == adjClassify(Xtest,  a, a0, Xtrain, kernel)) # compute accuracy on test set
    
    train_bin = np.zeros( (len(kernel_list), len(gen_model_list) ) ) # compute the average accuracies over runs
    train_hin = np.zeros( (len(kernel_list), len(gen_model_list) ) )
    test_bin  = np.zeros( (len(kernel_list), len(gen_model_list) ) )
    test_hin  = np.zeros( (len(kernel_list), len(gen_model_list) ) )
    
    for l in range(len(kernel_list)): #average data
        for j in range(len(gen_model_list)):
            for r in range(n_runs):
                train_bin[l, j] += train_acc_bindev[l, j, r] / float(n_runs)
                test_bin[l, j]  += test_acc_bindev[l, j, r]  / float(n_runs)
                train_hin[l, j] += train_acc_hinge[l, j, r]  / float(n_runs)
                test_hin[l, j]  += test_acc_hinge[l, j, r]   / float(n_runs)

    train_acc = np.concatenate((train_bin,train_hin),axis=1) # combine accuracies (bindev and hinge)
    test_acc  = np.concatenate((test_bin,test_hin),axis=1)
    return train_acc, test_acc

######### Question 3 #########

'''
Part a
'''
def dualHinge(X, y, lamb, kernel_func):
    n = X.shape[0]
    d = X.shape[1]
    
    K = kernel_func(X,X)

    P = (1/lamb) * (np.diagflat(y) @ K @ np.diagflat(y)) #minimize
    P = P + (1e-8) * np.eye(n)
    
    q = - np.ones((n,1))   #min a.T @ 1n
    
    G = np.concatenate((np.identity(n), -np.identity(n)), axis=0)           #a<1, -a<0
    h = np.concatenate((np.ones((n,1)), np.zeros((n,1))), axis=0)

    A = y.T            #y.T @ a = 0
    b = np.zeros((1,1))

    G = G * 1.0
    h = h * 1.0
    P = P * 1.0
    q = q * 1.0
    A = A * 1.0
    b = b * 1.0

    A = A.reshape(1,A.shape[1])
    
    G = cv.matrix(G)
    h = cv.matrix(h) #convert to cvxopt arrays
    P = cv.matrix(P)
    q = cv.matrix(q)
    A = cv.matrix(A)
    b = cv.matrix(b)

    cv.solvers.options['show_progress'] = False
    sol = cv.solvers.qp(P, q, G, h, A, b) #solve linear programming problem

    a=np.array(sol['x'])[:n]
    bestVal=0
    bestIndex=0
    for x in a:
     if abs(x-0.5)<bestVal:  #find ai closest to 0.5
      bestVal=abs(x-0.5)
      bestIndex=x
    b=y[bestIndex]-((1/lamb)*(K[bestIndex].T@np.diagflat(y)@a)) #calculate intercept
    return a, b
    
'''
Part b
'''
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    inner = kernel_func(Xtest,X) @ np.diagflat(y) @ a
    inner = (1/lamb) * inner + b
    return np.sign(inner)

'''
Part c
'''
def evalNumbers(fileName):
    X=[]
    y=[]
    data = np.loadtxt(fileName, delimiter=",")
    for row in data:
        X.append(row[1:])
        y.append(-1 if row[0]==4 else 1) #y is either 4 or 9, set to -1 and 1 respectively
    X=np.array(X)/255 #normalize values
    y=np.array(y)
    k=10
    s=500/k  #size of each chunk

    maxmax=0 #best score overall
    best=""
    for lamb in [0.001,0.01,0.1,1.0]:
        for kf, kernel_func in enumerate([lambda X1, X2:  polyKernel(X1, X2, 2),
                                          lambda X1, X2:  polyKernel(X1, X2, 3),
                                          lambda X1, X2:  polyKernel(X1, X2, 4),
                                          lambda X1, X2: gaussKernel(X1, X2, 1),
                                          lambda X1, X2: gaussKernel(X1, X2, 0.5),
                                          lambda X1, X2: gaussKernel(X1, X2, 0.1)]):
            results=[] #gets all results over k attempts
            for i in range(k):
                a=int(s*i)
                b=int(s*(i+1))
                c=a+500
                d=b+500 #set values to separate k from non-k in both first and second (4 and 9) halves
                trainX=np.concatenate((X[0:a],X[b:500],X[500:c],X[d:1000]), axis=0).reshape(900,X.shape[1])
                trainY=np.concatenate((y[0:a],y[b:500],y[500:c],y[d:1000]), axis=0).reshape(900,1)
                testX=np.concatenate((X[a:b],X[c:d]), axis=0).reshape(100,X.shape[1])
                testY=np.concatenate((y[a:b],y[c:d]),axis=0).reshape(100,1)
                w,w0=dualHinge(trainX, trainY, lamb, kernel_func) #train model
                accuracy=np.mean(testY == dualClassify(testX, w, w0, trainX, trainY, lamb, kernel_func))
                results.append(accuracy) #get accuracy of all k
            if np.median(results)>maxmax: #get best overall
                maxmax=np.median(results)
                best="lambda="+str(lamb)+"  bestKernel="+str(kf)+"  accuracy="+str(maxmax)
    print(best)
    return best