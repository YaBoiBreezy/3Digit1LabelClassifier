# COMP 3105 A3
# Alexander Breeze 101 143 291
# Viktor Litzanov  101 143 028

import numpy as np
import scipy as sp
from scipy import optimize, special, spatial
from A3helpers import *

######### Question 1 #########

'''
Part a
X=nxd, Y=nxk, returns dxk
'''
def minMulDev(X, Y):
 n,d = X.shape
 k = Y.shape[1]

 def obj_func(wdk):
  W = np.reshape(wdk, (d,k)) #convert input vector into weight matrix

  ret2 = 0
  for x in range(n):
   ret2 += sp.special.logsumexp(W.T @ X[x]) #get sum of logs
  return ret2 - np.sum(Y @ W.T * X) #return loss
 
 initial_w = np.zeros(d*k) # generate w
 sol = sp.optimize.minimize(obj_func, initial_w) # minimize
 w = np.reshape(sol['x'], (d,k)) #convert output vector into weight matrix
 return w

'''
Part b
Xtest=mxd, W=dxk, return mxk
'''
def classify(Xtest, W):
 tmp = Xtest @ W
 second = np.zeros_like(tmp) #makes all-0 matrix same size an tmp
 second[np.arange(len(tmp)), tmp.argmax(1)] = 1  #sets max val in each row in tmp to 1 in second, others stay 0
 return second

'''
Part c
Yhat=mxk, Y=mxk, return=scalar
'''
def calculateAcc(Yhat, Y):
 return np.sum(Yhat * Y) / Y.shape[0] #When Yhat[x]==Y[x], [1,0]*[1,0]=[1,0]  sum=1   Otherwise [1,0]*[0,1]=[0,0]  sum=0

######### Question 2 #########

'''
Part a
X=nxd, k=scalar, returns kxd
'''
def PCA(X, k):
 n,d = X.shape
 mean = np.mean(X, axis = 0) # find mean row of all rows
 centerX = X - mean #subtract mean from each row to center X at origin
 w,v = sp.linalg.eigh(centerX.T @ centerX) #get all the eigenvectors
 return (v.T)[-k:,:] #transpose matrix to get eigenvectors as rows, take last (highest) k

'''
Part b
'''
def shirtLoader(fileName, k=20):
 X = []
 data = np.loadtxt(fileName, delimiter=",") #read in csv file
 for row in data:
  X.append(row)
 X = np.array(X)

 #plotImgs(X)
 U = PCA(X,k) #get eigens U
 plotImgs(U)

'''
Part c
xTest=mxd, mu=dx1, U=kxd, returns mxk
'''
def projPCA(Xtest, mu, U):
 return (Xtest - mu.T) @ U.T #subtract mu to center Xtest, project over directions U

'''
Part d
'''
def synClsExperimentsPCA():
 n_runs = 100
 n_train = 128 #number of data points
 n_test = 1000 #number of data points
 dim_list = [1, 2]
 gen_model_list = [1, 2]
 train_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs]) #create results matrices
 test_acc  = np.zeros([len(dim_list), len(gen_model_list), n_runs])
 
 for r in range(n_runs):
  for i, k in enumerate(dim_list):
   for j, gen_model in enumerate(gen_model_list):
    Xtrain, Ytrain = generateData(n = n_train, gen_model = gen_model) #generate data points
    Xtest,   Ytest = generateData(n = n_test,  gen_model = gen_model) #generate data points
    Xtrain = unAugmentX(Xtrain) # remove augmentation before PCA
    Xtest  = unAugmentX(Xtest)
    
    U = PCA(Xtrain, k)
    
    Xtrain_proj = projPCA(Xtrain, np.mean(Xtrain, axis=0), U) #project points over largest directions
    Xtest_proj  = projPCA(Xtest,  np.mean(Xtrain, axis=0), U)
    Xtrain_proj = augmentX(Xtrain_proj) # add augmentation back
    Xtest_proj  = augmentX(Xtest_proj)
    
    W = minMulDev(Xtrain_proj, Ytrain) # from Q1
    
    Yhat = classify(Xtrain_proj, W) # from Q1
    train_acc[i, j, r] = calculateAcc(Yhat, Ytrain) # from Q1
    Yhat = classify(Xtest_proj, W)
    test_acc[i, j, r] = calculateAcc(Yhat, Ytest) #from Q1
 train = np.zeros([len(dim_list), len(gen_model_list)])
 test  = np.zeros([len(dim_list), len(gen_model_list)])
 
 for i, k in enumerate(dim_list):
  for j, gen_model in enumerate(gen_model_list): #get averages over runs
   train[i, j] = np.mean(train_acc[i, j])
   test[i, j]  = np.mean(test_acc[i, j])
 return train,test

######### Question 3 #########

'''
Part a
X=nxd, k=scalar, returns nxk, kxd, scalar
'''
def kmeans(X, k, max_iter=1000):
 n, d = X.shape
 U = np.random.rand(k, d) #initialize U
 for i in range(max_iter):
  D = sp.spatial.distance.cdist(X, U, 'sqeuclidean') #get squared euclidean distance between X and U
  Y = np.zeros_like(D)
  Y[np.arange(len(D)), D.argmax(1)] = 1 #Get matrix of 0's with 1 in each row at D max index
  old_U = U
  U = np.linalg.inv(Y.T @ Y + 1e-8 * np.eye(k)) @ Y.T @ X #augment Y for Y.T@Y
  if np.allclose(old_U, U): #break early if converged
   break
 obj_val = ( 1/ (2 * n) ) * np.sum((X - Y @ U).T @ (X - Y @ U)) #compute objective value using original equation
 return Y, U, obj_val

'''
Part b
X=nxd, k=scalar, returns nxk, kxd, scalar
'''
def repeatKmeans(X, k, n_runs=100):
 best = np.inf #want lowest value
 for _ in range(n_runs):
  Y, U, obj_val = kmeans(X, k) #call this multiple times
  if obj_val < best: #get call with lowest obj_val
   best = obj_val
   bestRes = (Y, U, obj_val)
 return bestRes[0], bestRes[1], bestRes[2] #return call with lowest obj_val

'''
Part c
X=nxd, returns kx1 list
'''
def chooseK(X, k_candidates=[2,3,4,5,6,7,8,9]):
 results = [[],[]] #generate result for each k
 for k in k_candidates:
  results[0].append(k)
  Y, U, obj_val = repeatKmeans(X, k)
  results[1].append(obj_val) #want obj_val of best run
 return results
