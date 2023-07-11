import cvxopt as cp
import numpy as np

######### Question 1 #########

'''
Part a
Takes X(nxd) and y(nx1) returns min w(dx1) for L2 losses
'''
def minimizeL2(X, y):
 #want min of 1/2n ||Xw-y||2    Gradient=0 is (X^T X)^-1 X^T y
 gradient=np.linalg.inv(X.T@X)@((X.T)@y)
 return gradient

'''
Part b
Takes X(nxd) and y(nx1) returns min w(dx1) for L1 losses
'''
def minimizeL1(X, y):
 #want min w for 1/n ||Xw-y||1    1/n doesn't matter, ignore
 #| X -I| |w| < | y|  <-- top half of matrix
 #|-X -I| |s| - |-y|  <-- bottom half of matrix
 n=X.shape[0]
 d=X.shape[1]
 
 topHalf=np.concatenate((X,-1*np.identity(n)), axis=1)       #top half of matrix
 bottomHalf=np.concatenate((-1*X,-1*np.identity(n)), axis=1) #bottom half of matrix
 
 a = np.concatenate((topHalf,bottomHalf), axis=0)     #w and s (weight and delta) are variables    A=2n x n+d
 b = np.concatenate((y,-1*y), axis=0)                 #A<=[y,-y].T    b=1 x 2n
 c = np.concatenate((np.zeros(d),np.ones(n)), axis=0) #array of w=0, second half delta*1, sum=value being minimized=delta*d
 
 a = cp.matrix(a, (a.shape[0], a.shape[1]), 'd')
 b = cp.matrix(b, (b.shape[0],1), 'd') #convert to cvxopt arrays
 c = cp.matrix(c, (c.shape[0],1), 'd')
 a=a*1.0
 b=b*1.0 #convert to floats
 c=c*1.0
 cp.solvers.options['show_progress'] = False
 sol=cp.solvers.lp(c, a, b) #solve linear programming problem
 return np.array(sol['x'][:d])

'''
Part c
Takes X(nxd) and y(nx1) returns min w(dx1) for L-infinity losses
'''
def minimizeLinf(X, y):
 #| X -1| |w| < | y|  <-- top half of matrix
 #|-X -1| |s| - |-y|  <-- bottom half of matrix
 n=X.shape[0]
 d=X.shape[1]
 topHalf=np.concatenate((X,-1*np.ones((n,1))), axis=1)       #top half of matrix
 bottomHalf=np.concatenate((-1*X,-1*np.ones((n,1))), axis=1) #bottom half of matrix
 a=np.concatenate((topHalf,bottomHalf), axis=0) #w and s (weight and delta) are variables    A=2n x d+1
 b=np.concatenate((y,-1*y), axis=0) #A<=[y,-y].T     b=1 x 2n
 c=np.concatenate((np.zeros(d),np.array([1])), axis=0) #array of w=0, last val is delta*1, sum=value being minimized=delta*1
 a = cp.matrix(a, (a.shape[0], a.shape[1]), 'd')  #convert to cvxopt arrays
 b = cp.matrix(b, (b.shape[0],1), 'd')
 c = cp.matrix(c, (c.shape[0],1), 'd')
 a=a*1.0
 b=b*1.0 #convert to floats
 c=c*1.0
 cp.solvers.options['show_progress'] = False
 sol=cp.solvers.lp(c,a,b) #solve lp problem
 return np.array(sol['x'][:d])

'''
Part d
Returns a 3×3 matrix train loss of average training losses and a 3×3 matrix test loss of average test losses
'''
def synRegExperiments():
 train=np.zeros((3,3)) #to hold results
 test=np.zeros((3,3))
 MAX_LOOP=100
 for _ in range (MAX_LOOP):
  n = 30 # number of data points
  d = 5 # dimension
  noise = 0.2
  X = np.random.randn(n, d) # input matrix
  X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment input
  w_true = np.random.randn(d + 1, 1) # true model parameters
  y = X @ w_true + np.random.randn(n, 1) * noise # ground truth label
  weights=[minimizeL2(X, y), minimizeL1(X, y), minimizeLinf(X, y)] #weights for L2, L1, Linf losses
  for weight in range(len(weights)):
   train[weight][0]+= (1/(2*n))*(X@weights[weight]-y).T@(X@weights[weight]-y) *(1/MAX_LOOP)  #L2 loss
   train[weight][1]+= (1/n)*(X@weights[weight]-y).T@(np.ones((n, 1))) * (1/MAX_LOOP)  #L1 loss
   train[weight][2]+= np.amax(np.absolute(((X@weights[weight]-y)))) * (1/MAX_LOOP)  #Linf loss

  #Generating 1000 new test data points
  m = 1000
  testX = np.random.randn(m, d)  #initial random data values
  testX = np.concatenate((np.ones((m, 1)), testX), axis=1) #augment input
  testY = testX @ w_true + np.random.randn(m, 1) * noise  #generate y  from x using w_true
  for weight in range(len(weights)):
   test[weight][0]+= (1/(2*m))*(testX@weights[weight]-testY).T@(testX@weights[weight]-testY) *(1/MAX_LOOP) #L2 loss
   test[weight][1]+= (1/m) * (testX@weights[weight]-testY).T @ (np.ones((m, 1))) * (1/MAX_LOOP) #L1 loss
   test[weight][2]+= np.amax(np.absolute((testX@weights[weight]-testY))) * (1/MAX_LOOP) #Linf loss
 return train,test

######### Question 2 #########

'''
Part a
Takes dx1 w, nxd X and nx1 y. returns scalar value of the objective function in Eq1 (L2 loss) and analytic form gradient of size dx1
'''
def linearRegL2Obj(w, X, y):
 n=X.shape[0]
 L2loss=(1/(2*n))*(X@w-y).T@(X@w-y)  #L2 loss function
 gradient=(1/n)*(X.T@(X@w-y))  #gradient of L2 loss function
 return L2loss, gradient

'''
Part b
Gradient descent, returns weights mapping X to y after it satisfies tol or exceeds max_iter
'''
def gd(obj_func, w_init, X, y, step_size, max_iter, tol):
 w=w_init #initialize w
 for _ in range(max_iter):
  objVal, grad = obj_func(w, X, y) #compute gradient and current loss
  if objVal<tol: #satisfied tolerance, break and return
   break
  w=w-(grad*step_size) #take step down gradient to approach solution
 return w

'''
Part c
Logistic regression, returns CE loss and gradient
'''
def logisticRegObj(w, X, y):
 n=X.shape[0]
 x = X@w
 scalarVal=((-1*y.T)@(-1*np.logaddexp(0.0,-1*x)))-((1-y.T)@(-1*np.logaddexp(0.0,x)))*(1/n)  #cross entropy loss equation using sigmoid
 positive = x >= 0 #positive values within x
 negative = x < 0 #negative values within x
 sigmoid = np.empty_like(x, dtype=np.float) #clone x and make sure to force float type just in case of bad input types
 sigmoid[positive] = 1 / (1 + np.exp(-1 * x[positive])) #calculate sigmoid result with negative-overflow equation on positive values
 exp = np.exp(x[negative]) #store result of exp to minimize duplication
 sigmoid[negative] = exp / (1 + exp) #calculate sigmoid result with positive-overflow equation on negative values
 grad= X.T @ ( sigmoid - y) #generate gradient
 return scalarVal.item(), grad

'''
Part  d
Returns a 4 × 3 matrix train acc of average training accuracies and a 4 × 3 matrix test acc of average test accuracies
'''
def synClsExperiments():
 train_acc=np.zeros((4,3)) #return values
 test_acc=np.zeros((4,3))
  #hardcoded n,d,eta for the 12 different slots in the return values
 stats=[[10,2,0.1],[50,2,0.1],[100,2,0.1],[200,2,0.1],[100,1,0.1],[100,2,0.1],[100,4,0.1],[100,8,0.1],[100,2,0.1],[100,2,1.0],[100,2,10.0],[100,2,100.0]]
 for _ in range (100):
  for stat in range(12):
   # data generation
   n=stats[stat][0] # number of data points
   d=stats[stat][1] #dimension
   c0 = np.ones([1, d]) # class 0 center
   c1 = -np.ones([1, d]) # class 1 center
   X0 = np.random.randn(n, d) + c0 # class 0 input
   X1 = np.random.randn(n, d) + c1 # class 1 input
   X = np.concatenate((X0, X1), axis=0)
   X = np.concatenate((np.ones((2 * n, 1)), X), axis=1) # augmentation
   y = np.concatenate([np.zeros([n, 1]), np.ones([n, 1])], axis=0)

   # test data generation
   m=1000
   testX0 = np.random.randn(m, d) + c0 # class 0 input
   testX1 = np.random.randn(m, d) + c1 # class 1 input
   testX = np.concatenate((testX0, testX1), axis=0)
   testX = np.concatenate((np.ones((2 * m, 1)), testX), axis=1) # augmentation
   testy = np.concatenate([np.zeros([m, 1]), np.ones([m, 1])], axis=0)

   # learning
   eta=stats[stat][2] #step size
   max_iter = 1000 #maximum umber of iterations
   tol = 1 #tolerance
   w_init = np.random.randn(d + 1, 1) #temporary array to initialize w
   w_logit = gd(logisticRegObj, w_init, X, y, eta, max_iter, tol) #train the weights on the given function

   scalar,grad=logisticRegObj(w_logit,X,y)  #evaluate weights on training data
   train_acc[stat%4][int(stat/4)]+= scalar/100.0
   
   scalar,grad=logisticRegObj(w_logit,testX,testy) #evaluate weights on test data
   test_acc[stat%4][int(stat/4)] += scalar/100.0
 
 return train_acc,test_acc

######### Question 3 #########

'''
Part a
Loads data from file, returns X nxd and y nx1, custom built for one of 3 given data files
'''
def loadData(folder,name):
 X=[]
 y=[]
 fname = open(folder+name, "r")
 if name=="auto-mpg.data":
  for line in fname.readlines():
   data=line.strip().split()
   if data[3]!='?': #ignore incomplete lines
    #convert to ints
    X.append(np.array(data[1:7]).astype(np.float)) #columns 2 to 7 are data
    y.append(float(data[0])) #first column is label
 elif name=="parkinsons.data":
  fname.readline()  #omits first line
  for line in fname.readlines():
   data=line.strip().split(",")
   X.append(np.array(data[1:16]+data[18:]).astype(np.float)) #omits column 17, as that is y
   y.append([float(data[17])]) #16th column is label
 elif name=="sonar.all-data":
  for line in fname.readlines():
   data=line.strip().split(",")
   X.append(np.array(data[:len(data)-1]).astype(np.float)) #all but last column are data
   y.append([1 if data[len(data)-1]=='M' else 0]) #Last column is M or R, M=1 and R=0
 X=np.array(X)
 X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # augment input
 y=np.array(y)
 return X,y

'''
Part b
Performs logistic regression and classification on given data set to evaluate effectivess of code
'''
def realExperiments(folder, name):
 X,y=loadData(folder,name)  #read in data from file
 for _ in range(100):
  n = int(X.shape[0] * .5) #This rounds down
  randList  = list(range(X.shape[0]))
  train = np.random.choice(randList, n, replace=False)  # randomly select half the row indices
  test = list(set(randList) - set(train)) #split data evenly into train and test
  
  trainX=X[train]
  testX=X[test]
  trainY=y[train]
  testY=y[test]

  if name=="auto-mpg.data": #linear regression
  
   train = np.zeros((3,3)) #return values
   test = np.zeros((3,3))
   
   weights=[minimizeL2(trainX, trainY), minimizeL1(trainX, trainY), minimizeLinf(trainX, trainY)] #weights for L2, L1, Linf losses
   for weight in range(len(weights)):
    temp = (trainX@weights[weight] - trainY) #temporary holder
    
    weights[weight] = weights[weight].reshape(weights[weight].shape[0],1) #handle (n,1) vs (n,)
    trainY = trainY.reshape(trainY.shape[0],1)
    
    loss, gradient = linearRegL2Obj(weights[weight], trainX, trainY) #evaluate weights for L2 loss
    train[weight][0]+= loss / 100.0
    train[weight][1]+= ((1 / n) * (trainX @ weights[weight] - trainY).T @ (np.ones((n, 1)))) / 100.0 #L1 loss function
    train[weight][2]+= np.amax(np.absolute((trainX @ weights[weight] - trainY))) / 100.0 #Linf loss
    
    testY = testY.reshape(testY.shape[0],1) #handle (n,1) vs (n,)
    
    loss, gradient = linearRegL2Obj(weights[weight], trainX, trainY) #evaluate weights for L2 loss
    test[weight][0] += loss / 100.0
    test[weight][1] += ((1 / n) * (testX@weights[weight] - testY).T @ (np.ones((n, 1)))) / 100.0 #L1 loss
    test[weight][2] += np.amax(np.absolute((testX @ weights[weight] - testY))) / 100.0 #Linf loss
  
  else: #linear classification
  
   train = np.zeros((4,1)) #return values
   test = np.zeros((4,1))
   
   for eta in range(4): #evaluating classification for different step sizes
   
    w_init = np.random.randn(trainX.shape[1], 1) #initialize w
    
    w = gd(logisticRegObj, w_init, trainX, trainY, 1/(10**(4-eta)), 1000, 1e-10) #perform gradient descent
    tr,grad = logisticRegObj(w,trainX,trainY) #evaluate weights on training data
    train[eta] += tr/100
    
    tr,grad = logisticRegObj(w,testX,testY) #evaluate weights on test data
    test[eta] += tr/100
   
 return train,test