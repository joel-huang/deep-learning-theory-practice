import numpy as np, os, sys

import matplotlib as mpl #patch-wise similarities, droi images

def datagen2d(w1,w2,eps,num):

  x=np.random.normal(size=(num,2))
  y=x[:,0]*w1+x[:,1]*w2+eps*np.random.normal(size=(num))

  #x.shape=(numdata,dims) dims=2 here
  #y.shape=(numdata)
  print(x.shape,y.shape)

  return x,y

def rndsplit(x,y,numtr):

  inds=np.arange(y.size)
  np.random.shuffle(inds)

  xtr=x[inds[0:numtr],:]
  ytr=y[inds[0:numtr]]

  xv=x[inds[numtr:],:]
  yv=y[inds[numtr:]]

  return xtr,ytr,xv,yv

def gendata(numtotal):
  
  w1=0.5
  w2=-2
  xtr,ytr = datagen2d(w1=w1,w2=w2,eps=0.8,num= int( numtotal) ) #numtotal
  xv,yv = datagen2d(w1=w1,w2=w2,eps=0.8,num= 3000 ) #fix 3k test samples
  #xtr,ytr,xv,yv = rndsplit(x,y,numtr=int( 0.7*numtotal))
  return xtr,ytr,xv,yv,w1,w2

def gendata2(numtotal):
  
  w1=0.5
  w2=-2
  xtr,ytr = datagen2d(w1=w1,w2=w2,eps=0.8,num= int( 0.7*numtotal) ) #70% of numtotal
  xv,yv = datagen2d(w1=w1,w2=w2,eps=0.8,num= int( 0.3*numtotal) ) #30 % of numtotal
  #xtr,ytr,xv,yv = rndsplit(x,y,numtr=int( 0.7*numtotal))
  return xtr,ytr,xv,yv,w1,w2

def linreg_train(xtr,ytr,C):

  mat=np.linalg.inv( np.dot(xtr.T,xtr)+ C*np.eye(xtr.shape[1]) )
  #print(mat.shape)
  w= np.dot(mat ,np.dot(xtr.T,ytr))
  return w




def linreg_apply(xv,w):
  return np.dot(xv,w)

def mse(ypred,ytrue):
  e=np.mean( (ypred-ytrue)**2 )
  return e

#one setup: 100 train, 5000 train. always 3000 test
# other setup: 100 total, 5000 total, always 30% test, observe means variances of w and of mse averaged over 100 runs

# 30% as test
#variance of estimated w as function of sample size, barplots
#variance of mse as function of sample size, barplots

def lossonvalset(w,xv,yv):
  ypred=linreg_apply(xv,w)
  e=mse(ypred,yv) 
  return e

def gradientononesample(w,x,y,C):
  # TODO   dLoss / dw (your current w), hint: mind the sign please
  
  loss = lossonvalset(w,x,y)
  grad = x.T.dot(loss)

  return grad

def gradientonminibatch(w,xb,yb,C):
  # xb.shape = (num samples, dimensions)
  # yb.shape = (num samples)
  g=np.zeros(2)
  for i in range(xb.shape[0]):
    g+=1.0/float(xb.shape[0])*gradientononesample(w,xb[i,:],yb[i],C)
  return g    


def sgdforthis(xtr, ytr, C, learningrate, batchsize, maxnumepochs, stopthresh, xv,yv):

  #the code which actually runs SGD

  # randomly initialize w, close to zero but not equal to zero
  w=np.random.normal(size=2)

  oldloss=lossonvalset(w,xv,yv) # initial validation loss
  conv=False #flag whether sgd has converged
  for ep in range(maxnumepochs):


    inds=np.arange(xtr.shape[0])

    # the order of samples should be randomized in every epoch           
    # TODO: get a random order of indices that are used to sample minibatches 
    inds = np.random.shuffle(inds)


    numbatches= int( xtr.shape[0] // batchsize )
    for b in range(numbatches):
      #print(w,xtr.shape,ytr.shape)

      # TODO: compute gradient over minibatch    
      # TODO:         g=gradientonminibatch(w,"minibatch of xtr", "same minibatch of ytr as of xtr",C)

      #TODO apply gradient here

  
    #get your validation loss with your newly updated parameters
    newloss=lossonvalset(w,xv,yv)
    # TODO: stop if loss change between old and current epoch is too small, update value of oldloss


  if False==conv:
    print('maxnumepochs reached without convergence')  
  
  return w


def linreg_train_sgd(xtr,ytr,C, learningrate, batchsize, maxnumepochs, stopthresh, xv,yv):



  # what def linreg_train_sgd(xtr,ytr,C, learningrate, batchsize, maxnumepochs, stopthresh, xv,yv):
  # actually does
  w=sgdforthis(xtr, ytr, C, learningrate, batchsize, maxnumepochs, stopthresh, xv,yv)
  return w





def run1(xtr,ytr,xv,yv,w1,w2,C):

  w=linreg_train(xtr,ytr,C=C) # 0.1

  wtrue=np.asarray([w1,w2])

  print('w',w, 'true w', [w1,w2], 'diff', np.dot((w-wtrue).T,w-wtrue))

  ypred=linreg_apply(xv,w)
  e=mse(ypred,yv)

  print('mse',e)

  return e, np.dot((w-wtrue).T,w-wtrue)


def run2(xtr,ytr,xv,yv,w1,w2,C):

  learningrate=0.01
  batchsize=5
  maxnumepochs=1000 
  stopthresh=0.0001

  w=linreg_train_sgd(xtr,ytr,C, learningrate, batchsize, maxnumepochs, stopthresh, xv,yv)

  wtrue=np.asarray([w1,w2])

  print('w',w, 'true w', [w1,w2], 'diff', np.dot((w-wtrue).T,w-wtrue))

  ypred=linreg_apply(xv,w)
  e=mse(ypred,yv)

  print('mse',e)

  return e, np.dot((w-wtrue).T,w-wtrue)

if __name__=='__main__':

  #xtr,ytr,xv,yv,w1,w2=gendata(50)
  #run1(xtr,ytr,xv,yv,w1,w2,1e-3)

  xtr,ytr,xv,yv,w1,w2=gendata(1000)  
  run1(xtr,ytr,xv,yv,w1,w2,1e-3)
  run2(xtr,ytr,xv,yv,w1,w2,1e-3)

