import os,sys,numpy as np
import torch
import time

def forloopdists(feats,protos):
  n=5000
  p=500
  d=300
  dist = np.zeros(n)
  for i in range(n):
    for j in range(p):
      dist[i] += np.sum((feats[i,:]-protos[j,:]) ** 2)
    dist[i] = np.sqrt(dist[i])
  return dist

def numpydists(feats,protos):
  return (feats - protos)**2
  
def pytorchdists(feats0,protos0,device):
  feats = torch.tensor(feats0) #   5000*300
  protos = torch.tensor(protos0) # 500*300

  # need to broadcast each row of feats to subtract all rows of proto
  difference = torch.add(feats, -protos)



  
  


def run():

  ########
  ##
  ## if you have less than 8 gbyte, then reduce from 250k
  ##
  ###############
  feats=np.random.normal(size=(5000,300)) #5000 instead of 250k for forloopdists
  protos=np.random.normal(size=(500,300))


#  '''
  since = time.time()
  #dists0=forloopdists(feats,protos)
  #print(dists0)
  #print(dists0.shape)
  time_elapsed=float(time.time()) - float(since)
  print('Comp complete in {:.3f}s'.format( time_elapsed ))
 # '''



  device=torch.device('cpu')
  since = time.time()

  dists1=pytorchdists(feats,protos,device)


  time_elapsed=float(time.time()) - float(since)

  print('Comp complete in {:.3f}s'.format( time_elapsed ))
  print(dists1.shape)

  #print('df0',np.max(np.abs(dists1-dists0)))


  since = time.time()

  dists2=numpydists(feats,protos)


  time_elapsed=float(time.time()) - float(since)

  print('Comp complete in {:.3f}s'.format( time_elapsed ))

  print(dists2.shape)

  print('df',np.max(np.abs(dists1-dists2)))


if __name__=='__main__':
  run()
