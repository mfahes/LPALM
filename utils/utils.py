import torch
import numpy as np
from utils.munkres import Munkres
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def prox(x,lmda,L):
	res = torch.sign(x) * torch.maximum(torch.abs(x) - lmda/L, torch.zeros_like(x))
	return res

def prox_l1(S_est,thrd): #numpy
  S_est[(abs(S_est) < thrd)] = 0
  indNZ = np.where(abs(S_est) > thrd)[0]
  S_est[indNZ] = S_est[indNZ] - thrd*np.sign(S_est[indNZ])
  return S_est

def prox_theta(x,theta,ch_type=False): #torch
	theta = torch.maximum(theta,torch.zeros_like(theta))
	res = torch.sign(x) * torch.maximum(torch.abs(x) - theta , torch.zeros_like(x))
	if ch_type:
		res = res.type(torch.FloatTensor)
	return res

def norm_col(A):
    An = A.copy()
    type(An)
    for ii in range(np.shape(An)[1]):
        An[:,ii] = An[:,ii]/np.sqrt(np.sum(An[:,ii]**2));
    
    return An

def prox_oblique(A):
    for ii in range(np.shape(A)[1]):
        normeA = np.sqrt(np.sum(A[:,ii]**2))
        if normeA > 0 and normeA > 1.:
            A[:,ii] /= normeA
        
    return A

def correctPerm(W0_en,W_en):
    # [WPerm,Jperm,err] = correctPerm(W0,W)
    # Correct the permutation so that W becomes the closest to W0.
    
    #W0_en = W0
    #W_en = W
    W0 = W0_en.copy()
    W = W_en.copy()
    
    W0 = norm_col(W0)
    W = norm_col(W)
    print(W0)
    costmat = -W0.T@W; # Avec Munkres, il faut bien un -
    print(costmat)
    
    m = Munkres()
    Jperm = m.compute(costmat.tolist())
    #print(Jperm)
    
    WPerm = np.zeros(np.shape(W0))
    indPerm = np.zeros(np.shape(W0_en)[1])
    
    for ii in range(W0_en.shape[1]):
        WPerm[:,ii] = W_en[:,Jperm[ii][1]]
        indPerm[ii] = Jperm[ii][1]
        
    return WPerm,indPerm.astype(int)


def NMSE(W0_en,W_en,S0_en,S_en):
    # W0 : true mixing matrix
    # W : estimated mixing matrix
    #
    # maxAngle : cosine of the maximum angle between the columns of W0 and W
    
    W0 = W0_en.copy()
    W = W_en.copy()
    
    S0 = S0_en.copy()
    S = S_en.copy()
    
    W,indPerm = correctPerm(W0,W);
    
    S = S[indPerm,:]
    
    nmse = np.sum((S-S0)**2)/(np.sum(S0**2))
    
    return nmse

def evalCriterion(W0_en,W_en):
    # W0 : true mixing matrix
    # W : estimated mixing matrix
    #
    # maxAngle : cosine of the maximum angle between the columns of W0 and W
    
    W0 = W0_en.copy()
    W = W_en.copy()
    
    W,indPerm = correctPerm(W0,W);
    
    W0 = norm_col(W0_en)
    W = norm_col(W)

    diff = W0.T@W;
    
    return np.mean(np.diag(diff));


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)
    
	
def train_val_dataset(dataset, val_split= 1/6):
	train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split,random_state=10, shuffle = True)
	datasets = {}
	datasets['train'] = Subset(dataset, train_idx)
	datasets['val'] = Subset(dataset, val_idx)
	return datasets


def split_A(dataset):

	A0 = next(iter(dataset))[1]
	l = []
	for i,(X,A,S) in enumerate(dataset):
		crit = np.sum(np.multiply(A0[:,0],A[:,0])) / (np.linalg.norm(A0[:,0]) * np.linalg.norm(A[:,0])) + np.sum(np.multiply(A0[:,1],A[:,1])) / (np.linalg.norm(A0[:,1]) * np.linalg.norm(A[:,1])) + np.sum(np.multiply(A0[:,2],A[:,2])) / (np.linalg.norm(A0[:,2]) * np.linalg.norm(A[:,2])) + np.sum(np.multiply(A0[:,3],A[:,3])) / (np.linalg.norm(A0[:,3]) * np.linalg.norm(A[:,3]))
		l.append(crit)
	ind = sorted(range(len(l)), key=lambda k: l[k])
	val_idx = ind[:150]
	train_idx = ind[150:]
	datasets = {}
	datasets['train'] = Subset(dataset, train_idx)
	datasets['val'] = Subset(dataset, val_idx)
	
	return datasets

def plot_train_test_loss(train_loss,test_loss):
	plt.plot(train_loss)
	plt.plot(test_loss)
	plt.xlabel('epochs')
	plt.ylabel('NMSE')
	plt.legend(["training loss", "validation loss"])
	plt.yscale('log')
	plt.show()
