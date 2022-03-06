import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

#%%
def prox_oblique(A_pred):
    '''
    Apply the projection on the l2-unit ball of A's columns
    '''
    for j in range(A_pred.shape[1]):
        if(torch.any(torch.norm(A_pred[:,j])>torch.tensor([1.]))):
            A_pred[:,j] /= torch.norm(A_pred[:,j].clone())
    return A_pred
   
def ISTA_A(X,S):
	'''
	Apply ISTA on A
	'''
	eps = 0.01
	A_est_list = []
	it=0
	A_est = torch.zeros((X.shape[0],S.shape[0]), dtype=torch.double)
	A_est_prev = A_est.clone()
	L_A = torch.tensor((1+eps) * np.linalg.norm(S.numpy(),ord=2)**2) 
	while(torch.norm(A_est-A_est_prev, p = 'fro') > 1e-6 or it < 2):
		if it>0:
			A_est_prev = A_est.clone()
		A_est = prox_oblique(A_est - (1. / L_A) * torch.matmul((torch.matmul(A_est, S) - X),torch.t(S)))
		it += 1
	return A_est,it		
