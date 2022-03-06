import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from utils.utils import prox

def ISTA(set_,lmda,nb_iter,nb_iter_fixed = False,noise_A=None):
	'''
	Function to estimate S from X and A using ISTA on a whole set
	
	Input: set of X and A
	Output: S_est, the estimation of S
	lmda: regularization parameter
	nb_iter: number of iterations specified by the user
	nb_iter_fixed: If True, stop the algorithm when nb_iter is reached. If False, wait until ISTA convergence
	noise_A : SNR 
	'''
	eps = 0.01
	criterion = nn.MSELoss()
	S_est = torch.zeros([len(set_), next(iter(set_))[2].shape[0],next(iter(set_))[2].shape[1]], dtype=torch.double) #initialization
	S_est_prev = deepcopy(S_est)
	loss = []
	loss_per_layer = torch.zeros((nb_iter))
	it=0
	total_it = 0
	for i, (X,A,S) in enumerate(set_):
		X = torch.from_numpy(X)
		A = torch.from_numpy(A)
		if noise_A is not None: #add noise to A inside the iterations of ISTA	
			N = np.random.randn(A.shape[0],A.shape[1])
			N = 10.**(-noise_A/20.)*np.linalg.norm(A)/np.linalg.norm(N)*N
			A = A + N
		S = torch.from_numpy(S)
		L = torch.tensor((1+eps) * np.linalg.norm(A.numpy(),ord=2)**2)
		print("example %d"%i)
		it= 0
		S_est_list = []
		while(torch.norm(S_est[i]-S_est_prev[i], p = 'fro') > 1e-6 or it < 2):
			if it>0:
				S_est_prev[i] = deepcopy(S_est[i])
			S_est[i] = prox(S_est[i] - (1 / L) * torch.matmul(torch.transpose(A, 0, 1), (torch.matmul(A, S_est[i]) - X)), lmda, L) #ISTA update
			S_est_list.append(torch.empty_like(S_est[i]).copy_(S_est[i]))
			it += 1
			total_it +=1
			if nb_iter_fixed and it==nb_iter:
				break
		loss.append(torch.numel(S) *criterion(S_est[i],S) / torch.norm(S)**2)
		for j in range(nb_iter):
			loss_per_layer[j] += torch.numel(S) *criterion(S_est_list[j],S) / torch.norm(S)**2

	for j in range(nb_iter):
		loss_per_layer[j] = loss_per_layer[j]/(i+1)

	print('the average number of iterations is: ',total_it/(i+1))
	print('The average loss of ISTA is {:.5f}'.format(sum(loss)/(i+1)))
	print(loss_per_layer)
	return S_est , sum(loss)/(i+1), loss_per_layer
