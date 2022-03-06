import torch
import torch.nn as nn
import numpy as np
from utils.utils import prox_theta

class ISTA_LLT(nn.Module):
	'''
	ISTA with learned L and threshold theta
	'''
	def __init__(self, T=10, alpha=1,  A=None, L_shared=False,theta_shared=False):
		super(ISTA_LLT, self).__init__()

		self.T = T
		self.L_shared = L_shared
		self.theta_shared = theta_shared
		self.A = A
		self.alpha = alpha
		
		self.L = torch.tensor((1.001) * np.linalg.norm(self.A[0].numpy(),ord=2)**2)
		self.theta = self.alpha / self.L
		
		if not self.L_shared:
			self.L = self.L.repeat(self.T,1)
		
		if not self.theta_shared:
			self.theta = self.theta.repeat(self.T,1)

		self.L = nn.Parameter(self.L, requires_grad = True)	
		self.theta = nn.Parameter(self.theta,requires_grad= True)

			
		#if self.L_shared:
		#	self.L = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
		#else:
		#	self.L = nn.Parameter(torch.Tensor([1.]*self.T), requires_grad=True)
			
			
		#if self.theta_shared:
		#	self.theta = nn.Parameter(torch.Tensor([1e-3]),requires_grad= True)
		#else:
		#	self.theta = nn.Parameter(torch.Tensor([1e-3]*self.T), requires_grad =True)
				
	def forward(self, X, A, Z_0=None,noise_A=None):

		b_size = X.shape[0]
		
		X = X.type(torch.FloatTensor)
		Z_s = []
		
		if noise_A is not None:	
			N = np.random.randn(A.shape[1],A.shape[2])
			for i in range(b_size):
				N = 10.**(-noise_A/20.)*np.linalg.norm(A[i])/np.linalg.norm(N)*N
				A[i] = A[i] + N
	
		if Z_0 is None:
			Z = torch.zeros([b_size, A.shape[2], X.shape[2]], dtype=torch.float)
		else:
			Z = Z_0	
		
		for t in range(self.T):
			if not self.L_shared and not self.theta_shared:
				Z = prox_theta(Z - (1/self.L[t])*torch.bmm(torch.transpose(A.float(),1,2),(torch.bmm(A.float(),Z)-X)), self.theta[t],ch_type=True)
				Z_s.append(Z)	

		return Z, Z_s
