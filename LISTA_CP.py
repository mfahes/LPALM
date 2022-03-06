import torch
import torch.nn as nn
import numpy as np
from utils.utils import prox_theta

class LISTA_CP(nn.Module):
	'''
	LISTA-CP proposed by Chen et al.
	'''
	def __init__(self, T=16, alpha=5, A=None,theta_shared=False,W_shared= False):
		super(LISTA_CP, self).__init__()
	
		self.T = T
		self.alpha = alpha
		self.A = A 
		self.theta_shared = theta_shared 
		self.W_shared = W_shared
		self.L = torch.tensor((1.001) * np.linalg.norm(self.A[0].numpy(),ord=2)**2)
		self.theta = (self.alpha / self.L).clone().detach()
		if not self.theta_shared:
			self.theta = self.theta.repeat(self.T,1)
		self.theta = nn.Parameter(self.theta,requires_grad= True)
		self.M = self.A.shape[1]
		self.N = self.A.shape[2]
		self.We = torch.div(torch.transpose(self.A[0],0,1), self.L)
		self.W = torch.transpose(self.We,0,1)
		if not self.W_shared:
			self.W = self.W.repeat(self.T,1,1)
		self.W = self.W.type(torch.FloatTensor)
		self.W = nn.Parameter(self.W,requires_grad = True)
	
	def forward(self,X,A,Z_0=None,noise_A=None):
	
		b_size = X.shape[0]
		X = X.type(torch.FloatTensor)
		Z_s = []
		
		if Z_0 is None:
			Z = torch.zeros(b_size,self.N,X.shape[2])
		else:
			Z = Z_0

		if noise_A is not None:	
			N = np.random.randn(A.shape[1],A.shape[2])
			for i in range(b_size):
				N = 10.**(-noise_A/20.)*np.linalg.norm(A[i])/np.linalg.norm(N)*N
				A[i] = A[i] + N
					
		A = A.type(torch.FloatTensor)
					
		for t in range(self.T):
			if self.W_shared:
				Z = Z + torch.bmm(torch.transpose(self.W.repeat(b_size,1,1),1,2),X-torch.bmm(A,Z))
			else:
				Z = Z + torch.bmm(torch.transpose(self.W[t].repeat(b_size,1,1),1,2),X-torch.bmm(A,Z))
			if self.theta_shared:
				Z = prox_theta(Z,self.theta,ch_type=True)
			else:
				Z = prox_theta(Z,self.theta[t],ch_type=True)			
			
			Z_s.append(Z)

		return Z, Z_s
