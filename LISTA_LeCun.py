import torch
import torch.nn as nn
import numpy as np
from utils.utils import prox_theta

class LISTA_LeCun(nn.Module):
	def __init__(self, T=16, alpha=5, A=None,theta_shared=True ,We_shared = True ,G_shared = True):
		super(LISTA_LeCun, self).__init__()
		'''
		LISTA_LeCun: original parametrization of ISTA proposed by Gregor et al.
		'''
		self.T = T 
		self.alpha = alpha
		self.A = A
		self.theta_shared = theta_shared
		self.We_shared = We_shared
		self.G_shared = G_shared
		self.L = torch.tensor((1.001) * np.linalg.norm(self.A[0].numpy(),ord=2)**2)
		self.theta = (self.alpha / self.L).clone().detach()
		if not self.theta_shared:
			self.theta = self.theta.repeat(self.T,1)
		self.theta = nn.Parameter(self.theta,requires_grad= True)
		self.M = self.A.shape[1]
		self.N = self.A.shape[2]
		self.We = torch.div(torch.transpose(self.A[0],0,1), self.L)
		if not self.We_shared:
			self.We = self.We.repeat(self.T,1,1)
		self.We = nn.Parameter(self.We, requires_grad=True)
		self.G = torch.eye(self.A.shape[2]) - torch.matmul(torch.div(torch.transpose(self.A[0],0,1), self.L), self.A[0])
		if not self.G_shared:
			self.G = self.G.repeat(self.T,1,1)
		self.G = self.G.type(torch.FloatTensor)
		self.G = nn.Parameter(self.G, requires_grad=True)
	
	def forward(self, X, Z_0=None):
	
		b_size = X.shape[0]
		Z_s = []
		
		if Z_0 is None:
			Z = torch.zeros(b_size,self.N,X.shape[2])
		else:
			Z = Z_0
		
		for t in range(self.T):
			if self.We_shared:
				p1 = torch.bmm(self.We.repeat(b_size,1,1),X)
			else:
				p1 = torch.bmm(self.We[t].repeat(b_size,1,1),X)
			if self.G_shared:
				p2 = torch.bmm(self.G.repeat(b_size,1,1),Z)
			else:
				p2 = torch.bmm(self.G[t].repeat(b_size,1,1),Z)
			p3 = p1+p2
			if self.theta_shared:
				Z = prox_theta(p3,self.theta,ch_type=True)
			else:
				Z = prox_theta(p3,self.theta[t],ch_type=True)
			Z_s.append(Z)

		return Z, Z_s
