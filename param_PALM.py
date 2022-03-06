import torch
import torch.nn as nn
import numpy as np
from utils.utils import prox_theta,tile,prox

class param_PALM(nn.Module):
	def __init__(self, T=10,S= None,A=None,learn_L_S = False ,L_S_shared=True, LISTA_S = False, W_X_S_shared= True, W_S_S_shared= True,theta_shared= True,  LISTA_CP_S = False, W_CP_S_shared= True, ISTA_LLT_S = True, learn_L_A = False , L_A_shared= True, LISTA_A = False , W_A_A_shared = True, W_X_A_shared = True, LISTA_CP_A = False, W_CP_A_shared = True, non_update_A = True):
		super(param_PALM, self).__init__()
		
		#learn_L_S : to learn L_S (non LISTA-CP)
        	#L_S_shared : when learn_L_S = True, specify if the L_S are shared among the layers
        	#
		#LISTA_S : to use LISTA-like updates for S. If true, theta, W_X_S and W_S_S are learnt such that S = soft(W_X_S X + W_S_S S, theta). REMARK : This makes no sense to do inside PALM.
		#W_X_S_shared, W_S_S_shared: when using LISTA_S, specify if W_X_S, W_X_X are shared among the layers
		#theta_shared: when using LISTA_S,LISTA_CP_S or ISTA_LLT_S, specify if W_X_S, W_X_X are shared among the layers
		#
		#LISTA_CP_S : to use LISTA-CP-like updates for S. If true, theta, W_X_S and W_S_S are learnt such that S = soft(S + W_CP_S.T(X - AS),theta)
		#W_CP_S_shared : when using LISTA_CP_S, specify if W_CP is shared
			#
		#ISTA_LLT_S: to learn L and theta (= lambda/L) for S
		#
		#learn_L_A : to learn L_A
		#L_A_shared : when learn_L_A = True, specify if L_A is shared among the layers
		#
		#LISTA_A, W_A_A_shared, W_X_A_shared, LISTA_CP_A, W_CP_A_shared: Same as S, but for A
		#non_update_A: epxlicit update for A
		
		torch.set_default_tensor_type(torch.DoubleTensor)
		self.T = T
		self.S = S
		self.A = A
		#######################################################     S
		self.learn_L_S = learn_L_S
		self.L_S_shared = L_S_shared
		
		self.LISTA_S = LISTA_S
		self.W_X_S_shared = W_X_S_shared
		self.W_S_S_shared = W_S_S_shared
		self.theta_shared = theta_shared
		
		self.LISTA_CP_S = LISTA_CP_S
		self.W_CP_S_shared = W_CP_S_shared
		
		self.ISTA_LLT_S = ISTA_LLT_S
		#######################################################     A
		self.learn_L_A = learn_L_A
		self.L_A_shared = L_A_shared
		
		self.LISTA_A = LISTA_A
		self.W_A_A_shared = W_A_A_shared
		self.W_X_A_shared = W_X_A_shared
		
		self.LISTA_CP_A = LISTA_CP_A
		self.W_CP_A_shared = W_CP_A_shared
		
		self.non_update_A = non_update_A
		
		##########
		self.alpha =0.00001
		self.L_S = torch.tensor((1.001) * np.linalg.norm(self.A[0].numpy(),ord=2)**2)
		self.L_A = torch.tensor((1.001) * np.linalg.norm(self.S[0].numpy(),ord=2)**2)
		
		#############################################################################
		
		if self.learn_L_S:
			if not self.L_S_shared:
				self.L_S = self.L_S.repeat(self.T,1)
			self.L_S =nn.Parameter(self.L_S,requires_grad= True)
		
		if self.LISTA_S:
			self.theta = (self.alpha / self.L_S).clone().detach()
			self.W_X_S = torch.div(torch.transpose(self.A[0],0,1), self.L_S)
			self.W_S_S = torch.eye(self.A.shape[2]) - torch.matmul(torch.div(torch.transpose(self.A[0],0,1), self.L_S), self.A[0])
			
			if not self.theta_shared:
				self.theta = self.theta.repeat(self.T,1)
			self.theta = nn.Parameter(self.theta, requires_grad= True)
			
			if not self.W_X_S_shared:
				self.W_X_S = self.W_X_S.repeat(self.T,1,1)
			self.W_X_S = nn.Parameter(self.W_X_S, requires_grad= True)
			
			if not self.W_S_S_shared:
				self.W_S_S = self.W_S_S.repeat(self.T,1,1)
			self.W_S_S = nn.Parameter(self.W_S_S, requires_grad= True)
		
		if self.LISTA_CP_S:
			self.theta = (self.alpha / self.L_S).clone().detach()			
			self.We = torch.div(torch.transpose(self.A[0],0,1), self.L_S)
			self.W_CP_S = torch.transpose(self.We,0,1)
			
			if not self.theta_shared:
				self.theta = self.theta.repeat(self.T,1)
			self.theta = nn.Parameter(self.theta, requires_grad= True)
			
			if not self.W_CP_S_shared:
				self.W_CP_S = self.W_CP_S.repeat(self.T,1,1)
			self.W_CP_S = nn.Parameter(self.W_CP_S, requires_grad= True)
			
		if self.ISTA_LLT_S:
			self.theta = (self.alpha/ self.L_S).clone().detach()
			self.theta = self.theta.repeat(self.T,1)
			self.theta = nn.Parameter(self.theta, requires_grad= True)
			self.L_S = self.L_S.repeat(self.T,1)
			self.L_S = nn.Parameter(self.L_S,requires_grad= True)
				
#	############################################################################
		
		if self.learn_L_A:
			if not self.L_A_shared:
				self.L_A = self.L_A.repeat(self.T,1)
			self.L_A =nn.Parameter(self.L_A,requires_grad= True)

		if self.LISTA_A:
			self.W_A_A = torch.eye(self.S.shape[1]) - torch.matmul(self.S[0], torch.div(torch.transpose(self.S[0],0,1), self.L_A))
			self.W_X_A = torch.div(torch.transpose(self.S[0],0,1), self.L_A)
			
			if not self.W_A_A_shared:
				self.W_A_A = self.W_A_A.repeat(self.T,1,1)
			self.W_A_A = nn.Parameter(self.W_A_A,requires_grad=True)
			
			if not self.W_X_A_shared:
				self.W_X_A = self.W_X_A.repeat(self.T,1,1)
			self.W_X_A = nn.Parameter(self.W_X_A,requires_grad=True)	

		if self.LISTA_CP_A:
			self.W_CP_A = torch.div(torch.transpose(self.S[0],0,1), self.L_A)
			
			if not self.W_CP_A_shared:
				self.W_CP_A = self.W_CP_A.repeat(self.T,1,1)
			self.W_CP_A = nn.Parameter(self.W_CP_A,requires_grad=True)
		
		
	def forward(self, X):
	
		b_size = X.shape[0]
		#initialize S
		S_pred = torch.zeros(b_size,self.A.shape[2], X.shape[2])
		S_pred = S_pred.type(torch.DoubleTensor)
		#initialize A
		A_pred = torch.ones([b_size, X.shape[1], self.S.shape[1]])
		n_ = torch.norm(A_pred, dim=1)
		A_pred = A_pred/tile(n_,0,A_pred.shape[1]).reshape(A_pred.shape[0],A_pred.shape[1],A_pred.shape[2])
		A_pred = A_pred.type(torch.DoubleTensor)
		S_s= []
		A_s= []
		
		for t in range(self.T):
			

			# iteration on S
			if self.learn_L_S:
				if self.L_S_shared:	
					S_pred = prox(S_pred - (1/self.L_S)*torch.bmm(torch.transpose(A_pred,1,2), (torch.bmm(A_pred,S_pred)-X)), self.alpha, self.L_S)
				else:
					S_pred = prox(S_pred - (1/self.L_S[t])*torch.bmm(torch.transpose(A_pred,1,2), (torch.bmm(A_pred,S_pred)-X)), self.alpha, self.L_S[t])
		
		
			elif self.LISTA_S:
				if self.W_X_S_shared:
					p1 = torch.bmm(self.W_X_S.repeat(b_size,1,1),X)
				else:
					p1 = torch.bmm(self.W_X_S[t].repeat(b_size,1,1),X)
				if self.W_S_S_shared:
					p2 = torch.bmm(self.W_S_S.repeat(b_size,1,1),S_pred)
				else:
					p2 = torch.bmm(self.W_S_S[t].repeat(b_size,1,1),S_pred)
				p3 = p1+p2
				if self.theta_shared:
					S_pred = prox_theta(p3,self.theta)
				else:
					S_pred = prox_theta(p3,self.theta[t])
					
					
			elif self.LISTA_CP_S:
				if self.W_CP_S_shared:
					S_pred = S_pred + torch.bmm(torch.transpose(self.W_CP_S.repeat(b_size,1,1),1,2),X-torch.bmm(A_pred,S_pred))
				else:
					S_pred = S_pred + torch.bmm(torch.transpose(self.W_CP_S[t].repeat(b_size,1,1),1,2),X-torch.bmm(A_pred,S_pred))
				if self.theta_shared:
					S_pred = prox_theta(S_pred,self.theta)
				else:
					S_pred = prox_theta(S_pred,self.theta[t])
					
			elif self.ISTA_LLT_S:
				S_pred = prox_theta(S_pred - (1/self.L_S[t])*torch.bmm(torch.transpose(A_pred,1,2), (torch.bmm(A_pred,S_pred)-X)), self.theta[t])
		#############################################################################
			#iteration on A
			
			if self.learn_L_A:
				p1 = torch.bmm(torch.bmm(A_pred,S_pred)-X,torch.transpose(S_pred,1,2))
				if self.L_A_shared:
					p2 = torch.div(p1, self.L_A)
				else:
					p2 = torch.div(p1, self.L_A[t])
				A_pred = A_pred - p2
					
			elif self.LISTA_A:
				if self.W_A_A_shared:
					p1 = torch.bmm(A_pred,self.W_A_A.repeat(b_size,1,1))
				else:
					p1 = torch.bmm(A_pred,self.W_A_A[t].repeat(b_size,1,1))
				if self.W_X_A_shared:
					p2 = torch.bmm(X,self.W_X_A.repeat(b_size,1,1))
				else:
					p2 = torch.bmm(X,self.W_X_A[t].repeat(b_size,1,1)) 
				A_pred = p1+p2
				
			elif self.LISTA_CP_A:
				p = torch.bmm(A_pred,S_pred) - X
				if self.W_CP_A_shared:
					A_pred = A_pred - torch.bmm(p,self.W_CP_A.repeat(b_size,1,1))
				else:
					A_pred = A_pred -torch.bmm(p,self.W_CP_A[t].repeat(b_size,1,1))
					
			elif self.non_update_A:
				p1 = torch.bmm(torch.bmm(A_pred,S_pred)-X,torch.transpose(S_pred,1,2))
				L_A = torch.tensor((1.001) * np.linalg.norm(S_pred[0].detach().numpy(),ord=2)**2)
				p2 = torch.div(p1, L_A)
				A_pred = A_pred - p2			
			
			#projection on the unit ball		
			for i in range(A_pred.shape[0]):
				for j in range(A_pred.shape[2]):
					if(torch.any(torch.norm(A_pred[i][:,j])>torch.tensor([1.]))):
						A_pred[i][:,j] /= torch.norm(A_pred[i][:,j].clone())
						
		#############################################################################

			A_s.append(A_pred)
			S_s.append(S_pred)

		return S_pred, A_pred , S_s , A_s
