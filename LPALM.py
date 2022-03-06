import torch
import torch.nn as nn
import numpy as np
from utils.utils import prox_theta,tile,prox

class LPALM(nn.Module):
    def __init__(self, T=10,S=None,A=None):
        super(LPALM, self).__init__()
        
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.T = T
        self.S = S
        self.A = A

        self.alpha =0.00001
        self.L_S = torch.tensor((1.001) * np.linalg.norm(self.A[0].numpy(),ord=2)**2)
        self.L_A = torch.tensor((1.001) * np.linalg.norm(self.S[0].numpy(),ord=2)**2)
			
        self.theta = (self.alpha / self.L_S).clone().detach()			
        self.We = torch.div(torch.transpose(self.A[0],0,1), self.L_S)
        self.W_CP_S = torch.transpose(self.We,0,1) # Transpose because W.T is used inside of the update.
        self.theta = self.theta.repeat(self.T,1)
        self.theta = nn.Parameter(self.theta, requires_grad = True)
        self.W_CP_S = self.W_CP_S.repeat(self.T,1,1)
        self.W_CP_S = nn.Parameter(self.W_CP_S, requires_grad = True)

        self.L_A = self.L_A.repeat(self.T,1)
        self.L_A = nn.Parameter(self.L_A,requires_grad = True)
		
		
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
            
            #iteration on S
            S_pred = S_pred + torch.bmm(torch.transpose(self.W_CP_S[t].repeat(b_size,1,1),1,2),X-torch.bmm(A_pred,S_pred))
            S_pred = prox_theta(S_pred,self.theta[t])
					
            #iteration on A
            grad_A = torch.bmm(torch.bmm(A_pred,S_pred)-X,torch.transpose(S_pred,1,2))
            p = torch.div(grad_A, self.L_A[t])
            A_pred = A_pred - p			
            #projection on the unit ball		
            for i in range(A_pred.shape[0]):# For all the A in the mini-batch
                for j in range(A_pred.shape[2]):# For all the columns of A
                    if(torch.any(torch.norm(A_pred[i][:,j])>torch.tensor([1.]))):
                        A_pred[i][:,j] /= torch.norm(A_pred[i][:,j].clone())
            S_s.append(S_pred)
            A_s.append(A_pred)
        return S_pred, A_pred, S_s, A_s
