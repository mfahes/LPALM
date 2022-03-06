import numpy as np
import copy as cp
from utils.utils import prox_l1, prox_oblique
from utils.munkres import Munkres

def PALM(X,lamb=0,nbIt = 5e6,Ainit=1):
    '''
    Function to apply PALM on a mixture X
    
    Input: Multispectral image X
    Output: S_est: estimation of S, A_est: estimation of A, it:number of iterations
    lamb: Regularization parameter
    nbIt: Maximum number of iterations
    Ainit: initialization of the mixing matrix A
    '''
    eps=0.01    
    A_est = Ainit.copy() #init_A_est
    S_est = np.dot(np.linalg.pinv(A_est),X) #init_S_est
    
    A_est_prev = A_est.copy()
    S_est_prev = S_est.copy()
    
    it = 0
    while((np.linalg.norm(S_est-S_est_prev) > 1e-7 or np.linalg.norm(A_est-A_est_prev) > 1e-7 or it < 2) and it < nbIt):

        if np.mod(it,100) == 2:
            print(it)
            print('Cs : %s, Ca : %s'%(np.linalg.norm(S_est-S_est_prev),np.linalg.norm(A_est-A_est_prev)))
        
        if it > 0:
            S_est_prev = cp.copy(S_est)
            A_est_prev = cp.copy(A_est)
            
            
        # S update
        gamma = 1./((np.linalg.norm(A_est,ord=2)**2)*(1.+eps))
        S_est = S_est + gamma*A_est.T@(X - A_est@S_est)
        

        S_est = prox_l1(S_est,lamb*gamma)
        
        # A update 
        eta = 1./((np.linalg.norm(S_est,ord=2)**2) *(1+eps))
        A_est = A_est + eta*(X - A_est@S_est)@S_est.T
        

        A_est = prox_oblique(A_est)

        it += 1
    return A_est,S_est,it
