import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from random import shuffle
import pickle
from utils.starlet import Starlet_Forward,Starlet_Inverse
import statistics
from ISTA_A import ISTA_A

def invert_permutation(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s
    
def perm_w(w):
	w = w.ravel()
	array = np.arange(len(w))         
	np.random.shuffle(array)         
	w = w[array]
	w_1d = w.reshape(1,w.size)
	return w_1d, array
		
def get_w_and_c():
	with open('data/real_sources.pkl', 'rb') as f:
		sources = pickle.load(f)

	c1,w1 = Starlet_Forward(sources['sync'],h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3)
	c2,w2 = Starlet_Forward(sources['therm'],h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3)
	c3,w3 = Starlet_Forward(sources['fe1'],h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3)
	c4,w4 = Starlet_Forward(sources['fe2'],h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3)
	
	perm = []
	w1_1d = w1.reshape(1,w1.size)
	w2_1d = w2.reshape(1,w2.size)
	w3_1d = w3.reshape(1,w3.size)
	w4_1d = w4.reshape(1,w4.size)

	#permutation of each source fine scale to limit the impact of partial correlation
	w1_1d, array1 = perm_w(w1)
	perm.append(array1)
	w2_1d, array2 = perm_w(w2)
	perm.append(array2)
	w3_1d, array3 = perm_w(w3)
	perm.append(array3)
	w4_1d, array4 = perm_w(w4)
	perm.append(array4)
	
	c1_1d = c1.reshape(1,c1.size)
	c2_1d = c2.reshape(1,c2.size)
	c3_1d = c3.reshape(1,c3.size)
	c4_1d = c4.reshape(1,c4.size)
	w = np.concatenate((w1_1d,w2_1d,w3_1d,w4_1d))
	c = np.concatenate((c1_1d,c2_1d,c3_1d,c4_1d))
	return c,w,sources,perm

def create_realistic_X(A):
	c,w,sources,perm = get_w_and_c()
	normw = np.zeros(w.shape[0])
	normc = np.zeros(c.shape[0])
	for j in range(w.shape[0]):
		normw[j] = np.linalg.norm(w[j,:]) 
		w[j,:] = w[j,:]/normw[j]
	#	plt.imshow(w[j,:].reshape(346,346),cmap='jet')
	#	plt.colorbar()
	#	plt.show()
	for j in range(c.shape[0]):
		normc[j] = np.linalg.norm(c[j,:])
		c[j,:] = c[j,:]/normc[j] 
	#	plt.imshow(c[j,:].reshape(346,346),cmap='jet')
	#	plt.colorbar()
	#	plt.show()
	X_real_tilde_w = torch.matmul(A,torch.from_numpy(w))
	X_real_tilde_c = torch.matmul(A,torch.from_numpy(c))
	X_real = Starlet_Inverse(X_real_tilde_c.numpy(),X_real_tilde_w.numpy().reshape(X_real_tilde_w.shape[0],X_real_tilde_w.shape[1],1))
	N = np.random.randn(X_real.shape[0],X_real.shape[1])
	SNR=30
	N = 10.**(-SNR/20.)*np.linalg.norm(X_real)/np.linalg.norm(N)*N
	X_real = X_real + N
	return X_real,w,c,normw,normc,sources,perm,A
		

def apply_LPALM_realistic(A,model_name):
	X_real,w,c,normw,normc,sources,perm,A = create_realistic_X(A)
	X_real_tilde_w = np.empty_like(X_real) #this will be the detail scale
	X_real_tilde_c = np.empty_like(X_real) #this will be the coarse scale
	for j in range(X_real.shape[0]):
		#get the coarse and detail scale of X
		c_ , w_ = Starlet_Forward(X_real[j].reshape(346,346),h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3) 
		X_real_tilde_c[j,:] , X_real_tilde_w[j,:] = c_.reshape(c_.size) , w_.reshape(w_.size) 
	X_real_tilde_c = torch.from_numpy(X_real_tilde_c)
	X_real_tilde_w = torch.from_numpy(X_real_tilde_w)
	criterion = nn.MSELoss()
	model = torch.load(model_name)
	model.eval()
	with torch.no_grad():
		X_real_tilde_w = X_real_tilde_w.reshape(1,X_real_tilde_w.shape[0],X_real_tilde_w.shape[1])
		S_pred_tilde_w, A_pred, S_s, A_s = model(X_real_tilde_w) #apply the model on the fine scale of X to obtain the fine scale of S_pred and A_pred
		S_pred_tilde_c = torch.matmul(torch.pinverse(A_pred),X_real_tilde_c) #coarse scale of S_pred
	for j in range(w.shape[0]): 
		S_pred_tilde_w[0,j,:] = S_pred_tilde_w[0,j,:]*normw[j] #correct the scale
		S_pred_tilde_w[0,j,:] = S_pred_tilde_w[0,j,:][invert_permutation(perm[j])] #invert the permutation used to avoid partial correlation
	for j in range(c.shape[0]): 
		S_pred_tilde_c[0,j,:] = S_pred_tilde_c[0,j,:]*normc[j] #correct the scale
		
	S_pred =  Starlet_Inverse(S_pred_tilde_c.numpy().reshape(S_pred_tilde_c.shape[1],S_pred_tilde_c.shape[2]),S_pred_tilde_w.numpy().reshape(S_pred_tilde_w.shape[1],S_pred_tilde_w.shape[2],1)) #get S_pred

	#for j in range(S_pred.shape[0]):
	#	im = plt.imshow(S_pred[j,:].reshape(346,346))
	#	im.set_cmap('jet')
	#	plt.colorbar(im)
	#	plt.show()
	#for j in range(A_pred.shape[2]):
	#	plt.plot(A_pred[0,:,j],label='predicted')
	#	plt.plot(A[:,j],label='ground truth')
	#	plt.legend()
	#	plt.show()

	S_pred = torch.from_numpy(S_pred)
	sources['sync'] = sources['sync'].reshape(1,346*346)
	sources['therm'] = sources['therm'].reshape(1,346*346)
	sources['fe1'] = sources['fe1'].reshape(1,346*346)
	sources['fe2'] = sources['fe2'].reshape(1,346*346)
	GT = np.concatenate((sources['sync'],sources['therm'],sources['fe1'],sources['fe2']))
	GT = torch.from_numpy(GT)
	NMSE_S = (torch.numel(GT) * criterion(GT, S_pred) / torch.norm(GT)**2).item()
	NMSE_A = (torch.numel(A) * criterion(A, A_pred[0,:,:]) / torch.norm(A)**2).item()
	return NMSE_A,NMSE_S
	
def apply_LISTA_realistic(A,model_name):
	
	X_real,w,c,normw,normc,sources,perm,A = create_realistic_X(A)
	X_real_tilde_w = np.empty_like(X_real) #this will be the detail scale
	X_real_tilde_c = np.empty_like(X_real) #this will be the coarse scale
	for k in range(X_real.shape[0]):
		#get the coarse and detail scale of X
		c_ , w_ = Starlet_Forward(X_real[k].reshape(346,346),h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3)
		X_real_tilde_c[k,:] , X_real_tilde_w[k,:] = c_.reshape(c_.size) , w_.reshape(w_.size)
	X_real_tilde_c = torch.from_numpy(X_real_tilde_c)
	X_real_tilde_w = torch.from_numpy(X_real_tilde_w)
	criterion = nn.MSELoss()
	model = torch.load(model_name)
	model.eval()
	with torch.no_grad():
		X_real_tilde_w = X_real_tilde_w.reshape(1,X_real_tilde_w.shape[0],X_real_tilde_w.shape[1])
		S_pred_tilde_w,S_s_train= model(X_real_tilde_w) #apply the model on the fine scale of X to obtain the fine scale of S_pred
		S_pred_tilde_w = S_pred_tilde_w.type(torch.DoubleTensor)
		#ISTA_on_A (to estimate A from the fine scales of X and S_pred)
		##########################################################
		A_pred,it = ISTA_A(X_real_tilde_w[0],S_pred_tilde_w[0])
		##############################################################
		S_pred_tilde_c = torch.matmul(torch.pinverse(A_pred),X_real_tilde_c) #coarse scale of S_pred
		S_pred_tilde_c = S_pred_tilde_c.reshape(1,S_pred_tilde_c.shape[0],S_pred_tilde_c.shape[1])
	for j in range(w.shape[0]): 
		S_pred_tilde_w[0,j,:] = S_pred_tilde_w[0,j,:]*normw[j] #correct the scale
		S_pred_tilde_w[0,j,:] = S_pred_tilde_w[0,j,:][invert_permutation(perm[j])] #invert the permutation used to avoid partial correlation
		
	for j in range(c.shape[0]): 
		S_pred_tilde_c[0,j,:] = S_pred_tilde_c[0,j,:]*normc[j] #correct the scale 

	S_pred =  Starlet_Inverse(S_pred_tilde_c.numpy().reshape(S_pred_tilde_c.shape[1],S_pred_tilde_c.shape[2]),S_pred_tilde_w.numpy().reshape(S_pred_tilde_w.shape[1],S_pred_tilde_w.shape[2],1)) # get S_pred

	#for i in range(S_pred.shape[0]):	
	#	im = plt.imshow(S_pred[i,:].reshape(346,346))
	#	im.set_cmap('jet')
	#	plt.colorbar(im)
	#	plt.show()

	#for k in range(A_pred.shape[1]):
	#	plt.figure(figsize=(32,24))
	#	plt.plot(A_pred[:,k],label='predicted',linewidth=6)
	#	plt.plot(A[:,k],label='ground truth',linewidth=6)
	#	plt.legend(fontsize='medium')
	#	plt.show()

	S_pred = torch.from_numpy(S_pred)
	sources['sync'] = sources['sync'].reshape(1,346*346)
	sources['therm'] = sources['therm'].reshape(1,346*346)
	sources['fe1'] = sources['fe1'].reshape(1,346*346)
	sources['fe2'] = sources['fe2'].reshape(1,346*346)
	GT = np.concatenate((sources['sync'],sources['therm'],sources['fe1'],sources['fe2']))
	GT = torch.from_numpy(GT)
	NMSE_A = (torch.numel(A) * criterion(A, A_pred) / torch.norm(A)**2).item()
	NMSE_S = (torch.numel(GT) * criterion(GT, S_pred) / torch.norm(GT)**2).item()
	return NMSE_A,NMSE_S
