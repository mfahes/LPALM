import torch
import torch.nn as nn
import numpy as np
from ISTA_LLT import ISTA_LLT
from LISTA_LeCun import LISTA_LeCun
from LISTA_CP import LISTA_CP
from LPALM import LPALM
from param_PALM import param_PALM 
import matplotlib.pyplot as plt
from scipy.io import savemat
import os

def train_non_blind(train_loader, val_loader, num_epochs=10, T=10, alpha = 10, mode = None,L_shared = True, theta_shared=True,We_shared=True,G_shared=True,W_shared=True,noise_A=None,realistic_train=False):

	criterion = nn.MSELoss()
	layers = np.arange(T)
	A = next(iter(train_loader))[1]
	
	if mode == 'LC':
		model = LISTA_LeCun(T, alpha= alpha, A=A,theta_shared=theta_shared,We_shared=We_shared,G_shared= G_shared)
		
	elif mode == 'CP':
		model = LISTA_CP(T,alpha=alpha,A=A,theta_shared=theta_shared, W_shared= W_shared)
		
	elif mode == 'LLT':
		model = ISTA_LLT(T,alpha=alpha,A=A,L_shared=L_shared, theta_shared= theta_shared)
		
	else: 
		print("The entered mode doesn't correspond to a model")
	
	total_params = [p.numel() for p in model.parameters()]
	print(total_params)

	optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999))

	train_total_loss = []
	val_total_loss = []
	
	list_err_val_layer = [[0 for i in range(T)] for j in range(num_epochs)]
	
	for epoch in range(num_epochs):
		train_total = 0
		model.train()
		for i, (X,A,S) in enumerate(train_loader):

			optimizer.zero_grad()
			if mode == 'LC':
				S_pred, S_s = model(X)
			else:
				S_pred, S_s = model(X,A,noise_A=noise_A)
			train_loss = torch.numel(S) * criterion(S.float(), S_pred.float()) / torch.norm(S.float())**2
			train_loss.backward()
			optimizer.step()
			train_total += train_loss.item()
		train_total /= i+1
		train_total_loss.append(train_total)

		model.eval()
		with torch.no_grad():
			val_total = 0
			for i, (X,A,S) in enumerate(val_loader):
				if mode == 'LC':
					S_pred, S_s_val = model(X)
				else:
					S_pred,S_s_val = model(X,A,noise_A=noise_A)
				val_loss = torch.numel(S) *  criterion(S.float(), S_pred.float()) / torch.norm(S.float())**2
				val_total += val_loss.item()
				for j,s in enumerate(S_s_val):
					list_err_val_layer[epoch][j] += torch.numel(S) *criterion(S.float(), s.float()).item() / torch.norm(S.float())**2
			list_err_val_layer[epoch] = [x/(i+1) for x in list_err_val_layer[epoch]]
			val_total /= i+1
			val_total_loss.append(val_total)
		
		print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
		if epoch % 5 == 0:
			print()

		print("epoch:{} | training loss:{:.5f} | validation loss:{:.5f} ".format(epoch, train_total,val_total))
		dirName = 'models_'+ mode + '_realistic_t' if realistic_train else 'models_'+ mode +'_synthetic'
		if noise_A: dirName = dirName+ str(noise_A)
		try:
    			os.mkdir(dirName)
    			print("Directory " , dirName ,  " Created ") 
		except FileExistsError:
    			print("Directory " , dirName ,  " already exists")
		torch.save(model, dirName+'/'+ mode +str(epoch)+'.pth')

	return train_total_loss, val_total_loss,model

def train_blind(train_loader, val_loader, num_epochs=10, T=10,learn_L_S = False ,L_S_shared=False, LISTA_S = False, W_X_S_shared= False, W_S_S_shared= False,theta_shared= False,  LISTA_CP_S = True, W_CP_S_shared= False, ISTA_LLT_S = False, learn_L_A = True , L_A_shared= False, LISTA_A = False , W_A_A_shared = False, W_X_A_shared = False, LISTA_CP_A = False, W_CP_A_shared = False,non_update_A = False,loss_funct='supervised_A_S',realistic_train=False):
	
	torch.autograd.set_detect_anomaly(True)
	criterion = nn.MSELoss()
	
	A = next(iter(train_loader))[1]
	S = next(iter(train_loader))[2]
	model = param_PALM(T, S=S , A=A,learn_L_S = learn_L_S ,L_S_shared=L_S_shared, LISTA_S = LISTA_S, W_X_S_shared= W_X_S_shared , W_S_S_shared= W_S_S_shared ,theta_shared= theta_shared,  LISTA_CP_S = LISTA_CP_S, W_CP_S_shared= W_CP_S_shared , ISTA_LLT_S = ISTA_LLT_S, learn_L_A = learn_L_A , L_A_shared= L_A_shared , LISTA_A = LISTA_A , W_A_A_shared = W_A_A_shared, W_X_A_shared = W_X_A_shared , LISTA_CP_A = LISTA_CP_A, W_CP_A_shared = W_CP_A_shared, non_update_A = non_update_A)
	
	total_params = [p.numel() for p in model.parameters()]
	print(total_params)
	
	optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999))

	train_total_loss = []
	val_total_loss = []
	
	list_err_val_S_layer = [[0 for i in range(T)] for j in range(num_epochs)]
	list_err_val_A_layer = [[0 for i in range(T)] for j in range(num_epochs)]
	
	for epoch in range(num_epochs):
		train_total = 0
		model.train()
		for i, (X,A,S) in enumerate(train_loader):

			optimizer.zero_grad()
			S_pred, A_pred, S_s_train, A_s_train = model(X)
			
			train_loss=0
			if loss_funct == 'unsupervised':
				train_loss = torch.numel(X) * criterion(X.float(), torch.bmm(A_pred.float(),S_pred.float())) / torch.norm(X.float())**2
			elif loss_funct == 'supervised_A':
				train_loss = torch.numel(A) * criterion(A.float(), A_pred.float()) / torch.norm(A.float())**2
			elif loss_funct == 'supervised_S':
				train_loss = torch.numel(S) * criterion(S.float(), S_pred.float()) / torch.norm(S.float())**2
			elif loss_funct == 'supervised_A_S':
				train_loss = torch.numel(S) * criterion(S.float(), S_pred.float()) / torch.norm(S.float())**2 + torch.numel(A) * criterion(A.float(),A_pred.float()) / torch.norm(A.float())**2
			train_loss.backward()
			optimizer.step()
			train_total += train_loss.item()
		train_total /= i+1
		train_total_loss.append(train_total)
		
		model.eval()
		with torch.no_grad():
			val_total = 0
			for i, (X,A,S) in enumerate(val_loader):
				S_pred, A_pred, S_s_val, A_s_val = model(X)
				val_loss = torch.numel(S) * criterion(S.float(), S_pred.float()) / torch.norm(S.float())**2 + torch.numel(A) * criterion(A.float(),A_pred.float()) / torch.norm(A.float())**2
				val_total += val_loss.item()
				for j,s in enumerate(S_s_val):
					list_err_val_S_layer[epoch][j] += torch.numel(S) * criterion(S.float(), s.float()).item() / torch.norm(S.float())**2
				for j,a in enumerate(A_s_val):
					list_err_val_A_layer[epoch][j] += torch.numel(A) * criterion(A.float(), a.float()).item() / torch.norm(A.float())**2
					
			list_err_val_S_layer[epoch] = [x/(i+1) for x in list_err_val_S_layer[epoch]]
			list_err_val_A_layer[epoch] = [x/(i+1) for x in list_err_val_A_layer[epoch]]
			val_total /= i+1
			val_total_loss.append(val_total)

		
		print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
		if epoch % 5 == 0:
			print()

		print("epoch:{} | training loss:{:.5f} | validation loss:{:.5f} ".format(epoch, train_total,val_total))
		dirName = 'models_LPALM_realistic_t' if realistic_train else 'models_LPALM_synthetic'
		try:
    			os.mkdir(dirName)
    			print("Directory " , dirName ,  " Created ") 
		except FileExistsError:
    			print("Directory " , dirName ,  " already exists")
		torch.save(model, dirName+'/LPALM'+str(epoch)+'.pth')
	
	return train_total_loss, val_total_loss,model
