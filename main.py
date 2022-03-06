import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,Subset
import matplotlib.pyplot as plt
import matplotlib
from utils.utils import train_val_dataset,split_A,plot_train_test_loss
from utils.create_mixtures import My_dataset
from train import train_non_blind,train_blind
from ISTA import ISTA
from PALM import PALM
import argparse
import numpy as np
import pickle
import seaborn as sns
import statistics
from evaluation_realistic import apply_LPALM_realistic,apply_LISTA_realistic

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 11}
matplotlib.rc('font', **font)

parser = argparse.ArgumentParser()
parser.add_argument('--realistic', action= 'store_true', required=False)
parser.add_argument('--create_dataset', action= 'store_true', required=False)
parser.add_argument('--synthetic_exp', action= 'store_true', required=False)
parser.add_argument('--realistic_exp', action= 'store_true', required=False)

parser.add_argument('--ISTA', action= 'store_true', required=False)
parser.add_argument('--LISTA_CP',action = 'store_true', required=False)
parser.add_argument('--LISTA_LeCun',action = 'store_true', required=False)
parser.add_argument('--ISTA_LLT',action = 'store_true', required=False)
parser.add_argument('--LPALM', action = 'store_true', required = False)
parser.add_argument('--shared_L', action='store_true', required= False) 
parser.add_argument('--share_theta',action = 'store_true' ,required = False)
parser.add_argument('--share_We', action = 'store_true',required=False)
parser.add_argument('--share_G', action = 'store_true',required = False)
parser.add_argument('--share_W_CP', action= 'store_true',required= False)
parser.add_argument('--learn_L_S', action= 'store_true', required= False)
parser.add_argument('--L_S_shared', action= 'store_true', required= False)
parser.add_argument('--LISTA_S', action= 'store_true', required= False)
parser.add_argument('--W_X_S_shared', action= 'store_true', required= False)
parser.add_argument('--W_S_S_shared', action= 'store_true', required= False)
parser.add_argument('--theta_shared', action= 'store_true', required= False)
parser.add_argument('--LISTA_CP_S', action= 'store_true', required= False)
parser.add_argument('--W_CP_S_shared', action= 'store_true', required= False)
parser.add_argument('--ISTA_LLT_S',action='store_true', required=False)
parser.add_argument('--learn_L_A', action= 'store_true', required= False)
parser.add_argument('--L_A_shared', action= 'store_true', required= False)
parser.add_argument('--LISTA_A', action= 'store_true', required= False)
parser.add_argument('--W_A_A_shared', action= 'store_true', required= False)
parser.add_argument('--W_X_A_shared', action= 'store_true', required= False)
parser.add_argument('--LISTA_CP_A', action= 'store_true', required= False)
parser.add_argument('--W_CP_A_shared', action= 'store_true', required= False)
parser.add_argument('--non_update_A',action='store_true',required=False)
parser.add_argument('--add_noise_A', action= 'store_true', required= False)

parser.add_argument('--LISTA_realistic',action= 'store_true', required= False)
parser.add_argument('--LPALM_realistic',action= 'store_true', required= False)

parser.add_argument('--T', type=int, default=25)
parser.add_argument('--lf', type=str, default='supervised_A_S')

parser.add_argument('--model_path', type=str)

args = parser.parse_args()


if args.create_dataset:
	dataset = My_dataset(realistic=args.realistic) #synthetic=False to generate synthetic data for realistic experiment training
	if args.realistic:
		with open('dataset_4_sources_realistic.pickle', 'wb') as f:
			pickle.dump(dataset, f)
	else:
		with open('dataset_4_sources_synthetic.pickle', 'wb') as f:
			pickle.dump(dataset, f)

if args.synthetic_exp:
	with open('dataset_4_sources_synthetic.pickle', 'rb') as f:
			dataset = pickle.load(f)
if args.realistic_exp:
	with open('dataset_4_sources_realistic.pickle', 'rb') as f:
			dataset = pickle.load(f)
			
datasets = split_A(dataset) #used to roughly create a relatively "hard" validation set (using the sum of cosine similarity of A columns)  
#comment the line above and uncomment the one below to split the dataset randomly.
#datasets = train_val_dataset(dataset, val_split= 1/6)
train_set = datasets['train']
val_set = datasets['val']
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set,batch_size=1, shuffle=True, num_workers=4)
			
if args.ISTA:
	lmda = 1.1e-5
	nb_iter=25
	if args.add_noise_A:
		noise_A = 10
	else:
		noise_A = None
	S_est , loss, loss_per_layer = ISTA(val_set,lmda,nb_iter,nb_iter_fixed = True,noise_A=noise_A)

if args.ISTA_LLT:
	
	alpha =1e-5
	mode = 'LLT'
	if args.add_noise_A:
		noise_A = 10
	else:
		noise_A = None
	train_total_loss, test_total_loss, model = train_non_blind(train_loader, val_loader, num_epochs=100, T= 25 , alpha = alpha, mode=mode, L_shared = args.shared_L, theta_shared=args.share_theta,We_shared=args.share_We,G_shared=args.share_G ,W_shared=args.share_W_CP ,noise_A=noise_A,realistic_train = args.realistic_exp)
	plot_train_test_loss(train_total_loss,test_total_loss)

	
if args.LISTA_CP:
	alpha = 1e-5
	mode = 'CP'
	if args.add_noise_A:
		noise_A = 10
	else:
		noise_A = None
	train_total_loss, test_total_loss,model = train_non_blind(train_loader, val_loader, num_epochs=100, T= 25 , alpha = alpha,mode=mode,L_shared = args.shared_L, theta_shared=args.share_theta,We_shared=args.share_We,G_shared=args.share_G ,W_shared=args.share_W_CP ,noise_A=noise_A,realistic_train = args.realistic_exp)
	plot_train_test_loss(train_total_loss,test_total_loss)
	
if args.LISTA_LeCun:
	alpha= 1e-5
	mode = 'LC'
	print(args.share_theta)
	print(args.share_We)
	print(args.share_G)
	train_total_loss, test_total_loss, model = train_non_blind(train_loader, val_loader, num_epochs=100, T= 25 , alpha = alpha,mode=mode,L_shared = args.shared_L, theta_shared=args.share_theta,We_shared=args.share_We,G_shared=args.share_G ,W_shared=args.share_W_CP ,noise_A=None, realistic_train = args.realistic_exp)
	plot_train_test_loss(train_total_loss,test_total_loss)


if args.LPALM:

	train_total_loss, test_total_loss, model = train_blind(train_loader, val_loader, num_epochs=100, T = args.T, learn_L_S = args.learn_L_S ,L_S_shared=args.L_S_shared, LISTA_S = args.LISTA_S, W_X_S_shared= args.W_X_S_shared , W_S_S_shared= args.W_S_S_shared ,theta_shared= args.theta_shared,  LISTA_CP_S = args.LISTA_CP_S, W_CP_S_shared= args.W_CP_S_shared ,ISTA_LLT_S = args.ISTA_LLT_S, learn_L_A = args.learn_L_A , L_A_shared= args.L_A_shared , LISTA_A = args.LISTA_A , W_A_A_shared = args.W_A_A_shared, W_X_A_shared = args.W_X_A_shared , LISTA_CP_A = args.LISTA_CP_A, W_CP_A_shared = args.W_CP_A_shared,non_update_A= args.non_update_A, loss_funct = args.lf,realistic_train = args.realistic_exp)
	plot_train_test_loss(train_total_loss,test_total_loss)

if args.LISTA_realistic:
	NMSE_A_list = []
	NMSE_S_list = []
	for i,(X,A,S) in enumerate(val_loader):
		NMSE_A, NMSE_S = apply_LISTA_realistic(A[0], args.model_path) #'model_path' is the path for the trained model used for inference
		NMSE_A_list.append(NMSE_A)
		NMSE_S_list.append(NMSE_S)
		print(NMSE_A,NMSE_S)
	print(statistics.median(NMSE_A_list))
	
if args.LPALM_realistic:
	NMSE_A_list = []
	NMSE_S_list = []
	for i,(X,A,S) in enumerate(val_loader):
		NMSE_A, NMSE_S = apply_LPALM_realistic(A[0],args.model_path) #'model_path' is the path for the trained model used for inference
		NMSE_A_list.append(NMSE_A)
		NMSE_S_list.append(NMSE_S)
		print(NMSE_A,NMSE_S)
	print(statistics.median(NMSE_A_list))
