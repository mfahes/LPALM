import numpy as np
from scipy.stats import gennorm
from torch.utils.data import Dataset

def load_data():

	spectres = np.load('./data/spectra.npy', allow_pickle= True)
	spectres = spectres.reshape(1)[0]
	A1 = spectres['sync']
	A2 = spectres['therm'][:900]
	A3 = spectres['fe1']
	A4 = spectres['fe2']
		
	return A1,A2,A3,A4

def mixing_synthetic(beta = 0.3 , t =500, noise=True,normalize_S_lines = False):
	
	A1,A2,A3,A4 = load_data()
	
	X = np.zeros((A1.shape[0],A1.shape[1],t))
	A = np.zeros((A1.shape[0],A1.shape[1],4))
	S = np.zeros((A1.shape[0],4,t))
	for i in range(A1.shape[0]):
		A[i] = np.concatenate((A1[i].reshape(-1,1),A2[i].reshape(-1,1),A3[i].reshape(-1,1),A4[i].reshape(-1,1)) , axis = 1)
		S1 = gennorm.rvs(beta,size=(1,t))
		S2 = gennorm.rvs(beta,size=(1,t))
		S3 = gennorm.rvs(beta,size=(1,t)) 
		S4 = gennorm.rvs(beta,size=(1,t))
		S[i]=np.concatenate((S1,S2,S3,S4))
		if normalize_S_lines:
			for j in range(S.shape[1]):
				S[i][j,:] = S[i][j,:]/np.linalg.norm(S[i][j,:]) 
		X[i] = np.matmul(A[i],S[i])
		if noise:
			SNR = 30
			N = np.random.randn(X[i].shape[0],X[i].shape[1])
			N = 10.**(-SNR/20.)*np.linalg.norm(X[i])/np.linalg.norm(N)*N
			X[i] = X[i] + N
			
	return (X,A,S)

def mixing_realistic(beta1 = 0.28 ,beta2=0.28, beta3=0.28,beta4=0.28, t = 3000, noise=True,normalize_S_lines = True):
	A1,A2,A3,A4 = load_data()
	
	X = np.zeros((A1.shape[0],A1.shape[1],t))
	A = np.zeros((A1.shape[0],A1.shape[1],4))
	S = np.zeros((A1.shape[0],4,t))
	for i in range(A1.shape[0]):
		A[i] = np.concatenate((A1[i].reshape(-1,1),A2[i].reshape(-1,1),A3[i].reshape(-1,1),A4[i].reshape(-1,1)) , axis = 1)
		S1 = gennorm.rvs(beta3,loc=0, scale=5*(10**-7),size=(1,int(0.76*t)))
		S2 = gennorm.rvs(beta4,loc=0, scale=9*(10**-7),size=(1,int(0.35*t)))
		S3 = gennorm.rvs(beta1,loc=0, scale=2*(10**-6),size=(1,int(0.27*t)))
		S4 = gennorm.rvs(beta2,loc=0, scale=4*(10**-6),size=(1,int(0.15*t)))
		_1 = np.zeros((1,int(0.24*t)))
		_2 = np.zeros((1,int(0.65*t)))
		_3 = np.zeros((1,int(0.73*t)))
		_4 = np.zeros((1,int(0.85*t)))
		S1 = np.concatenate((S1,_1),axis=None).reshape(1,S1.size+_1.size)
		S2 = np.concatenate((S2,_2),axis=None).reshape(1,S2.size+_2.size)
		S3 = np.concatenate((S3,_3),axis=None).reshape(1,S3.size+_3.size)
		S4 = np.concatenate((S4,_4),axis=None).reshape(1,S4.size+_4.size)
		np.random.shuffle(np.transpose(S1))
		np.random.shuffle(np.transpose(S2))
		np.random.shuffle(np.transpose(S3))
		np.random.shuffle(np.transpose(S4))
		S[i]=np.concatenate((S1,S2,S3,S4))
		if normalize_S_lines:
			for j in range(S.shape[1]):
				S[i][j,:] = S[i][j,:]/np.linalg.norm(S[i][j,:]) 
		X[i] = np.matmul(A[i],S[i])
		if noise:
			SNR = 30
			N = np.random.randn(X[i].shape[0],X[i].shape[1])
			N = 10.**(-SNR/20.)*np.linalg.norm(X[i])/np.linalg.norm(N)*N
			X[i] = X[i] + N
	
	return (X,A,S)

class My_dataset(Dataset):
	def __init__(self, realistic=False):
		if realistic == False:
			self.X, self.A, self.S = mixing_synthetic(beta=0.3, t=500, noise= True, normalize_S_lines= True)
		else:
			self.X, self.A, self.S = mixing_realistic(beta1 = 0.75 ,beta2=0.75, beta3=0.75 ,beta4=0.75, t = 3000, noise=True,normalize_S_lines = True)
	def __getitem__(self, item):
		return self.X[item], self.A[item], self.S[item]
	def __len__(self):
		return len(self.X)
