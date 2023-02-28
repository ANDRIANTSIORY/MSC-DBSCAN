# Packages
from scipy.io import loadmat
import numpy as np
from numpy import linalg as la
import random
import msc_multiple as msc_multiple
import multiple_dbscan as multiple_dbscan
import itertools
import fonctions as f
# -----------------------------

def run() :
	data = loadmat('../../data/Flow_Injection/fia.mat')
	dt = data['X']
	data = np.reshape(dt, (12,100,89), order="F")

	e_ = 0.00013
	res = msc_multiple.Msc(data, e_ = e_)
  
	msc_output = res.get_result_triclustering()
	sim_matrices = res.get_cij()
    
	# MSC-Dbscan
	index = []
	for j in range(3):
		intermediate = []
		for i in range(len(msc_output[0])):
			intermediate = intermediate  + msc_output[j][i] 
		index.append(list(set(intermediate)))

	dim = data.shape
	MscExtension = multiple_dbscan.Msc_extension(dim, sim_matrices, index, e_, minP=2)

	# -----The clusters------
	print("MSC cluster : ", msc_output)
	print("MSC_DBSCAN cluster: ", MscExtension._cluster_real_index)


	# Mean Square Error
	mse_msc = []
	for j in itertools.product(msc_output[0], msc_output[1], msc_output[2]):
		mse_msc.append(f.rmse(data, j[0], j[1], j[2]))

	mse_dbscan = []
	for j in itertools.product(MscExtension._cluster_real_index[0], MscExtension._cluster_real_index[1], MscExtension._cluster_real_index[2]):
		mse_dbscan.append(f.rmse(data, j[0], j[1], j[2]))
    
	rmse_MSC = (np.mean(mse_msc)**0.5)
	rmse_Extension = (np.mean(mse_dbscan)**0.5)
    
	# ---- RMSE -----------
	print("RMSE MSC : ", rmse_MSC, )
	print("RMSE MSC-extension : ", rmse_Extension)