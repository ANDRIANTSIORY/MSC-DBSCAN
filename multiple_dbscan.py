
import numpy as np
from sklearn.cluster import DBSCAN

class Msc_extension():
	
	# data_dim is the dimension of the data
	# similarit is the similarity matrix of each mode, list of three similarity matrix
	# index is the selected element of clusters  (its element should be arranged in ascending order)
	# epsilon is the parameter of the MSC algorithm
    # minP : minimum number of neighborhood
	
	def __init__(self, dim_data, similarity, index, epsilon, minP=2):


		self._result = []
		self._index = index
		for h in range(3):
			data = similarity[h].T
			data = data[self._index[h],:]
			eps = (len(self._index[h]) * epsilon / 2  +  (np.log(dim_data[h] - len(index[h])))**0.5 )**0.5
			clustering = DBSCAN(eps=eps, min_samples=minP).fit(np.abs(data))
			cluster = clustering.labels_
			self._result.append(cluster)



		self._cluster_real_index = []
		for h in range(3):
			intermediate = []
			for i in range(len(np.unique(self._result[h]))):
				intermediate.append([self._index[h][j] for j, g in enumerate(self._result[h]) if g == i])
			self._cluster_real_index.append(intermediate)
		






