# Packages
import generate_3_D_tensor as gdata
import numpy as np
import msc_multiple as msc_multiple
from sklearn.cluster import DBSCAN
import itertools
import fonctions as f
import matplotlib.pyplot as plt
import multiple_dbscan as multiple_dbscan
from sklearn.metrics.cluster import adjusted_rand_score


def run():
    sigma = 80
    m = 50 # the dimension of each mode
    k = 2  # the number of cluster in each mode

    # generate data
    D = gdata.Data_generator(m,m,m,k1=10,k2=10,k3=10, cluster=k, sigma=sigma) 
    data1 = D.multiple_cluster()
    # -------------Set the value of epsilon-------
    e_ = 0.001
    res = msc_multiple.Msc(data1, norm="normalized", e_ = e_)
    
    msc_output = res.get_result_triclustering()
    
    sim_matrices = res.get_cij()
    # separation of indices
    output_dbscan = []
    for h in range(3): # the cluster in the three mode
        # recover the similarity matrix
        data = sim_matrices[h].T
        data = data[msc_output[h][0],:]
        epsilon = (len(data[:,0])*e_ / 2 + np.sqrt(np.log(m - len(data[:,0]))))**0.5
        clustering = DBSCAN(eps=epsilon, min_samples=2).fit(np.abs(data))
        cluster = clustering.labels_
        cluster_uniq = np.unique(cluster)
        #print(len(cluster_uniq))
        
        indices = []
        for i in cluster_uniq:
            intermediate = [msc_output[h][0][g] for g, j in enumerate(cluster) if j == i]
            indices.append(intermediate)
        output_dbscan.append(indices)
        
    # MSC
    rmse_MSC = f.rmse(data1, msc_output[0][0], msc_output[1][0], msc_output[2][0]) 
    # MSC-Extension
    mse_E = []
    for j in itertools.product(output_dbscan[0], output_dbscan[1], output_dbscan[2]):
        mse_E.append(f.rmse(data1, j[0], j[1], j[2]))
    rmse_Extension = np.mean(mse_E)

    return [rmse_MSC, rmse_Extension] #  rmse_

def rmse_boxplot(rmse_MSC, rmse_Dbscan):
    rmse = [rmse_MSC, rmse_Dbscan]
    plt.boxplot(rmse,patch_artist=True,labels=['MSC','MSC-DBSCAN'])
    plt.ylabel("Root Mean square Error")
    #plt.savefig('./image/msc_and_extension.png')
    plt.show()




        

