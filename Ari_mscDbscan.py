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
    #------Adjusted Rand Index------------------
    sigma = 80
    m = 50 # the dimension of each mode
    k = 2  # the number of cluster in each mode
    # -------Real clusters----------------
    real = [0 for i in range(10)] + [1 for i in range(10)] + [2 for _ in range(m-20)]
    # -----------------------
    ARI_mean, ARI_std = [], []
    for sigma in range(50,101,5):
        ari = []
        for _ in range(10):
            # generate data
            D = gdata.Data_generator(m,m,m,k1=10,k2=10,k3=10, cluster=k, sigma=sigma) 
            data1 = D.multiple_cluster()
            # -------------Set the value of epsilon-------
            e_ = 0.001
            res = msc_multiple.Msc(data1, norm="normalized", e_ = e_)
    
            msc_output = res.get_result_triclustering()
            sim_matrices = res.get_cij()
            # MSC-Extension
            indices = []
            for j in range(3):
                intermediate = []
                for i in range(len(msc_output[0])):
                    intermediate = intermediate  + msc_output[j][i] 
                indices.append(list(set(intermediate)))
        
            dim = (m,m,m)  # data1.shape
            MscExtension = multiple_dbscan.Msc_extension(dim, sim_matrices, indices, e_, minP=2)
            cluster = MscExtension._result[0].tolist()
            cluster = cluster + [2 for _ in range(m-len(cluster))]
            ari.append(adjusted_rand_score(real, cluster))
        ARI_mean.append( np.mean(ari)  )
        ARI_std.append( np.std(ari) )

    return [ ARI_mean, ARI_std ]



def ari_plot(ARI_mean, ARI_std ):
    x = [i for i in range(50,101,5)]
    plt.errorbar(x, ARI_mean, ARI_std, linestyle='None', marker='')
    plt.plot(x, ARI_mean, label="ARI")
    plt.xlabel("gamma")
    plt.ylabel("ARI")
    plt.legend(loc="lower right")
    #plt.savefig('./image/ARI_Extension.png')
    plt.show()