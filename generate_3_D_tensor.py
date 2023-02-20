import numpy as np


class Data_generator():
    def __init__(self, n, n1=0, n2=0, k1=1, k2=1, k3=0, cluster = 1, sigma = 1):
        # m, n1 and n2 are the first, the second and the third dimension of the tensor
        # k1, k2 and k3 are the cardinality of the cluter size in the three modes
        # cluster : the number of the cluster in each mode
        # sigma : the weight of the signal 

        self._m = n
        self._n1 = n1
        self._n2 = n2
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._sigma = sigma
        self._cluster = cluster



    def multiple_cluster(self):

        T = np.zeros((self._m, self._n1, self._n2))
        r = 0 # counter

        for s in range(self._cluster):
            v = np.zeros((self._m))
            u = np.zeros((self._n1))
            w= np.zeros((self._n2))
            X = np.zeros((self._m,self._n1,self._n2))

            for i in range(r, r + self._k3):
                v[i] = 1/np.sqrt( self._k3)
                

            for i in range(r, r + self._k1):
                u[i] = 1/np.sqrt( self._k1)

            for i in range(r, r + self._k2):
               w[i] = 1/np.sqrt( self._k2)

            for i in range(self._m):
                for j in range(self._n1):
                    for k in range(self._n2):
                        X[i,j,k] = self._sigma * v[i]*u[j]*w[k]
            T += X
            r += max(self._k1, self._k2, self._k3)
            

        return T + np.random.normal(0, 1,  size= (self._m, self._n1, self._n2))



        