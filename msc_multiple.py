from sklearn import preprocessing
import scipy.linalg as la
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.decomposition import PCA
import fonctions as f


class Msc():
    def __init__(self, tensor, norm="", e_= 0.0, method="", sigmax = 1):
        # tensor :  the dataset
        # norm :  normalization of the column of each slice
        # e_ :  the parameter epsilon
        # sigmax : 
        self._tensor = tensor
        self._norm = norm
        self._e_ = e_
        self._method = method
        self._dim = self._tensor.shape
        self._sigmax = sigmax
        self._result_triclustering, self._cij, self._similarity_triclustering = self.tricluster_method()


    def normed_and_covariance(self, M):
        if self._norm == "centralized":
            m = np.mean(M, axis=0).reshape(1,-1)  # mean of each column
            M = M - m
            M = (1/len(M[0,:])) * ((M.T).dot(M))
        elif self._norm == "normalized":
            M = normalize(M, axis=0)
            M = (M.T).dot(M)
        else:
            M = (M.T).dot(M)
        return M


    def tricluster_method(self):
        l = len(self._dim)   # l = 3
        # the eigenvalue and eigenvector of each slice for each dimension
        for i in range(l):
            if i == 0:
                e0 = []
                for k in range(self._dim[0]):
                    frontal =  self.normed_and_covariance(self._tensor[k,:,:])
                    if (self._method == "rayleigh"):
                        w, v = f.rayleigh_power_iteration(frontal, 10)
                        e0.append([w.real, v.real])
                    else :
                        w, v = la.eig(frontal)
                        a = np.argmax(w.real)
                        e0.append([w[a].real, v[:,a].real])
                    
            elif i == 1:
                e1 = []
                for k in range(self._dim[1]):
                    horizontale = self.normed_and_covariance(self._tensor[:,k,:])
                    if (self._method == "rayleigh"):
                        w, v = f.rayleigh_power_iteration(horizontale, 10)
                        e1.append([w.real, v.real])
                    else :
                        w, v = la.eig(horizontale)
                        a = np.argmax(w.real)
                        e1.append([w[a].real, v[:,a].real])
                    
                    
            elif i==2:
                e2 = []
                for k in range(self._dim[2]):
                    laterale = self.normed_and_covariance(self._tensor[:,:,k])
                    if (self._method == "rayleigh"):
                        w, v = f.rayleigh_power_iteration(laterale, 10)
                        e2.append([w.real, v.real])
                    else :
                        w, v = la.eig(laterale)
                        a = np.argmax(w.real)
                        e2.append([w[a].real, v[:,a].real])


        e = [e0, e1, e2]

        j = 0

        Result = []
        Similarity, cij = [], []
        counteur = 0  # conteur
        for i in e:
            lam = [k[0] for k in i]   # The eigenvalues
            result, similarity = [], []
            m =  len(i)  # length of element in a fix dimension
            for g in range(1):     # number of desired cluster ( especially for iterative method)
           
                M = np.zeros((len(i[0][1]), len(i)))  

                for k in range(len(i)):
                    M[:,k] = (i[k][0] * i[k][1] ) / np.max(lam)  # product of eigenvalue and eigenvector of slice k
                
                M = (M.T).dot(M)     # covariance matrix ,  symmetric matrix
                Matrices = np.abs( M.copy() )
            
                M = np.sum(np.abs(M), axis = 0)  # sum of each line of M
            
                M_order = M.copy()
                M_order.sort()
            
                gap = self.initialize_c(M_order) 
            
                thm1 = [k for k, l in enumerate(M) if l >= gap]   # M is the original vector
                thm = thm1.copy()
                value  = [M[k] for k in thm1]    # The value for all indices in thm1 
                c = len(thm1)   

                if (self._e_ == 0.0 ) :


                    self._e_ = 1 / ((len(M) - c)**2)

                    #if ( self._e_**0.5 <= 1/(len(M)-c)) :
                    while ((self.max_difference(value) > (c * (self._e_ / 2)) + (np.log(len(M)-c)**0.5)) ):
                        thm = [k for k, l in enumerate(M) if l > min(value)]
                        value.remove(min(value))
                        c -= 1
                        if (value == list()):
                            thm = thm1
                            break
                        self._e_ = 1 / ((len(M) - c)**2)   # update the value of epsilon
                    print ("epsilon - ",counteur," = ", self._e_)
                    self._e_ = 0.0

                elif (self._e_ < 1 / ((m - c)**2)) :   # with a fix value of epsilon
                    while ((self.max_difference(value) > (c * (self._e_ / 2)) + (np.log(m-c)**0.5)) ):
                        thm = [k for k, l in enumerate(M) if l > min(value)]
                        value.remove(min(value))
                        c -= 1
                        if (value == list()):
                            thm = thm1
                            break
                counteur += 1

                # mean of the similarity
                if g == 0 :
                    cij.append(Matrices)
                similarity.append(np.mean(Matrices[thm,:][:,thm]))

                # result is a list with three elements and its elements have type list
                result.append(thm)
                m -= len(thm)     # adjusted the length of slices taken into account
 
            # save the cluster in each mode
            Result.append(result)
            Similarity.append(similarity)
            j += 1
  



        return Result, cij, Similarity

    def find_cprime(self, a, b):
        if ((a**0.5) * b < np.sqrt(b )) :
            return int((a**0.5) * b)
        else:
            return b + 1

    def get_result_triclustering(self):
        return self._result_triclustering
    def get_similarity_triclustering(self):     # V^tV
        return self._similarity_triclustering
    def get_cij(self):
        return self._cij


    def initialize_c(self, Liste):   # Listte is the vector d in assending order
        i = 1
        c = abs(Liste[0] - Liste[1])
        for j in range(1, len(Liste)-1):
            if c < abs(Liste[j] - Liste[j+1]) :
                c = abs(Liste[j] - Liste[j+1])
                i = j

        return Liste[i]


    def max_difference(self, valeur):
        c = []
        for i in range(len(valeur)):
            for j in range(i, len(valeur)):
                c.append(np.abs(valeur[i]-valeur[j]))
        return max(c)


    def borneSuppWhishart(self, m, n):
        mu = ( (m-1)**0.5 + n**0.5 )**2
        sigma = (mu**0.5) * ( ( 1/(m**0.5) ) + ( 1/(n**0.5) ) )**(1/3)
        x0 = -9.8209
        return self._sigmax**2 *( mu + x0 * sigma)




        

