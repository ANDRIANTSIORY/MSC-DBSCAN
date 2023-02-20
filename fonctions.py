from sklearn.metrics.cluster import adjusted_rand_score
import scipy.linalg as la
import numpy as np

# remove liste of slice in fixed dimension
def rem(donnee, dim, liste):
    while (liste != []):
        dimT = list(donnee.shape)
        if dim ==1:
            dimT[1] = dimT[1] - 1
            
            donneeI = np.zeros(dimT)
            for i in range(dimT[1]):
                if (i >= liste[-1]):
                    donneeI[:,1,:] = donnee[:,i+1,:]
                else:
                    donneeI[:,i,:] = donnee[:,i,:]
            donnee = donneeI  
            liste = liste[:-1]
        elif dim == 2:
            dimT[2] = dimT[2] - 1
            
            donneeI = np.zeros((dimT))
            for i in range(dimT[2]):
                if (i >= liste[-1]):
                    donneeI[:,:,i] = donnee[:,:,i+1]
                else:
                    donneeI[:,:,i] = donnee[:,:,i]
            donnee = donneeI  
            liste = liste[:-1]
            
        elif dim == 0:
            dimT[0] = dimT[0] - 1
            
            donneeI = np.zeros((dimT))
            for i in range(dimT[0]):
                if (i >= liste[-1]):
                    donneeI[i,:,:] = donnee[i+1,:,:]
                else:
                    donneeI[i,:,:] = donnee[i,:,:]
            donnee = donneeI  
            liste = liste[:-1]
                
    return donnee


# recovery rate
# I is the list the true cluster
# J is the list of estimated cluster
def recovery_rate(I, J):
    r_rate = 0
    for i in range(3):
        r  = set(I[i]).intersection(set(J[i]))
        r_rate += (len(r) / (3 * len(J[0])))
    return r_rate



def gound_truth_known_tensor_biclustering(true, estimation):  # true is a couple and the estimation as well
    # recovery rate
    r = len(set(true[1]).intersection(set(estimation[1]))) / (2*len(true[1]))
    r += len(set(true[0]).intersection(set(estimation[0]))) / (2*len(true[0]))
    #rint("recovery rate : ", r)
    return r


def find_adjusted_rand_score(vrai, estimation):
    result = 0
    for i in range(len(vrai)):
        result += adjusted_rand_score(vrai[i], estimation[i])
    return result/len(vrai)


# find all the top eigenvalue the covariance matrix of each slice
def top_eigenvalue(tensor, dimension = 0):
    dim = tensor.shape[dimension]   # the dimension of the data in the choosen dimension
    e = []
    if dimension == 0:
        for k in range(dim):
            frontal =  tensor[k,:,:].T.dot(tensor[k,:,:])
            w, v = la.eig(frontal)
            p = np.argmax(w.real)  # the index of the maximum eigenvalue
            e.append(w[p].real)
                    
    elif dimension == 1:
        for k in range(dim):
            horizontale = tensor[:,k,:].T.dot(tensor[:,k,:])
            w, v = la.eig(horizontale)
            p = np.argmax(w.real)  # the index of the maximum eigenvalue
            e.append(w[p].real)
                    
    elif dimension ==2:
        for k in range(dim):
            laterale = tensor[:,:,k].T.dot(tensor[:,:,k])
            w, v = la.eig(laterale)
            p = np.argmax(w.real)  # the index of the maximum eigenvalue
            e.append(w[p].real,)
    return e

def rayleigh_power_iteration(A, num_simulations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    mu = np.dot(np.dot(b_k.T, A),b_k)

    return  mu, b_k



# Root means square error  RMSE = sqrt( sum_i(a_i - â_i)^2 )
# between the estimated clustering and the real data 
def rmse(data,a, b, c):
    resultat = data[a,:,:][:,b,:][:,:,c]
    return np.sqrt(np.sum((resultat - np.mean(resultat))*(resultat - np.mean(resultat))))
