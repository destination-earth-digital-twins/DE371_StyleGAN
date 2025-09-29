import numpy as np
from scipy.stats import multivariate_normal
from tqdm import trange

def PCA(X , num_components=None):

    if len(X.shape) != 2 :
        raise NotImplementedError
    
    if num_components is None:
        num_components=np.min(X.shape)
     
    # Mean of data
    X_meaned = X - np.mean(X , axis = 0)
     
    # Covariance Matrix
    cov_mat = np.cov(X_meaned , rowvar = False) 
    # intercor_mat = np.correlate(X_meaned.flatten(),X_meaned.flatten())
     
    # Eigen Values and Eigen Vectors
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    # Selecting only the principal eigenvectors (that have the highest eigenvalues)
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    # Projecting our initial data along the principal components axis
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced, sorted_eigenvalue, sorted_eigenvectors



def log_likelihood(evals):
    ''' Extracted from https://github.com/probml/pyprobml/blob/master/notebooks/book1/20/pcaOverfitDemo.ipynb'''

    Lmax = len(evals)
    ll = np.arange(0.0, Lmax)

    for L in trange(Lmax):

        group1 = evals[0 : L + 1]  # Divide Eigenvalues in two groups
        group2 = evals[L + 1 : Lmax]

        mu1 = np.mean(group1)
        mu2 = np.mean(group2)

        # eqn (20.30)
        sigma = (np.sum((group1 - mu1) ** 2) + np.sum((group2 - mu2) ** 2)) / Lmax

        ll_group1 = np.sum(multivariate_normal.logpdf(group1, mu1, sigma))
        ll_group2 = np.sum(multivariate_normal.logpdf(group2, mu2, sigma))

        ll[L] = ll_group1 + ll_group2  # eqn (20.31)

    return ll
