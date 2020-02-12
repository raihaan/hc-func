import numpy as np
from sklearn.covariance import GraphicalLassoCV

def get_fc_mx(tseries, method='corr'):
    #Return a connectivity matrix, using one of three methods
    #Expects tseries to have dimensions regions x time

 
    if method=='corr':
        fc = np.corrcoef(tseries)
    elif method=='cov':
        fc = np.cov(tseries)
    elif method=='precision':
        #the GraphicalLasso fcn wants time x regions dimensions
        estimator = GraphicalLassoCV()
        estimator.fit(np.transpose(tseries))
        fc = estimator.covariance_
    return fc