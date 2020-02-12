import numpy as np

def get_fc_mx(tseries, method='corr'):
    #Return a connectivity matrix, using one of three methods
    #Expects tseries to have dimensions regions x time
    fc=np.corrcoef(tseries)
    return fc