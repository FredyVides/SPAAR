"""
Created on Wed Mar 31 02:57:52 2021
INPUTSPREPRESENTATION Sparse representation of input weights of the GRU model
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""
def IWSpRepresentation(H,model,delta,tol):
    from spsolver import spsolver
    from numpy import array
    W = model.layers[0].get_weights()
    X01 = H@W[0]
    U = spsolver(H,X01,W[0].shape[0],"svd",W[0].shape[0],delta,tol)
    U = array(U)
    W[0] = U
    model.layers[0].set_weights(W)
    return model