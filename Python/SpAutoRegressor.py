#!/usr/bin/env python3
"""
Created on Wed Mar 31 02:57:52 2021
SPAUTOREGRESSOR  Sparse autoregressor for time series modeling
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""
def SpAutoRegressor(x,ssp,sp,pp,L0,tol,delta,nz = 100):
    from numpy import ceil,floor,max,min
    from scipy.linalg import hankel
    from spsolver import spsolver
    sl = len(x)
    ssp=int(ceil(sl*ssp))
    x = x[0:sl:ssp]
    sl = len(x)
    sp = int(ceil(sp*sl))
    xt = x[:sp]
    pp = max([min([floor(pp*sp),sl-sp]),L0])
    L=L0
    H=hankel(xt[:L],xt[(L-1):])
    H0=H[:,:-1]
    H1=H[L-1:,1:]
    A = spsolver(H0.T,H1.T,L,"svd",nz,tol,delta)
    return A.T,H0[:,0]   