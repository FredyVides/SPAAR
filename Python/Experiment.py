"""
Created on Wed Mar 31 02:57:52 2021
NUMERICALEXPERIMENT Numerical experiments using SPAAR technology
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""
def Experiment(experiment_number):
    from pandas import read_csv
    from matplotlib.pyplot import show,subplots
    from numpy import append,array,fliplr,reshape,zeros,real,imag,linspace,cos,sin,pi,count_nonzero
    from numpy.linalg import eig,svd
    from statsmodels.tsa.ar_model import AutoReg
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from spsolver import spsolver
    from LagEstimate import LagEstimate
    from SpAutoRegressor import SpAutoRegressor
    from SPARPredictor import SPARPredictor
    from SpGRUModel import SpGRUModel
    from SpGRUPredictor import SpGRUPredictor
    from SpGRUModelTorch import SpGRUModelTorch
    from SpGRUPredictorTorch import SpGRUPredictorTorch
    from IWSpRepresentation import IWSpRepresentation
    
    
    if experiment_number <= 1:
        tol = 5e-4
        delta = 5e-3
        sampling_proportion = 17/100
        url = "https://raw.githubusercontent.com/FredyVides/SPAAR/main/DataSets/NoisyNLSystem1Data.csv"
        data = read_csv(url,header=None).values[:,0]
        Lag = LagEstimate(data,1)
        nz = Lag
        Lag_AR = Lag
        L0 = Lag+int(0.17*nz)
        L0_AR = 2017
        M0 = int((9/50)*len(data))
        T = 576
    else:
        tol = 5e-3
        delta = 3e-2
        sampling_proportion = 19/100
        url = "https://raw.githubusercontent.com/FredyVides/SPAAR/main/DataSets/NoisyNLSystem2Data.csv"
        data = read_csv(url,header=None).values[:,0]
        Lag = 3*LagEstimate(data.copy(),10)
        nz = Lag
        Lag_AR = Lag
        L0 = Lag+int(0.02*nz)
        L0_AR = 2*Lag_AR+86
        M0 = int((1/5)*len(data))
        T = 295
        
    nn = 16
    ep = 200
    md = data.min()
    Md = abs(data - md).max()
    steps = len(data)-Lag
    steps_AR = len(data)-1
    spp = 5e-6
    
    x = data.copy()
    xs = 2*(x-md)/Md-1
    
    # Compute Models
    
    # Using standard autoregressor
    ss=1
    Xs = xs.copy()
    Xs = Xs[:len(Xs):ss]
    train = Xs[:L0_AR]
    model = AutoReg(train, lags=Lag_AR, old_names=False, trend = 'n')
    model_fit = model.fit()
    B = fliplr(reshape(model_fit.params,(1,Lag_AR)))
    # Using sparse autoregressor
    A,h_0 = SpAutoRegressor(xs,1/len(xs),sampling_proportion,1,Lag,tol,delta,nz)
    # Using TensorFlow GRU RNN
    TS_Model,H,h = SpGRUModel(data.copy(),Lag,sampling_proportion,nn,ep,spp)
    # Using PyTorch RNN
    TS_Model_1,H_1,h_1 = SpGRUModelTorch(data.copy(),Lag,sampling_proportion,nn,ep,spp)
    
    # Compute sparse refinement of the input weights for the TensorFlow GRU models
    TS_Model = IWSpRepresentation(H,TS_Model,tol,1e2*spp)
    
    # Compute predictions
    # With the sparse autoregressor
    y = Md*(SPARPredictor(A,h_0,steps)+1)/2+md
    # With the standard autoregressor
    predictions = model_fit.predict(Lag_AR, end=steps_AR, dynamic=False)
    predictions = Md*(predictions+1)/2+md
    predictions = append(data[:Lag_AR],predictions)
    # With the TensorFlow GRU model
    X = SpGRUPredictor(data.copy(),TS_Model,h.copy(),steps)
    # With the PyTorch GRU model
    X1 = SpGRUPredictorTorch(data.copy(),TS_Model_1,h_1.copy(),steps)
    
    # Extract approximately periodic component from the approximate spectra
    threshold = 8*delta/Md
    C0 = zeros((Lag,Lag))
    C1 = zeros((Lag_AR,Lag_AR))
    for k in range(Lag-1):
        C0[k,k+1]=1
    for k in range(Lag_AR-1):
        C1[k,k+1]=1
        
    C0[-1,:] = A
    C1[-1,:] = B
    
    t = linspace(0,1,1001)
    xt = cos(2*pi*t)
    yt = sin(2*pi*t)
    
    if experiment_number > 1:
        h01 = xs[Lag:2*Lag].copy()
        h02 = xs[Lag_AR:2*Lag_AR].copy()
    else:
        h01 = xs[:Lag].copy()
        h02 = xs[:Lag_AR].copy()
    
    h1 = []
    h2 = []
    
    for k in range(T):
        h1.append(h01)
        h2.append(h02)
        h01 = C0@h01
        h02 = C1@h02
        
    h01 = array(h1).T
    h02 = array(h2).T
    
    u1,s1,_ = svd(h01,full_matrices=False)
    u2,s2,_ = svd(h02,full_matrices=False)
    
    r1 = sum(s1>threshold)
    r2 = sum(s2>threshold)
    
    u1 = u1[:,:r1]
    u2 = u2[:,:r2]
    
    C00 = u1.conj().T@C0@u1
    C01 = u2.conj().T@C1@u2
    D00 = C00
    D01 = C01
    
    l0,_ = eig(C00)
    l1,_ = eig(C01)
    
    for k in range(T-1):
        D00 = D00@C00
        D01 = D01@C01
        
    l00,_ = eig(D00)
    l01,_ = eig(D01)
    
    # Compute mixed model and plot the results summary
    
    A0 = zeros((3,M0))
    A0[0,:] = y[:M0]
    A0[1,:] = X[:M0]
    A0[2,:] = X1[:M0]
    d = zeros((M0,1))
    d[:,0] = data[:M0]
    W0 = spsolver(A0.T,d,100,"svd",3,1e-3,5e-5)
    print("Mixing coefficients: ",W0)
    
    z = W0[0]*y+W0[1]*X+W0[2]*X1
    fig_0,axs_0 = subplots(2,1)
    axs_0[0].plot(data),axs_0[0].plot(z,'--')
    axs_0[0].set_xlabel('t')
    axs_0[0].set_ylabel('Signal')
    axs_0[0].legend(['reference','identification'],loc = 'lower center')
    axs_0[0].grid(True)
    axs_0[1].plot(data),axs_0[1].plot(predictions,'--')
    axs_0[1].set_xlabel('t')
    axs_0[1].set_ylabel('Signal')
    axs_0[1].legend(['reference','identification'],loc = 'lower center')
    axs_0[1].grid(True)
    show()
    #fig_0.savefig('fig_results_summary_0_'+str(experiment_number)+'.png',dpi=600,format='png')
    tp1 = linspace(1, Lag, Lag)
    tp2 = linspace(1, Lag_AR, Lag_AR)
    fig_0,axs_0 = subplots(2,1)
    axs_0[0].stem(tp1,fliplr(reshape(A[0,:],(1,Lag)))[0,:])
    axs_0[0].set_xlabel('$k$')
    axs_0[0].set_ylabel('$c_{k}$')
    axs_0[0].grid(True)
    axs_0[1].stem(tp2,fliplr(reshape(B[0,:],(1,Lag_AR)))[0,:])
    axs_0[1].set_xlabel('$k$')
    axs_0[1].set_ylabel('$c_{k}$')
    axs_0[1].grid(True)
    show()
    #fig_0.savefig('fig_results_summary_1_'+str(experiment_number)+'.png',dpi=600,format='png')
    print("nnz(A): ",count_nonzero(A))
    print("nnz(B): ",count_nonzero(B[0,:]))
    print("Sample size for SpARS: ",L0)
    print("Sample size for standard AR: ",L0_AR)
    
    fig_1,axs_1 = subplots(2,2)
    axs_1[0,0].plot(xt,yt,'b')
    axs_1[0,0].plot(real(l0),imag(l0),'r.')
    axs_1[0,0].grid(True)
    axs_1[0,0].axis('square')
    axs_1[0,1].plot(xt,yt,'b')
    axs_1[0,1].plot(real(l1),imag(l1),'r.')
    axs_1[0,1].grid(True)
    axs_1[0,1].axis('square')
    axs_1[1,0].plot(xt,yt,'b')
    axs_1[1,0].plot(real(l00),imag(l00),'r.')
    axs_1[1,0].grid(True)
    axs_1[1,0].axis('square')
    axs_1[1,1].plot(xt,yt,'b')
    axs_1[1,1].plot(real(l01),imag(l01),'r.')
    axs_1[1,1].grid(True)
    axs_1[1,1].axis('square')
    show()
    #fig_1.savefig('fig_results_summary_2_'+str(experiment_number)+'.png',dpi=600,format='png')
    
    rmse_0 = sqrt(mean_squared_error(data[Lag:], z[Lag:len(data)]))
    rmse_1 = sqrt(mean_squared_error(data[Lag:], predictions[Lag:len(data)]))
    print('SpARS RMSE: %.10f' %rmse_0)
    print('AR RMSE: %.10f' %rmse_1)