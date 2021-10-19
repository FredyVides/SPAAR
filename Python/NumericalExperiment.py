"""
Created on Wed Mar 31 02:57:52 2021
NUMERICALEXPERIMENT Numerical experiments using SPAAR technology
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""
def NumericalExperiment(experiment_number):
    from pandas import read_csv
    from matplotlib.pyplot import plot,stem,show,subplots,legend,semilogy
    from numpy import append,array,ceil,fliplr,roots,reshape,zeros,real,imag,linspace,cos,sin,pi,inf,where,identity,ceil,count_nonzero
    from numpy.linalg import norm,eig,svd
    from statsmodels.tsa.ar_model import AutoReg
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from lsspsolver import lsspsolver
    from LagEstimate import LagEstimate
    from SpAutoRegressor import SpAutoRegressor
    from SPARPredictor import SPARPredictor
    from SpGRUModel import SpGRUModel
    from SpGRUPredictor import SpGRUPredictor
    from SpGRUModelTorch import SpGRUModelTorch
    from SpGRUPredictorTorch import SpGRUPredictorTorch
    from IWSpRepresentation import IWSpRepresentation
    
    delta = 1e-1
    tol = 1e-1
    sampling_proportion = 0.5
    if experiment_number <= 1:
        tol = 4e-2
        url = "https://raw.githubusercontent.com/FredyVides/SPAAR/main/DataSets/VortexSheddingSignal.csv"
        data = read_csv(url)
        data = data.values[:,0]
        Lag = LagEstimate(data.copy(),215)
        L0 = int(ceil(sampling_proportion*len(data)))
        L0_AR = int(ceil(0.9*len(data)))
        T = 16
    elif experiment_number == 2:
        url = "https://raw.githubusercontent.com/FredyVides/SPAAR/main/DataSets/NLOscillatorSignal.csv"
        tol = 9e-3
        data = read_csv(url)
        data = data.values[1,:]
        Lag = LagEstimate(data.copy(),10)
        L0 = int(ceil(sampling_proportion*len(data)))
        L0_AR = L0
        T = Lag
    elif experiment_number == 3.1:
        url = "https://raw.githubusercontent.com/numenta/NAB/master/data/artificialNoAnomaly/art_daily_no_noise.csv"
        data = read_csv(url, usecols=[1])
        data = data.values[:,0]
        Lag = LagEstimate(data.copy(),10)
        L0 = int(ceil(sampling_proportion*len(data)))
        L0_AR = L0
        T = Lag
    elif experiment_number >= 3.2:
        url = "https://raw.githubusercontent.com/numenta/NAB/master/data/artificialNoAnomaly/art_daily_small_noise.csv"
        data = read_csv(url, usecols=[1])
        data = data.values[:,0]
        Lag = LagEstimate(data.copy(),10)
        L0 = int(ceil(sampling_proportion*len(data)))
        L0_AR = L0
        T = Lag
        
    nn = 16
    ep = 60
    md = data.min()
    Md = abs(data - md).max()
    steps = len(data)-Lag
    steps_AR = len(data)-1
    spp = 1e-5
    
    x = data.copy()
    xs = 2*(x-md)/Md-1
    
    # Compute Models
    
    # Using standard autoregressor
    ss=1
    Xs = xs.copy()
    Xs = Xs[:len(Xs):ss]
    train, test = Xs[:L0_AR], Xs[L0_AR:]
    model = AutoReg(train, lags=Lag, old_names=False)
    model_fit = model.fit()
    B = fliplr(reshape(model_fit.params[1:],(1,Lag)))
    # Using sparse autoregressor
    A,h_0 = SpAutoRegressor(xs,1/len(xs),sampling_proportion,1,Lag,delta,tol)
    # Using TensorFlow GRU RNN
    TS_Model,H,h = SpGRUModel(data.copy(),Lag,sampling_proportion,nn,ep,spp)
    # Using PyTorch RNN
    TS_Model_1,H_1,h_1 = SpGRUModelTorch(data.copy(),Lag,sampling_proportion,nn,ep,spp)
    
    # Compute sparse refinement of the input weights for the TensorFlow GRU models
    TS_Model = IWSpRepresentation(H,TS_Model,delta,5e-3)
    
    # Compute predictions
    # With the sparse autoregressor
    y = Md*(SPARPredictor(A,h_0,steps)+1)/2+md
    # With the standard autoregressor
    predictions = model_fit.predict(Lag, end=steps_AR, dynamic=False)
    predictions = Md*(predictions+1)/2+md
    predictions = append(data[:Lag],predictions)
    # With the TensorFlow GRU model
    X = SpGRUPredictor(data.copy(),TS_Model,h.copy(),steps)
    # With the PyTorch GRU model
    X1 = SpGRUPredictorTorch(data.copy(),TS_Model_1,h_1.copy(),steps)
    
    # Extract approximately periodic component from the approximate spectra
    threshold = 2*delta/Md
    C0 = zeros((Lag,Lag))
    for k in range(Lag-1):
        C0[k,k+1]=1
        
    C1 = C0.copy()
    C0[-1,:] = A
    C1[-1,:] = B
    
    t = linspace(0,1,1001)
    xt = cos(2*pi*t)
    yt = sin(2*pi*t)
    
    if experiment_number == 1:
        h01 = xs[Lag:2*Lag].copy()
        h02 = h01.copy()
    elif experiment_number != 1:
        h01 = xs[:Lag].copy()
        h02 = h01.copy()
    
    h1 = []
    h2 = []
    
    for k in range(Lag):
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
    
    A0 = zeros((3,L0))
    A0[0,:] = y[:L0]
    A0[1,:] = X[:L0]
    A0[2,:] = X1[:L0]
    W0 = lsspsolver(A0.T,data[:L0],100,1e-2,1e-2)
    print("Mixing coefficients: ",W0)
    
    z = W0[0]*y+W0[1]*X+W0[2]*X1
    fig_0,axs_0 = subplots(2,2)
    axs_0[0,0].plot(data),axs_0[0,0].plot(z,'.-')
    axs_0[0,0].legend(['reference','identification'],loc = 'lower left')
    axs_0[0,1].semilogy(abs(data[Lag:]-z[Lag:len(data)]),'b.-')
    axs_0[0,1].semilogy(abs(predictions[Lag:len(data)]-data[Lag:]),'r.-')
    axs_0[0,1].grid(True)
    axs_0[0,1].legend(['SpARS','AR'])
    axs_0[1,0].stem(A)
    axs_0[1,1].stem(B[0,:])
    print("nnz(A): ",count_nonzero(A))
    print("nnz(B): ",count_nonzero(B[0,:]))
    
    fig_1,axs_1 = subplots(2,2)
    axs_1[0,0].plot(xt,yt,'b')
    axs_1[0,0].plot(real(l0),imag(l0),'r.')
    axs_1[0,0].axis('equal')
    axs_1[0,1].plot(xt,yt,'b')
    axs_1[0,1].plot(real(l1),imag(l1),'r.')
    axs_1[0,1].axis('equal')
    axs_1[1,0].plot(xt,yt,'b')
    axs_1[1,0].plot(real(l00),imag(l00),'r.')
    axs_1[1,0].axis('equal')
    axs_1[1,1].plot(xt,yt,'b')
    axs_1[1,1].plot(real(l01),imag(l01),'r.')
    axs_1[1,1].axis('equal')
    show()
    
    #fig_0.savefig('fig_results_summary_0_'+str(experiment_number)+'.eps',dpi=600,format='eps')
    #fig_0.savefig('fig_results_summary_0_'+str(experiment_number)+'.png',dpi=600,format='png')
    #fig_1.savefig('fig_results_summary_1_'+str(experiment_number)+'.eps',dpi=600,format='eps')
    #fig_1.savefig('fig_results_summary_1_'+str(experiment_number)+'.png',dpi=600,format='png')
    rmse_0 = sqrt(mean_squared_error(data[Lag:], z[Lag:len(data)]))
    rmse_1 = sqrt(mean_squared_error(data[Lag:], predictions[Lag:len(data)]))
    print('SpARS RMSE: %.10f' %rmse_0)
    print('AR RMSE: %.10f' %rmse_1)