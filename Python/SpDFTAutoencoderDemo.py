#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:35:15 2021

@author: doctor
"""

from scipy.io.wavfile import read
import numpy as np
from scipy.fft import fft,ifft
import matplotlib.pyplot as plt
import pandas as pd
from SpDFTEncoder import SpDFTEncoder
from SpDFTDecoder import SpDFTDecoder

#rate, data = read('ReferenceSensorSignal.wav')
#rate, datab = read('SensorSignalWithAnomaly.wav')
data = pd.read_csv('../DataSets/signal_with_anomaly.csv', usecols=[1], engine='python')


threshold=.05
S=2000
L=250

y0 = data.values
rdata = y0[:S,0]
tdata = y0[(len(y0)-S):,0]
m0=np.abs(rdata).max()
y0=rdata/m0
w0=tdata/m0

Jy,y = SpDFTEncoder(y0,S,L,threshold,threshold)
Jw,w = SpDFTEncoder(w0,S,L,threshold,threshold)
yr = SpDFTDecoder(Jy,y)
wr = SpDFTDecoder(Jy,w)
    
plt.plot(y0),plt.plot(yr,'r')
plt.show()
plt.plot(w0),plt.plot(wr,'r')
plt.show()

plt.plot(abs(y0[:len(wr)]-yr))
plt.show()
plt.plot(abs(w0[:len(wr)]-wr))
