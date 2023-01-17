from cwt import ScalogramCWT
import os
import scipy
import numpy as np

filename = "A0033_lead1_seg1"
signal = scipy.io.loadmat(filename)

signal = signal["ECG_segment"]
print(signal.shape)

# check shape valid
# t = np.linspace(-5, 5, 10*100)
# x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) + 0.1*np.sin(2*np.pi*1.25*t + 1) + 0.18*np.cos(2*np.pi*3.85*t))
# xn = x + np.random.randn(len(t)) * 0.5
# print(xn.shape)
# print(t.shape)
# XW,S = ScalogramCWT(xn,t,fs=100,wType='Gauss',PlotPSD=True)
# print(type(XW))
# print(XW.shape)
# print(type(S))
# print(S.shape)

# real
signal = signal.reshape(-1)
print(signal.shape)
t = np.linspace(-5, 5, 1600)
XW,S = ScalogramCWT(signal,t,fs=100,wType='Gauss',PlotPSD=True)