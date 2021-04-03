#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:50:28 2020

@author: daniel
"""
import numpy as np
import pandas as pd
from scipy import signal
# Find time to steady state when eigenvalues are within +/- delta% of final value

def getMaxLength(arr, n): 
  
    # intitialize count 
    count = 0 
      
    # initialize max 
    result = 0 
  
    for i in range(0, arr.size): 
      
        # Reset count when 0 is found 
        if (arr[i] == 0): 
            count = 0
  
        # If 1 is found, increment count 
        # and update result if count  
        # becomes more. 
        else: 
              
            # increase count 
            count+= 1 
            result = max(result, count)
        
        if result==n:
            break
        
            
          
    return i-n+1  
def SteadyStateRM(t,y,tolperc=0.1,strainwindow=0.5):
    
    
    ypanda  = pd.Series(y.real,index=t)
    window = int(np.round(strainwindow*t.size/t[-1]))
    #Smooth
    yroll = ypanda.rolling(window).mean()
    
    
    delta = np.abs(yroll-y[-1])/(y[-1]-y[0])
    
   
    tol = tolperc/100
    
    sstime = delta[delta.lt(tol)].index[0]
    # Find first point which has 3 consectutive non-zero values
    
    # Return time at ind
    return sstime
def SteadyStateRM(t,y,tolperc=0.1,strainwindow=0.5):
    
    
    ypanda  = pd.Series(y.real,index=t)
    window = int(np.round(strainwindow*t.size/t[-1]))
    #Smooth
    yroll = ypanda.rolling(window).mean()
    
    
    delta = np.abs(yroll-y[-1])/(y[-1]-y[0])
    
   
    tol = tolperc/100
    
    sstime = delta[delta.lt(tol)].index[0]
    # Find first point which has 3 consectutive non-zero values
    
    # Return time at ind
    return sstime

def SteadyStateSG(t,y,tolperc=0.1,strainwindow=3):
    window = int(np.round(strainwindow*t.size/t[-1]))
    if np.mod(window,2)==0:
        window=window+1

    
    filt = signal.savgol_filter(y, window, 2)

    
    filtpanda = pd.Series(filt.real,index=t)
    delta = np.abs(filtpanda-y[-1])/(y[-1]-y[0])
    
   
    tol = tolperc/100
    
    sstime = delta[delta.lt(tol)].index[0]
    # Find first point which has 3 consectutive non-zero values
    
    # Return time at ind
    return sstime

def SteadyStateButter(t,y,tolperc=0.1,freq=0.5):
    
    sos = signal.butter(10,freq,output='sos')
    filt = signal.sosfilt(sos,y-y[0])
    
    filtpanda = pd.Series(filt.real,index=t)
    delta = np.abs(filtpanda-y[-1])/(y[-1]-y[0])
    
   
    tol = tolperc/100
    
    sstime = delta[delta.lt(tol)].index[0]
    # Find first point which has 3 consectutive non-zero values
    
    # Return time at ind
    return sstime

def SteadyStateTime(t,y,tolperc=0.1,strainwindow=0.5):
    
  
    delta = np.abs(y-y[-1])/y[-1]
    
    #Pandas arrays
    peig = pd.Series(delta.real,index=t)
    
    window = int(np.round(strainwindow*t.size/t[-1]))
    

    
    maxvar = peig.rolling(window).mean()
    
    tol = tolperc/100
    
    sstime = maxvar[maxvar.lt(tol)].index[0]
    # Find first point which has 3 consectutive non-zero values
    
    # Return time at ind
    return sstime

def SteadyStateSD(t,y,tolperc=0.1,strainwindow=0.5):
    
    # m, n = A.shape[-2:]
    # if m != n:
    #     A = np.moveaxis(A,-1,0)
    
    # #Eigenvalues
    # eigA,w = np.linalg.eig(A)
    
    # # Largest Eigenvalue
    # largeig = np.amax(eigA,axis=1).real
    
    #Pandas arrays
    peig = pd.Series(y,index=t)
    
    window = int(np.round(strainwindow*t.size/t[-1]))
    
    std = peig.rolling(window).std()
    mean = peig.rolling(window).mean()
    cv = std/mean # coefficient of variation
    
    # std or cv lesss than tol
    tol = tolperc/100
    #sstime = std[std.lt(tol)].index[0]
    sstime = cv[cv.lt(tol)].index[0]
    
    return sstime
    
    
    