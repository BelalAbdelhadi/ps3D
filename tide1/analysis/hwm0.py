# -*- coding: utf-8 -*-
"""
Created on Sun May  2 18:52:04 2021

@author: han
"""

import numpy as np
import math
import scipy.special as sp
import sys
if 'C:/Users/han/Desktop/hw_pythonfuncs' not in sys.path:
    sys.path.append('C:/Users/han/Desktop/hw_pythonfuncs')
import hwffts as hwfft
import warnings

pi=math.pi

#G is a 2D function defined on x and y. Note: x=0 is in the middle of x for hwfft to work. 
#Output: the zeroth azimuthal mode of G, averaged using the Bessel transform, as explained in the appendix of Wang and Buhler 2021.
#Output doesn't include r=0. Output starts at r=dx and ends at r=Lx/2
def m0_bessel(x,y,G):
    Nx=len(x)
    Ny=len(y)
    if Nx!=Ny:
        print('WARNING: case when Nx is not equal to Ny is not tested yet!')
    Lx=np.max(x)-np.min(x)
    Ly=np.max(y)-np.min(y)
    k=hwfft.k_of_x(x)
    l=hwfft.k_of_x(y)
    L,K = np.meshgrid(l,k)
    Kappa=np.sqrt(K**2+L**2)
    Gt=hwfft.hwfft2(x,y,G)
    if np.mean(np.imag(Gt))/np.mean(np.real(Gt))>10**(-12):
        print('WARNING: Fourier coefficients of input function contain significant imaginary parts, which means the inputs are not even.')

    G_m0=np.zeros((Nx//2-1))
    rr=0
    drr=x[1]-x[0]
    for irr in range(Nx//2-1):
        rr=rr+drr
        for ik in range(Nx):
            for il in range(Ny):
                coef=sp.jv(0,Kappa[ik,il]*rr)
                G_m0[irr]=G_m0[irr]+Gt[ik,il]*coef
    
    G_m0_normed=G_m0/Lx/Ly
    return G_m0_normed

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
    
def m0_savage(x,y,G,r):
    Y,X=np.meshgrid(y,x)
    R=np.sqrt(X**2+Y**2)
    Nr=len(r)
    Gtot=np.zeros(Nr)
    Gnum=np.zeros(Nr)
    for ir in range(Nr):
        ri=r[ir]
    
        mask_r=R<ri
        Gtot[ir]=np.sum(G*mask_r)
        Gnum[ir]=np.sum(mask_r)
    
    npoints=np.diff(Gnum)
    tvalues=np.diff(Gtot)
    
    Gr=np.zeros(len(npoints))
    for ig in np.arange(len(npoints)):
        if npoints[ig] != 0:
            Gr[ig]=tvalues[ig]/npoints[ig]
        elif ig>0:
            Gr[ig]=Gr[ig-1]

    Gr=np.concatenate(([Gnum[0]], Gr), axis=0)
    return Gr
        
        
