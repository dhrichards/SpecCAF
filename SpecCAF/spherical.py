#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:21:32 2020

@author: daniel
"""


import numpy as np
from cmath import sqrt
from scipy.special import sph_harm
from numpy.linalg import pinv
from matplotlib import pyplot as plt
#import shtns


class spherical:
    def __init__(self, lmax):
        self.lmax = lmax
        self.nlm = (lmax//2)**2 + lmax +1
        self.l = np.asarray([int(2*np.floor(np.sqrt(i))) for i in range(self.nlm)])
        self.m = np.asarray([int(i-(self.l[i]//2)**2) for i in range(self.nlm)])
        self.ntheta=80
        self.nphi=160


    def idx(self,l,m):
        return ((l*l)//4 + m)
  
    
    def spec_array(self):
        return np.zeros((self.nlm),dtype=complex)
   


    def index(self,f,l,m):
        l=int(l)
        m=int(m)
        if l>self.lmax or abs(m)>l or l<0:
            v=0.
        elif m>=0:
            v=f[self.idx(l,m)]
        else:
            if (m % 2) == 0:#even
                v = f[self.idx(l,-m)].conjugate()
            else:#odd
                v = -1.*f[self.idx(l,-m)].conjugate()
        
        return v

    def set_grid(self):
        theta_vals = (1.*np.pi/self.ntheta)*np.arange(self.ntheta)
        phi_vals = (2.0*np.pi/self.nphi)*np.arange(self.nphi)
        phi, theta = np.meshgrid(phi_vals, theta_vals)
        
        return theta, phi
    
    
    # def spherical2shtns(self,f):
    #     sht=shtns.sht(self.lmax,self.lmax)
    #     sht.set_grid(self.ntheta,self.nphi)
    #     g=sht.spec_array()
    #     for l in range(0,self.lmax+1,2):
    #         for m in range(0,l+1,1):
    #             g[sht.idx(l,m)]=f[self.idx(l,m)]
    #     return g,sht
    
    # def shtns2spherical(self,g,sht):
    #     f=self.spec_array()
    #     for l in range(0,self.lmax+1,2):
    #         for m in range(0,l+1,1):
    #             f[self.idx(l,m)]=g[sht.idx(l,m)]
    #     return f
        
    # def synth2(self,f):
    #     g,sht=self.spherical2shtns(f)
    #     return sht.synth(g)/np.sqrt(4*np.pi)
        
    
    def synth(self,f): # Transfrom from spherical harmonics to grid
        theta,phi=self.set_grid()
        Y = np.zeros_like(theta)
        for l in range(0,self.lmax+1,2):
            for m in range(-l,l+1,1):
                M=abs(m)
                Y = Y + f[self.idx(l,abs(m))]*sph_harm(M,l,phi,theta)
        Y=Y.real/sqrt(4*np.pi)
        return Y
        
    
    def analys(self,fgrid): # Tansfrom from grid to spherical harmonics
        
        theta,phi=self.set_grid()
        
        
        Ymn=np.zeros((theta.flatten('F').size,self.nlm),dtype=complex)
        for l in range(0,self.lmax+1,2):
            for m in range(0,l+1,2):
                Ymn[:,self.idx(l,m)]+= sph_harm(m,l,phi.flatten('F'),theta.flatten('F'))

        
        
        return np.sqrt(4*np.pi)*pinv(Ymn).dot(fgrid.flatten('F'))


    def polefigure(self,f):
        theta,phi=self.set_grid()
        theta_vals = theta[:,0]
        phi_vals = phi[0,:]
        fgrid = self.synth(f)
        
        # Equidistant 
        xx=theta*np.cos(phi)
        yy=theta*np.sin(phi)
        
        
        xx=xx[theta_vals<=np.pi/2,:]
        yy=yy[theta_vals<=np.pi/2,:]
        fgrid=fgrid[theta_vals<=np.pi/2,:]
        
        xx = np.hstack([xx, xx[:, 0][:, None]])
        yy = np.hstack([yy, yy[:, 0][:, None]])
        fgrid = np.hstack([fgrid, fgrid[:, 0][:, None]])
        
        return xx,yy,fgrid
    
        
    

    def y0synth(self,fvec):
        nang=1000
        nt = fvec.shape[1]
        th = np.linspace(-np.pi/2,np.pi/2,nang)
        
        fg = np.zeros((nang,nt))
        ph = np.zeros(th.shape)
        ph[th<0]=np.pi
        
        for l in range(0,self.lmax+1,2):
            for m in range(-l,l+1,1):
                M=abs(m)
                fg = fg + fvec[self.idx(l,abs(m)),:]*sph_harm(M,l,ph,th).reshape(nang,1)
      
        
        return fg.real/sqrt(4*np.pi),th




    def multiconstant(self,l,m):
        mc = [0.0] * 45
          
        if (abs(m-1)<=l):
            mc[0] = (0)+(-(sqrt(1+l-m)*sqrt(l+m))/4.)*1j

        if (abs(m+1)<=l):
            mc[1] = (0)+(-(sqrt(l-m)*sqrt(1+l+m))/4.)*1j
        
        if (abs(m-1)<=l):
            mc[2] = (-(sqrt(1+l-m)*sqrt(l+m))/4.)+(0)*1j
        
        if (abs(m+1)<=l):
            mc[3] = ((sqrt(l-m)*sqrt(1+l+m))/4.)+(0)*1j
        
        if (abs(m)<=l):
            mc[4] = (0)+(-m/2.)*1j
        
        if (abs(m-2)<=l-2):
            mc[5] = (((1+l)*sqrt(-3+l+m)*sqrt(-2+l+m)*sqrt(-1+l+m)*sqrt(l+m))/(8.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))+(0)*1j
        
        if (abs(m)<=l-2):
            mc[6] = (-((1+l)*sqrt(-1+l-m)*sqrt(l-m)*sqrt(-1+l+m)*sqrt(l+m))/(4.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))+(0)*1j
        
        if (abs(m+2)<=l-2):
            mc[7] = (((1+l)*sqrt(-3+l-m)*sqrt(-2+l-m)*sqrt(-1+l-m)*sqrt(l-m))/(8.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))+(0)*1j
        
        if (abs(m-2)<=l):
            mc[8] = ((-3*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(-1+l+m)*sqrt(l+m))/(8.*(-3+4*l+4*pow(l,2))))+(0)*1j
        
        if (abs(m)<=l):
            mc[9] = ((l+pow(l,2)-3*pow(m,2))/(12-16*l-16*pow(l,2)))+(0)*1j
        
        if (abs(m+2)<=l):
            mc[10] = ((-3*sqrt(-1+l-m)*sqrt(l-m)*sqrt(1+l+m)*sqrt(2+l+m))/(8.*(-3+4*l+4*pow(l,2))))+(0)*1j
        
        if (abs(m-2)<=l+2):
            mc[11] = (-((l*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(3+l-m)*sqrt(4+l-m))/(sqrt(1+2*l)*sqrt(5+2*l)*(24+16*l))))+(0)*1j
        
        if (abs(m)<=l+2):
            mc[12] = ((l*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(1+l+m)*sqrt(2+l+m))/(sqrt(1+2*l)*sqrt(5+2*l)*(12+8*l)))+(0)*1j
        
        if (abs(m+2)<=l+2):
            mc[13] = (-((l*sqrt(1+l+m)*sqrt(2+l+m)*sqrt(3+l+m)*sqrt(4+l+m))/(sqrt(1+2*l)*sqrt(5+2*l)*(24+16*l))))+(0)*1j
        
        if (abs(m-2)<=l-2):
            mc[14] = (0)+(-((1+l)*sqrt(-3+l+m)*sqrt(-2+l+m)*sqrt(-1+l+m)*sqrt(l+m))/(4.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))*1j
        
        if (abs(m+2)<=l-2):
            mc[15] = (0)+(((1+l)*sqrt(-3+l-m)*sqrt(-2+l-m)*sqrt(-1+l-m)*sqrt(l-m))/(4.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))*1j
        
        if (abs(m-2)<=l):
            mc[16] = (0)+((3*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(-1+l+m)*sqrt(l+m))/(4.*(-3+4*l+4*pow(l,2))))*1j
        
        if (abs(m+2)<=l):
            mc[17] = (0)+((-3*sqrt(-1+l-m)*sqrt(l-m)*sqrt(1+l+m)*sqrt(2+l+m))/(4.*(-3+4*l+4*pow(l,2))))*1j
        
        if (abs(m-2)<=l+2):
            mc[18] = (0)+((l*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(3+l-m)*sqrt(4+l-m))/(sqrt(1+2*l)*sqrt(5+2*l)*(12+8*l)))*1j
        
        if (abs(m+2)<=l+2):
            mc[19] = (0)+(-((l*sqrt(1+l+m)*sqrt(2+l+m)*sqrt(3+l+m)*sqrt(4+l+m))/(sqrt(1+2*l)*sqrt(5+2*l)*(12+8*l))))*1j
        
        if (abs(m-1)<=l-2):
            mc[20] = (-((1+l)*sqrt(l-m)*sqrt(-2+l+m)*sqrt(-1+l+m)*sqrt(l+m))/(2.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))+(0)*1j
        
        if (abs(m+1)<=l-2):
            mc[21] = (((1+l)*sqrt(-2+l-m)*sqrt(-1+l-m)*sqrt(l-m)*sqrt(l+m))/(2.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))+(0)*1j
        
        if (abs(m-1)<=l):
            mc[22] = ((3*(1-2*m)*sqrt(1+l-m)*sqrt(l+m))/(4.*(-3+4*l+4*pow(l,2))))+(0)*1j
        
        if (abs(m+1)<=l):
            mc[23] = ((-3*sqrt(l-m)*sqrt(1+l+m)*(1+2*m))/(4.*(-3+4*l+4*pow(l,2))))+(0)*1j
        
        if (abs(m-1)<=l+2):
            mc[24] = (-((l*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(3+l-m)*sqrt(1+l+m))/(sqrt(1+2*l)*sqrt(5+2*l)*(6+4*l))))+(0)*1j
        
        if (abs(m+1)<=l+2):
            mc[25] = ((l*sqrt(1+l-m)*sqrt(1+l+m)*sqrt(2+l+m)*sqrt(3+l+m))/(sqrt(1+2*l)*sqrt(5+2*l)*(6+4*l)))+(0)*1j
        
        if (abs(m-2)<=l-2):
            mc[26] = (-((1+l)*sqrt(-3+l+m)*sqrt(-2+l+m)*sqrt(-1+l+m)*sqrt(l+m))/(8.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))+(0)*1j
        
        if (abs(m)<=l-2):
            mc[27] = (-((1+l)*sqrt(-1+l-m)*sqrt(l-m)*sqrt(-1+l+m)*sqrt(l+m))/(4.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))+(0)*1j
        
        if (abs(m+2)<=l-2):
            mc[28] = (-((1+l)*sqrt(-3+l-m)*sqrt(-2+l-m)*sqrt(-1+l-m)*sqrt(l-m))/(8.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))+(0)*1j
        
        if (abs(m-2)<=l):
            mc[29] = ((3*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(-1+l+m)*sqrt(l+m))/(8.*(-3+4*l+4*pow(l,2))))+(0)*1j
        
        if (abs(m)<=l):
            mc[30] = ((l+pow(l,2)-3*pow(m,2))/(12-16*l-16*pow(l,2)))+(0)*1j
        
        if (abs(m+2)<=l):
            mc[31] = ((3*sqrt(-1+l-m)*sqrt(l-m)*sqrt(1+l+m)*sqrt(2+l+m))/(8.*(-3+4*l+4*pow(l,2))))+(0)*1j
        
        if (abs(m-2)<=l+2):
            mc[32] = ((l*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(3+l-m)*sqrt(4+l-m))/(sqrt(1+2*l)*sqrt(5+2*l)*(24+16*l)))+(0)*1j
        
        if (abs(m)<=l+2):
            mc[33] = ((l*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(1+l+m)*sqrt(2+l+m))/(sqrt(1+2*l)*sqrt(5+2*l)*(12+8*l)))+(0)*1j
        
        if (abs(m+2)<=l+2):
            mc[34] = ((l*sqrt(1+l+m)*sqrt(2+l+m)*sqrt(3+l+m)*sqrt(4+l+m))/(sqrt(1+2*l)*sqrt(5+2*l)*(24+16*l)))+(0)*1j
        
        if (abs(m-1)<=l-2):
            mc[35] = (0)+(((1+l)*sqrt(l-m)*sqrt(-2+l+m)*sqrt(-1+l+m)*sqrt(l+m))/(2.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))*1j
        
        if (abs(m+1)<=l-2):
            mc[36] = (0)+(((1+l)*sqrt(-2+l-m)*sqrt(-1+l-m)*sqrt(l-m)*sqrt(l+m))/(2.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))*1j
        
        if (abs(m-1)<=l):
            mc[37] = (0)+((3*sqrt(1+l-m)*sqrt(l+m)*(-1+2*m))/(4.*(-3+4*l+4*pow(l,2))))*1j
        
        if (abs(m+1)<=l):
            mc[38] = (0)+((-3*sqrt(l-m)*sqrt(1+l+m)*(1+2*m))/(4.*(-3+4*l+4*pow(l,2))))*1j
        
        if (abs(m-1)<=l+2):
            mc[39] = (0)+((l*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(3+l-m)*sqrt(1+l+m))/(sqrt(1+2*l)*sqrt(5+2*l)*(6+4*l)))*1j
        
        if (abs(m+1)<=l+2):
            mc[40] = (0)+((l*sqrt(1+l-m)*sqrt(1+l+m)*sqrt(2+l+m)*sqrt(3+l+m))/(sqrt(1+2*l)*sqrt(5+2*l)*(6+4*l)))*1j
        
        if (abs(m)<=l-2):
            mc[41] = (((1+l)*sqrt(-1+l-m)*sqrt(l-m)*sqrt(-1+l+m)*sqrt(l+m))/(2.*sqrt(-3+2*l)*(-1+2*l)*sqrt(1+2*l)))+(0)*1j
        
        if (abs(m)<=l):
            mc[42] = ((l+pow(l,2)-3*pow(m,2))/(-6+8*l+8*pow(l,2)))+(0)*1j
        
        if (abs(m)<=l+2):
            mc[43] = (-((l*sqrt(1+l-m)*sqrt(2+l-m)*sqrt(1+l+m)*sqrt(2+l+m))/(sqrt(1+2*l)*sqrt(5+2*l)*(6+4*l))))+(0)*1j
        
        mc=2*mc
        return mc
  
    
    def xx(self,psi,l,m):
    #Expansion in spherical harmonics of xx
    
        f=  self.index(psi,l-2,m-2)*((((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(32*pow(l,3)-48*pow(l,2)-8*l+12.))+(0.)*1j)) + \
            self.index(psi,l-2,m)*(((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(16*pow(l,3)-24*pow(l,2)-4*l+6.))+(0.)*1j)) + \
            self.index(psi,l-2,m+2)*((((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(32*pow(l,3)-48*pow(l,2)-8*l+12.))+(0.)*1j)) + \
            self.index(psi,l,m-2)*(((-(sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l))/(8*pow(l,2)+8*l-6.))+(0.)*1j)) + \
            self.index(psi,l,m)*((((pow(m,2)+pow(l,2)+l-1.)/(4*pow(l,2)+4*l-3.))+(0.)*1j)) + \
            self.index(psi,l,m+2)*(((-(sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(8*pow(l,2)+8*l-6.))+(0.)*1j)) + \
            self.index(psi,l+2,m-2)*((((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.))/(32*pow(l,3)+144*pow(l,2)+184*l+60.))+(0.)*1j)) + \
            self.index(psi,l+2,m)*(((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(16*pow(l,3)+72*pow(l,2)+92*l+30.))+(0.)*1j)) + \
            self.index(psi,l+2,m+2)*((((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(32*pow(l,3)+144*pow(l,2)+184*l+60.))+(0.)*1j))
        
        
        return f
    
    def xxxy(self,psi,l,m):
    #Expansion in spherical harmonics of xxxy
    
    
        f= self.index(psi,l+-2,m+-2)*((0.)+(-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(2*pow(m,2)+((-2*l)-3.)*m+2*pow(l,2)-5.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))*1j) + \
        self.index(psi,l+-2,m+-4)*((0.)+((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))*1j) + \
        self.index(psi,l+-2,m+2)*((0.)+((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*(2*pow(m,2)+(2*l+3.)*m+2*pow(l,2)-5.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))*1j) + \
        self.index(psi,l+-2,m+4)*((0.)+(-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))*1j) + \
        self.index(psi,l+-4,m+-2)*((0.)+((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))*1j) + \
        self.index(psi,l+-4,m+-4)*((0.)+(-(sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt(m+l-7.)*sqrt(m+l-6.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(512*pow(l,5)-3840*pow(l,4)+8960*pow(l,3)-5760*pow(l,2)-2272*l+1680.))*1j) + \
        self.index(psi,l+-4,m+2)*((0.)+(-(sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))*1j) + \
        self.index(psi,l+-4,m+4)*((0.)+((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-7.)*sqrt((-m)+l-6.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(512*pow(l,5)-3840*pow(l,4)+8960*pow(l,3)-5760*pow(l,2)-2272*l+1680.))*1j) + \
        self.index(psi,l+0,m+-2)*((0.)+((sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l)*(3*pow(m,2)-6*m+3*pow(l,2)+3*l-9.))/(64*pow(l,4)+128*pow(l,3)-224*pow(l,2)-288*l+180.))*1j) + \
        self.index(psi,l+0,m+-4)*((0.)+(-(3*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j) + \
        self.index(psi,l+0,m+2)*((0.)+(-(sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*(3*pow(m,2)+6*m+3*pow(l,2)+3*l-9.))/(64*pow(l,4)+128*pow(l,3)-224*pow(l,2)-288*l+180.))*1j) + \
        self.index(psi,l+0,m+4)*((0.)+((3*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j) + \
        self.index(psi,l+2,m+-2)*((0.)+(-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*(2*pow(m,2)+(2*l-1.)*m+2*pow(l,2)+4*l-3.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))*1j) + \
        self.index(psi,l+2,m+-4)*((0.)+((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))*1j) + \
        self.index(psi,l+2,m+2)*((0.)+((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*(2*pow(m,2)+(1.-2*l)*m+2*pow(l,2)+4*l-3.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))*1j) + \
        self.index(psi,l+2,m+4)*((0.)+(-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))*1j) + \
        self.index(psi,l+4,m+-2)*((0.)+((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))*1j) + \
        self.index(psi,l+4,m+-4)*((0.)+(-(sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt((-m)+l+7.)*sqrt((-m)+l+8.))/(512*pow(l,5)+6400*pow(l,4)+29440*pow(l,3)+60800*pow(l,2)+54048*l+15120.))*1j) + \
        self.index(psi,l+4,m+2)*((0.)+(-(sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))*1j) + \
        self.index(psi,l+4,m+4)*((0.)+((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.)*sqrt(m+l+7.)*sqrt(m+l+8.))/(512*pow(l,5)+6400*pow(l,4)+29440*pow(l,3)+60800*pow(l,2)+54048*l+15120.))*1j)
        
        return f
    
    def xxyy(self,psi,l,m):
    #Expansion in spherical harmonics of xxyy
    
    
        f=  self.index(psi,l+-2,m+-4)*(((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))+(0.)*1j) + \
            self.index(psi,l+-2,m+0)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l)*(pow(m,2)+pow(l,2)-l-4.))/(64*pow(l,5)-160*pow(l,4)-160*pow(l,3)+400*pow(l,2)+36*l-90.))+(0.)*1j) + \
            self.index(psi,l+-2,m+4)*(((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))+(0.)*1j) + \
            self.index(psi,l+-4,m+-4)*((-(sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt(m+l-7.)*sqrt(m+l-6.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(512*pow(l,5)-3840*pow(l,4)+8960*pow(l,3)-5760*pow(l,2)-2272*l+1680.))+(0.)*1j) + \
            self.index(psi,l+-4,m+0)*(((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))+(0.)*1j) + \
            self.index(psi,l+-4,m+4)*((-(sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-7.)*sqrt((-m)+l-6.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(512*pow(l,5)-3840*pow(l,4)+8960*pow(l,3)-5760*pow(l,2)-2272*l+1680.))+(0.)*1j) + \
            self.index(psi,l+0,m+-4)*((-(3*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j) + \
            self.index(psi,l+0,m+0)*(((3*pow(m,4)+(2*pow(l,2)+2*l-15.)*pow(m,2)+3*pow(l,4)+6*pow(l,3)-11*pow(l,2)-14*l+12.)/(64*pow(l,4)+128*pow(l,3)-224*pow(l,2)-288*l+180.))+(0.)*1j) + \
            self.index(psi,l+0,m+4)*((-(3*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j) + \
            self.index(psi,l+2,m+-4)*(((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))+(0.)*1j) + \
            self.index(psi,l+2,m+0)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*(pow(m,2)+pow(l,2)+3*l-2.))/(64*pow(l,5)+480*pow(l,4)+1120*pow(l,3)+720*pow(l,2)-284*l-210.))+(0.)*1j) + \
            self.index(psi,l+2,m+4)*(((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))+(0.)*1j) + \
            self.index(psi,l+4,m+-4)*((-(sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt((-m)+l+7.)*sqrt((-m)+l+8.))/(512*pow(l,5)+6400*pow(l,4)+29440*pow(l,3)+60800*pow(l,2)+54048*l+15120.))+(0.)*1j) + \
            self.index(psi,l+4,m+0)*(((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))+(0.)*1j) + \
            self.index(psi,l+4,m+4)*((-(sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.)*sqrt(m+l+7.)*sqrt(m+l+8.))/(512*pow(l,5)+6400*pow(l,4)+29440*pow(l,3)+60800*pow(l,2)+54048*l+15120.))+(0.)*1j)
        
        return f 
    
    def xxzz(self,psi,l,m):
    #Expansion in spherical harmonics of xxzz
    
    
        f= self.index(psi,l+-2,m+-2)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*pow(m,2)+((-4*l)-6.)*m+4*l+5.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))+(0.)*1j) + \
            self.index(psi,l+-2,m+0)*(((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l)*(4*pow(m,2)-1.))/(64*pow(l,5)-160*pow(l,4)-160*pow(l,3)+400*pow(l,2)+36*l-90.))+(0.)*1j) + \
            self.index(psi,l+-2,m+2)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*(4*pow(m,2)+(4*l+6.)*m+4*l+5.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))+(0.)*1j) + \
            self.index(psi,l+-4,m+-2)*(((sqrt(2*l-7.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(64*pow(l,4)-512*pow(l,3)+1376*pow(l,2)-1408*l+420.)))+(0.)*1j) + \
            self.index(psi,l+-4,m+0)*((-(sqrt(2*l-7.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(32*pow(l,4)-256*pow(l,3)+688*pow(l,2)-704*l+210.)))+(0.)*1j) + \
            self.index(psi,l+-4,m+2)*(((sqrt(2*l-7.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(64*pow(l,4)-512*pow(l,3)+1376*pow(l,2)-1408*l+420.)))+(0.)*1j) + \
            self.index(psi,l+0,m+-2)*(((sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l)*(3*pow(m,2)-6*m-pow(l,2)-l+6.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))+(0.)*1j) + \
            self.index(psi,l+0,m+0)*((-(3*pow(m,4)+((-2*pow(l,2))-2*l)*pow(m,2)-pow(l,4)-2*pow(l,3)+4*pow(l,2)+5*l-3.)/(16*pow(l,4)+32*pow(l,3)-56*pow(l,2)-72*l+45.))+(0.)*1j) + \
            self.index(psi,l+0,m+2)*(((sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*(3*pow(m,2)+6*m-pow(l,2)-l+6.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))+(0.)*1j) + \
            self.index(psi,l+2,m+-2)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*(4*pow(m,2)+(4*l-2.)*m-4*l+1.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))+(0.)*1j) + \
            self.index(psi,l+2,m+0)*(((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*(4*pow(m,2)-1.))/(64*pow(l,5)+480*pow(l,4)+1120*pow(l,3)+720*pow(l,2)-284*l-210.))+(0.)*1j) + \
            self.index(psi,l+2,m+2)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*(4*pow(m,2)+(2.-4*l)*m-4*l+1.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))+(0.)*1j) + \
            self.index(psi,l+4,m+-2)*(((sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(sqrt(2*l+1.)*(64*pow(l,4)+768*pow(l,3)+3296*pow(l,2)+5952*l+3780.)))+(0.)*1j) + \
            self.index(psi,l+4,m+0)*((-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(sqrt(2*l+1.)*(32*pow(l,4)+384*pow(l,3)+1648*pow(l,2)+2976*l+1890.)))+(0.)*1j) + \
            self.index(psi,l+4,m+2)*(((sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(sqrt(2*l+1.)*(64*pow(l,4)+768*pow(l,3)+3296*pow(l,2)+5952*l+3780.)))+(0.)*1j)
        
        return f
    
    def xyyy(self,psi,l,m):
    #Expansion in spherical harmonics of xyyy
    
    
        f=self.index(psi,l+-2,m+-2)*( (0.)+(-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(2*pow(m,2)+((-2*l)-3.)*m+2*pow(l,2)-5.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))*1j )+ \
            self.index(psi,l+-2,m+-4)*( (0.)+(-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))*1j )+ \
            self.index(psi,l+-2,m+2)*( (0.)+((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*(2*pow(m,2)+(2*l+3.)*m+2*pow(l,2)-5.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))*1j )+ \
            self.index(psi,l+-2,m+4)*( (0.)+((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))*1j )+ \
            self.index(psi,l+-4,m+-2)*( (0.)+((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))*1j )+ \
            self.index(psi,l+-4,m+-4)*( (0.)+((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt(m+l-7.)*sqrt(m+l-6.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(512*pow(l,5)-3840*pow(l,4)+8960*pow(l,3)-5760*pow(l,2)-2272*l+1680.))*1j )+ \
            self.index(psi,l+-4,m+2)*( (0.)+(-(sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))*1j )+ \
            self.index(psi,l+-4,m+4)*( (0.)+(-(sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-7.)*sqrt((-m)+l-6.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(512*pow(l,5)-3840*pow(l,4)+8960*pow(l,3)-5760*pow(l,2)-2272*l+1680.))*1j )+ \
            self.index(psi,l+0,m+-2)*( (0.)+((sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l)*(3*pow(m,2)-6*m+3*pow(l,2)+3*l-9.))/(64*pow(l,4)+128*pow(l,3)-224*pow(l,2)-288*l+180.))*1j )+ \
            self.index(psi,l+0,m+-4)*( (0.)+((3*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j )+ \
            self.index(psi,l+0,m+2)*( (0.)+(-(sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*(3*pow(m,2)+6*m+3*pow(l,2)+3*l-9.))/(64*pow(l,4)+128*pow(l,3)-224*pow(l,2)-288*l+180.))*1j )+ \
            self.index(psi,l+0,m+4)*( (0.)+(-(3*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j )+ \
            self.index(psi,l+2,m+-2)*( (0.)+(-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*(2*pow(m,2)+(2*l-1.)*m+2*pow(l,2)+4*l-3.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))*1j )+ \
            self.index(psi,l+2,m+-4)*( (0.)+(-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))*1j )+ \
            self.index(psi,l+2,m+2)*( (0.)+((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*(2*pow(m,2)+(1.-2*l)*m+2*pow(l,2)+4*l-3.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))*1j )+ \
            self.index(psi,l+2,m+4)*( (0.)+((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))*1j )+ \
            self.index(psi,l+4,m+-2)*( (0.)+((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))*1j )+ \
            self.index(psi,l+4,m+-4)*( (0.)+((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt((-m)+l+7.)*sqrt((-m)+l+8.))/(512*pow(l,5)+6400*pow(l,4)+29440*pow(l,3)+60800*pow(l,2)+54048*l+15120.))*1j )+ \
            self.index(psi,l+4,m+2)*( (0.)+(-(sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))*1j )+ \
            self.index(psi,l+4,m+4)*( (0.)+(-(sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.)*sqrt(m+l+7.)*sqrt(m+l+8.))/(512*pow(l,5)+6400*pow(l,4)+29440*pow(l,3)+60800*pow(l,2)+54048*l+15120.))*1j )
        
        return f
    
    def xyzz(self,psi,l,m):
    #Expansion in spherical harmonics of xyyz
    
    
        f=self.index(psi,l+-2,m+-2)*( (0.)+((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*pow(m,2)+((-4*l)-6.)*m+4*l+5.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))*1j )+ \
            self.index(psi,l+-2,m+2)*( (0.)+(-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*(4*pow(m,2)+(4*l+6.)*m+4*l+5.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))*1j )+ \
            self.index(psi,l+-4,m+-2)*( (0.)+(-(sqrt(2*l-7.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(64*pow(l,4)-512*pow(l,3)+1376*pow(l,2)-1408*l+420.)))*1j )+ \
            self.index(psi,l+-4,m+2)*( (0.)+((sqrt(2*l-7.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(64*pow(l,4)-512*pow(l,3)+1376*pow(l,2)-1408*l+420.)))*1j )+ \
            self.index(psi,l+0,m+-2)*( (0.)+(-(sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l)*(3*pow(m,2)-6*m-pow(l,2)-l+6.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))*1j )+ \
            self.index(psi,l+0,m+2)*( (0.)+((sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*(3*pow(m,2)+6*m-pow(l,2)-l+6.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))*1j )+ \
            self.index(psi,l+2,m+-2)*( (0.)+((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*(4*pow(m,2)+(4*l-2.)*m-4*l+1.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))*1j )+ \
            self.index(psi,l+2,m+2)*( (0.)+(-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*(4*pow(m,2)+(2.-4*l)*m-4*l+1.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))*1j )+ \
            self.index(psi,l+4,m+-2)*( (0.)+(-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(sqrt(2*l+1.)*(64*pow(l,4)+768*pow(l,3)+3296*pow(l,2)+5952*l+3780.)))*1j )+ \
            self.index(psi,l+4,m+2)*( (0.)+((sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(sqrt(2*l+1.)*(64*pow(l,4)+768*pow(l,3)+3296*pow(l,2)+5952*l+3780.)))*1j )
        
        return f
    
    def xzzz(self,psi,l,m):
    #Expansion in spherical harmonics of xzzz
    
    
        f=self.index(psi,l+-2,m+-1)*( ((sqrt(2*l-3.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*pow(m,2)+((-2*l)-3.)*m-2*pow(l,2)+3*l+8.))/(sqrt(2*l+1.)*(32*pow(l,4)-96*pow(l,3)-32*pow(l,2)+216*l-90.)))+(0.)*1j )+ \
            self.index(psi,l+-2,m+1)*( (-(sqrt(2*l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l)*(4*pow(m,2)+(2*l+3.)*m-2*pow(l,2)+3*l+8.))/(sqrt(2*l+1.)*(32*pow(l,4)-96*pow(l,3)-32*pow(l,2)+216*l-90.)))+(0.)*1j )+ \
            self.index(psi,l+-4,m+-1)*( (-(sqrt(2*l-7.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(32*pow(l,4)-256*pow(l,3)+688*pow(l,2)-704*l+210.)))+(0.)*1j )+ \
            self.index(psi,l+-4,m+1)*( ((sqrt(2*l-7.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(32*pow(l,4)-256*pow(l,3)+688*pow(l,2)-704*l+210.)))+(0.)*1j )+ \
            self.index(psi,l+0,m+-1)*( ((sqrt((-m)+l+1.)*sqrt(m+l)*(6*pow(m,3)-9*pow(m,2)+((-6*pow(l,2))-6*l+21.)*m+3*pow(l,2)+3*l-9.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))+(0.)*1j )+ \
            self.index(psi,l+0,m+1)*( ((sqrt(l-m)*sqrt(m+l+1.)*(6*pow(m,3)+9*pow(m,2)+((-6*pow(l,2))-6*l+21.)*m-3*pow(l,2)-3*l+9.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))+(0.)*1j )+ \
            self.index(psi,l+2,m+-1)*( (-(sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*(4*pow(m,2)+(2*l-1.)*m-2*pow(l,2)-7*l+3.))/(sqrt(2*l+1.)*(32*pow(l,4)+224*pow(l,3)+448*pow(l,2)+136*l-210.)))+(0.)*1j )+ \
            self.index(psi,l+2,m+1)*( ((sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*(4*pow(m,2)+(1.-2*l)*m-2*pow(l,2)-7*l+3.))/(sqrt(2*l+1.)*(32*pow(l,4)+224*pow(l,3)+448*pow(l,2)+136*l-210.)))+(0.)*1j )+ \
            self.index(psi,l+4,m+-1)*( ((sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(sqrt(2*l+1.)*(32*pow(l,4)+384*pow(l,3)+1648*pow(l,2)+2976*l+1890.)))+(0.)*1j )+ \
            self.index(psi,l+4,m+1)*( (-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.))/(sqrt(2*l+1.)*(32*pow(l,4)+384*pow(l,3)+1648*pow(l,2)+2976*l+1890.)))+(0.)*1j )
        
        return f
    
    def yyyy(self,psi,l,m):
    #Expansion in spherical harmonics of yyyy
    
    
        f=  self.index(psi,l+-2,m+-2)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(2*pow(m,2)+((-2*l)-3.)*m+2*pow(l,2)-5.))/(64*pow(l,5)-160*pow(l,4)-160*pow(l,3)+400*pow(l,2)+36*l-90.))+(0.)*1j) + \
            self.index(psi,l+-2,m+-4)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))+(0.)*1j) + \
            self.index(psi,l+-2,m+0)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l)*(3*pow(m,2)+3*pow(l,2)-3*l-12.))/(64*pow(l,5)-160*pow(l,4)-160*pow(l,3)+400*pow(l,2)+36*l-90.))+(0.)*1j) + \
            self.index(psi,l+-2,m+2)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*(2*pow(m,2)+(2*l+3.)*m+2*pow(l,2)-5.))/(64*pow(l,5)-160*pow(l,4)-160*pow(l,3)+400*pow(l,2)+36*l-90.))+(0.)*1j) + \
            self.index(psi,l+-2,m+4)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))+(0.)*1j) + \
            self.index(psi,l+-4,m+-2)*(((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)-960*pow(l,4)+2240*pow(l,3)-1440*pow(l,2)-568*l+420.))+(0.)*1j) + \
            self.index(psi,l+-4,m+-4)*(((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt(m+l-7.)*sqrt(m+l-6.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(512*pow(l,5)-3840*pow(l,4)+8960*pow(l,3)-5760*pow(l,2)-2272*l+1680.))+(0.)*1j) + \
            self.index(psi,l+-4,m+0)*(((3*sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))+(0.)*1j) + \
            self.index(psi,l+-4,m+2)*(((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)-960*pow(l,4)+2240*pow(l,3)-1440*pow(l,2)-568*l+420.))+(0.)*1j) + \
            self.index(psi,l+-4,m+4)*(((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-7.)*sqrt((-m)+l-6.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(512*pow(l,5)-3840*pow(l,4)+8960*pow(l,3)-5760*pow(l,2)-2272*l+1680.))+(0.)*1j) + \
            self.index(psi,l+0,m+-2)*(((sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l)*(3*pow(m,2)-6*m+3*pow(l,2)+3*l-9.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))+(0.)*1j) + \
            self.index(psi,l+0,m+-4)*(((3*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j) + \
            self.index(psi,l+0,m+0)*(((9*pow(m,4)+(6*pow(l,2)+6*l-45.)*pow(m,2)+9*pow(l,4)+18*pow(l,3)-33*pow(l,2)-42*l+36.)/(64*pow(l,4)+128*pow(l,3)-224*pow(l,2)-288*l+180.))+(0.)*1j) + \
            self.index(psi,l+0,m+2)*(((sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*(3*pow(m,2)+6*m+3*pow(l,2)+3*l-9.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))+(0.)*1j) + \
            self.index(psi,l+0,m+4)*(((3*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j) + \
            self.index(psi,l+2,m+-2)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*(2*pow(m,2)+(2*l-1.)*m+2*pow(l,2)+4*l-3.))/(64*pow(l,5)+480*pow(l,4)+1120*pow(l,3)+720*pow(l,2)-284*l-210.))+(0.)*1j) + \
            self.index(psi,l+2,m+-4)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))+(0.)*1j) + \
            self.index(psi,l+2,m+0)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*(3*pow(m,2)+3*pow(l,2)+9*l-6.))/(64*pow(l,5)+480*pow(l,4)+1120*pow(l,3)+720*pow(l,2)-284*l-210.))+(0.)*1j) + \
            self.index(psi,l+2,m+2)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*(2*pow(m,2)+(1.-2*l)*m+2*pow(l,2)+4*l-3.))/(64*pow(l,5)+480*pow(l,4)+1120*pow(l,3)+720*pow(l,2)-284*l-210.))+(0.)*1j) + \
            self.index(psi,l+2,m+4)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))+(0.)*1j) + \
            self.index(psi,l+4,m+-2)*(((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(128*pow(l,5)+1600*pow(l,4)+7360*pow(l,3)+15200*pow(l,2)+13512*l+3780.))+(0.)*1j) + \
            self.index(psi,l+4,m+-4)*(((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt((-m)+l+7.)*sqrt((-m)+l+8.))/(512*pow(l,5)+6400*pow(l,4)+29440*pow(l,3)+60800*pow(l,2)+54048*l+15120.))+(0.)*1j) + \
            self.index(psi,l+4,m+0)*(((3*sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))+(0.)*1j) + \
            self.index(psi,l+4,m+2)*(((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(128*pow(l,5)+1600*pow(l,4)+7360*pow(l,3)+15200*pow(l,2)+13512*l+3780.))+(0.)*1j) + \
            self.index(psi,l+4,m+4)*(((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.)*sqrt(m+l+7.)*sqrt(m+l+8.))/(512*pow(l,5)+6400*pow(l,4)+29440*pow(l,3)+60800*pow(l,2)+54048*l+15120.))+(0.)*1j)
        
        return f
    
    
    def yyzz(self,psi,l,m):
    #Expansion in spherical harmonics of yyzz
    
    
        f= self.index(psi,l+-2,m+-2)*(((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*pow(m,2)+((-4*l)-6.)*m+4*l+5.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))+(0.)*1j) + \
        self.index(psi,l+-2,m+0)*(((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l)*(4*pow(m,2)-1.))/(64*pow(l,5)-160*pow(l,4)-160*pow(l,3)+400*pow(l,2)+36*l-90.))+(0.)*1j) + \
        self.index(psi,l+-2,m+2)*(((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*(4*pow(m,2)+(4*l+6.)*m+4*l+5.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))+(0.)*1j) + \
        self.index(psi,l+-4,m+-2)*((-(sqrt(2*l-7.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(64*pow(l,4)-512*pow(l,3)+1376*pow(l,2)-1408*l+420.)))+(0.)*1j) + \
        self.index(psi,l+-4,m+0)*((-(sqrt(2*l-7.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(32*pow(l,4)-256*pow(l,3)+688*pow(l,2)-704*l+210.)))+(0.)*1j) + \
        self.index(psi,l+-4,m+2)*((-(sqrt(2*l-7.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(64*pow(l,4)-512*pow(l,3)+1376*pow(l,2)-1408*l+420.)))+(0.)*1j) + \
        self.index(psi,l+0,m+-2)*((-(sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l)*(3*pow(m,2)-6*m-pow(l,2)-l+6.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))+(0.)*1j) + \
        self.index(psi,l+0,m+0)*((-(3*pow(m,4)+((-2*pow(l,2))-2*l)*pow(m,2)-pow(l,4)-2*pow(l,3)+4*pow(l,2)+5*l-3.)/(16*pow(l,4)+32*pow(l,3)-56*pow(l,2)-72*l+45.))+(0.)*1j) + \
        self.index(psi,l+0,m+2)*((-(sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*(3*pow(m,2)+6*m-pow(l,2)-l+6.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))+(0.)*1j) + \
        self.index(psi,l+2,m+-2)*(((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*(4*pow(m,2)+(4*l-2.)*m-4*l+1.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))+(0.)*1j) + \
        self.index(psi,l+2,m+0)*(((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*(4*pow(m,2)-1.))/(64*pow(l,5)+480*pow(l,4)+1120*pow(l,3)+720*pow(l,2)-284*l-210.))+(0.)*1j) + \
        self.index(psi,l+2,m+2)*(((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*(4*pow(m,2)+(2.-4*l)*m-4*l+1.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))+(0.)*1j) + \
        self.index(psi,l+4,m+-2)*((-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(sqrt(2*l+1.)*(64*pow(l,4)+768*pow(l,3)+3296*pow(l,2)+5952*l+3780.)))+(0.)*1j) + \
        self.index(psi,l+4,m+0)*((-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(sqrt(2*l+1.)*(32*pow(l,4)+384*pow(l,3)+1648*pow(l,2)+2976*l+1890.)))+(0.)*1j) + \
        self.index(psi,l+4,m+2)*((-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(sqrt(2*l+1.)*(64*pow(l,4)+768*pow(l,3)+3296*pow(l,2)+5952*l+3780.)))+(0.)*1j)
        
        return f
    
    def yzzz(self,psi,l,m):
    #Expansion in spherical harmonics of yzzz
    
    
        f=self.index(psi,l+-2,m+-1)*( (0.)+(-(sqrt(2*l-3.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*pow(m,2)+((-2*l)-3.)*m-2*pow(l,2)+3*l+8.))/(sqrt(2*l+1.)*(32*pow(l,4)-96*pow(l,3)-32*pow(l,2)+216*l-90.)))*1j )+ \
            self.index(psi,l+-2,m+1)*( (0.)+(-(sqrt(2*l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l)*(4*pow(m,2)+(2*l+3.)*m-2*pow(l,2)+3*l+8.))/(sqrt(2*l+1.)*(32*pow(l,4)-96*pow(l,3)-32*pow(l,2)+216*l-90.)))*1j )+ \
            self.index(psi,l+-4,m+-1)*( (0.)+((sqrt(2*l-7.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(32*pow(l,4)-256*pow(l,3)+688*pow(l,2)-704*l+210.)))*1j )+ \
            self.index(psi,l+-4,m+1)*( (0.)+((sqrt(2*l-7.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(32*pow(l,4)-256*pow(l,3)+688*pow(l,2)-704*l+210.)))*1j )+ \
            self.index(psi,l+0,m+-1)*( (0.)+(-(sqrt((-m)+l+1.)*sqrt(m+l)*(6*pow(m,3)-9*pow(m,2)+((-6*pow(l,2))-6*l+21.)*m+3*pow(l,2)+3*l-9.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))*1j )+ \
            self.index(psi,l+0,m+1)*( (0.)+((sqrt(l-m)*sqrt(m+l+1.)*(6*pow(m,3)+9*pow(m,2)+((-6*pow(l,2))-6*l+21.)*m-3*pow(l,2)-3*l+9.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))*1j )+ \
            self.index(psi,l+2,m+-1)*( (0.)+((sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*(4*pow(m,2)+(2*l-1.)*m-2*pow(l,2)-7*l+3.))/(sqrt(2*l+1.)*(32*pow(l,4)+224*pow(l,3)+448*pow(l,2)+136*l-210.)))*1j )+ \
            self.index(psi,l+2,m+1)*( (0.)+((sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*(4*pow(m,2)+(1.-2*l)*m-2*pow(l,2)-7*l+3.))/(sqrt(2*l+1.)*(32*pow(l,4)+224*pow(l,3)+448*pow(l,2)+136*l-210.)))*1j )+ \
            self.index(psi,l+4,m+-1)*( (0.)+(-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(sqrt(2*l+1.)*(32*pow(l,4)+384*pow(l,3)+1648*pow(l,2)+2976*l+1890.)))*1j )+ \
            self.index(psi,l+4,m+1)*( (0.)+(-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.))/(sqrt(2*l+1.)*(32*pow(l,4)+384*pow(l,3)+1648*pow(l,2)+2976*l+1890.)))*1j )
        
        return f
    
    def zzzz(self,psi,l,m):
    #Expansion in spherical harmonics of zzzz
    
    
        f=  self.index(psi,l+-2,m+0)*((-(sqrt(2*l-3.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l)*(4*pow(m,2)-4*pow(l,2)+4*l+14.))/(sqrt(2*l+1.)*(16*pow(l,4)-48*pow(l,3)-16*pow(l,2)+108*l-45.)))+(0.)*1j) + \
            self.index(psi,l+-4,m+0)*(((sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l-7.)*sqrt(2*l+1.)*(8*pow(l,3)-36*pow(l,2)+46*l-15.)))+(0.)*1j) + \
            self.index(psi,l+0,m+0)*(((6*pow(m,4)+((-12*pow(l,2))-12*l+30.)*pow(m,2)+6*pow(l,4)+12*pow(l,3)-18*pow(l,2)-24*l+9.)/(16*pow(l,4)+32*pow(l,3)-56*pow(l,2)-72*l+45.))+(0.)*1j) + \
            self.index(psi,l+2,m+0)*((-(sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*(4*pow(m,2)-4*pow(l,2)-12*l+6.))/(sqrt(2*l+1.)*(16*pow(l,4)+112*pow(l,3)+224*pow(l,2)+68*l-105.)))+(0.)*1j) + \
            self.index(psi,l+4,m+0)*(((sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(sqrt(2*l+1.)*sqrt(2*l+9.)*(8*pow(l,3)+60*pow(l,2)+142*l+105.)))+(0.)*1j)
        
        return f 
    def xxxx(self,psi,l,m):
    #Expansion in spherical harmonics of xxxx
    
    
        f=  self.index(psi,l+-2,m+-2)*(((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(2*pow(m,2)+((-2*l)-3.)*m+2*pow(l,2)-5.))/(64*pow(l,5)-160*pow(l,4)-160*pow(l,3)+400*pow(l,2)+36*l-90.))+(0.)*1j) + \
            self.index(psi,l+-2,m+-4)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))+(0.)*1j) + \
            self.index(psi,l+-2,m+0)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l)*(3*pow(m,2)+3*pow(l,2)-3*l-12.))/(64*pow(l,5)-160*pow(l,4)-160*pow(l,3)+400*pow(l,2)+36*l-90.))+(0.)*1j) + \
            self.index(psi,l+-2,m+2)*(((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*(2*pow(m,2)+(2*l+3.)*m+2*pow(l,2)-5.))/(64*pow(l,5)-160*pow(l,4)-160*pow(l,3)+400*pow(l,2)+36*l-90.))+(0.)*1j) + \
            self.index(psi,l+-2,m+4)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(128*pow(l,5)-320*pow(l,4)-320*pow(l,3)+800*pow(l,2)+72*l-180.))+(0.)*1j) + \
            self.index(psi,l+-4,m+-2)*((-(sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)-960*pow(l,4)+2240*pow(l,3)-1440*pow(l,2)-568*l+420.))+(0.)*1j) + \
            self.index(psi,l+-4,m+-4)*(((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt(m+l-7.)*sqrt(m+l-6.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(512*pow(l,5)-3840*pow(l,4)+8960*pow(l,3)-5760*pow(l,2)-2272*l+1680.))+(0.)*1j) + \
            self.index(psi,l+-4,m+0)*(((3*sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))+(0.)*1j) + \
            self.index(psi,l+-4,m+2)*((-(sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)-960*pow(l,4)+2240*pow(l,3)-1440*pow(l,2)-568*l+420.))+(0.)*1j) + \
            self.index(psi,l+-4,m+4)*(((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-7.)*sqrt((-m)+l-6.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(512*pow(l,5)-3840*pow(l,4)+8960*pow(l,3)-5760*pow(l,2)-2272*l+1680.))+(0.)*1j) + \
            self.index(psi,l+0,m+-2)*((-(sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l)*(3*pow(m,2)-6*m+3*pow(l,2)+3*l-9.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))+(0.)*1j) + \
            self.index(psi,l+0,m+-4)*(((3*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j) + \
            self.index(psi,l+0,m+0)*(((9*pow(m,4)+(6*pow(l,2)+6*l-45.)*pow(m,2)+9*pow(l,4)+18*pow(l,3)-33*pow(l,2)-42*l+36.)/(64*pow(l,4)+128*pow(l,3)-224*pow(l,2)-288*l+180.))+(0.)*1j) + \
            self.index(psi,l+0,m+2)*((-(sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*(3*pow(m,2)+6*m+3*pow(l,2)+3*l-9.))/(32*pow(l,4)+64*pow(l,3)-112*pow(l,2)-144*l+90.))+(0.)*1j) + \
            self.index(psi,l+0,m+4)*(((3*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j) + \
            self.index(psi,l+2,m+-2)*(((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*(2*pow(m,2)+(2*l-1.)*m+2*pow(l,2)+4*l-3.))/(64*pow(l,5)+480*pow(l,4)+1120*pow(l,3)+720*pow(l,2)-284*l-210.))+(0.)*1j) + \
            self.index(psi,l+2,m+-4)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l-1.)*sqrt(m+l))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))+(0.)*1j) + \
            self.index(psi,l+2,m+0)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*(3*pow(m,2)+3*pow(l,2)+9*l-6.))/(64*pow(l,5)+480*pow(l,4)+1120*pow(l,3)+720*pow(l,2)-284*l-210.))+(0.)*1j) + \
            self.index(psi,l+2,m+2)*(((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*(2*pow(m,2)+(1.-2*l)*m+2*pow(l,2)+4*l-3.))/(64*pow(l,5)+480*pow(l,4)+1120*pow(l,3)+720*pow(l,2)-284*l-210.))+(0.)*1j) + \
            self.index(psi,l+2,m+4)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(128*pow(l,5)+960*pow(l,4)+2240*pow(l,3)+1440*pow(l,2)-568*l-420.))+(0.)*1j) + \
            self.index(psi,l+4,m+-2)*((-(sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(128*pow(l,5)+1600*pow(l,4)+7360*pow(l,3)+15200*pow(l,2)+13512*l+3780.))+(0.)*1j) + \
            self.index(psi,l+4,m+-4)*(((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt((-m)+l+7.)*sqrt((-m)+l+8.))/(512*pow(l,5)+6400*pow(l,4)+29440*pow(l,3)+60800*pow(l,2)+54048*l+15120.))+(0.)*1j) + \
            self.index(psi,l+4,m+0)*(((3*sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))+(0.)*1j) + \
            self.index(psi,l+4,m+2)*((-(sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.))/(128*pow(l,5)+1600*pow(l,4)+7360*pow(l,3)+15200*pow(l,2)+13512*l+3780.))+(0.)*1j) + \
            self.index(psi,l+4,m+4)*(((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.)*sqrt(m+l+7.)*sqrt(m+l+8.))/(512*pow(l,5)+6400*pow(l,4)+29440*pow(l,3)+60800*pow(l,2)+54048*l+15120.))+(0.)*1j)
        
        
        return f
    
    
    def xxxz(self,psi,l,m):
    #Expansion in spherical harmonics of xxxz
    
    
        f= self.index(psi,l+-2,m+-1)*((-(sqrt(2*l-3.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(12*pow(m,2)+((-6*l)-9.)*m+6*pow(l,2)-3*l-21.))/(sqrt(2*l+1.)*(128*pow(l,4)-384*pow(l,3)-128*pow(l,2)+864*l-360.)))+(0.)*1j) + \
            self.index(psi,l+-2,m+-3)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l+1.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*m-2*l-5.))/(256*pow(l,5)-640*pow(l,4)-640*pow(l,3)+1600*pow(l,2)+144*l-360.))+(0.)*1j) + \
            self.index(psi,l+-2,m+1)*(((sqrt(2*l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l)*(12*pow(m,2)+(6*l+9.)*m+6*pow(l,2)-3*l-21.))/(sqrt(2*l+1.)*(128*pow(l,4)-384*pow(l,3)-128*pow(l,2)+864*l-360.)))+(0.)*1j) + \
            self.index(psi,l+-2,m+3)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*(4*m+2*l+5.))/(256*pow(l,5)-640*pow(l,4)-640*pow(l,3)+1600*pow(l,2)+144*l-360.))+(0.)*1j) + \
            self.index(psi,l+-4,m+-1)*(((3*sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))+(0.)*1j) + \
            self.index(psi,l+-4,m+-3)*((-(sqrt(2*l-7.)*sqrt(l-m)*sqrt(m+l-6.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(128*pow(l,4)-1024*pow(l,3)+2752*pow(l,2)-2816*l+840.)))+(0.)*1j) + \
            self.index(psi,l+-4,m+1)*((-(3*sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))+(0.)*1j) + \
            self.index(psi,l+-4,m+3)*(((sqrt(2*l-7.)*sqrt((-m)+l-6.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l))/(sqrt(2*l+1.)*(128*pow(l,4)-1024*pow(l,3)+2752*pow(l,2)-2816*l+840.)))+(0.)*1j) + \
            self.index(psi,l+0,m+-1)*((-(sqrt((-m)+l+1.)*sqrt(m+l)*(18*pow(m,3)-27*pow(m,2)+(6*pow(l,2)+6*l-27.)*m-3*pow(l,2)-3*l+18.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j) + \
            self.index(psi,l+0,m+-3)*(((sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(6*m-9.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j) + \
            self.index(psi,l+0,m+1)*((-(sqrt(l-m)*sqrt(m+l+1.)*(18*pow(m,3)+27*pow(m,2)+(6*pow(l,2)+6*l-27.)*m+3*pow(l,2)+3*l-18.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j) + \
            self.index(psi,l+0,m+3)*(((sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*(6*m+9.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j) + \
            self.index(psi,l+2,m+-1)*(((sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*(12*pow(m,2)+(6*l-3.)*m+6*pow(l,2)+15*l-12.))/(sqrt(2*l+1.)*(128*pow(l,4)+896*pow(l,3)+1792*pow(l,2)+544*l-840.)))+(0.)*1j) + \
            self.index(psi,l+2,m+-3)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt(m+l)*(4*m+2*l-3.))/(256*pow(l,5)+1920*pow(l,4)+4480*pow(l,3)+2880*pow(l,2)-1136*l-840.))+(0.)*1j) + \
            self.index(psi,l+2,m+1)*((-(sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*(12*pow(m,2)+(3.-6*l)*m+6*pow(l,2)+15*l-12.))/(sqrt(2*l+1.)*(128*pow(l,4)+896*pow(l,3)+1792*pow(l,2)+544*l-840.)))+(0.)*1j) + \
            self.index(psi,l+2,m+3)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*(4*m-2*l+3.))/(256*pow(l,5)+1920*pow(l,4)+4480*pow(l,3)+2880*pow(l,2)-1136*l-840.))+(0.)*1j) + \
            self.index(psi,l+4,m+-1)*((-(3*sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))+(0.)*1j) + \
            self.index(psi,l+4,m+-3)*(((sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt((-m)+l+7.)*sqrt(m+l+1.))/(sqrt(2*l+1.)*(128*pow(l,4)+1536*pow(l,3)+6592*pow(l,2)+11904*l+7560.)))+(0.)*1j) + \
            self.index(psi,l+4,m+1)*(((3*sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))+(0.)*1j) + \
            self.index(psi,l+4,m+3)*((-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.)*sqrt(m+l+7.))/(sqrt(2*l+1.)*(128*pow(l,4)+1536*pow(l,3)+6592*pow(l,2)+11904*l+7560.)))+(0.)*1j)
        
        return f
    
    def xxyz(self,psi,l,m):
    #Expansion in spherical harmonics of xxyz
    
    
        f= self.index(psi,l+-2,m+-1)*((0.)+((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*pow(m,2)+((-2*l)-3.)*m+2*pow(l,2)-l-7.))/(256*pow(l,5)-640*pow(l,4)-640*pow(l,3)+1600*pow(l,2)+144*l-360.))*1j) + \
            self.index(psi,l+-2,m+-3)*((0.)+((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l+1.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*m-2*l-5.))/(256*pow(l,5)-640*pow(l,4)-640*pow(l,3)+1600*pow(l,2)+144*l-360.))*1j) + \
            self.index(psi,l+-2,m+1)*((0.)+((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l)*(4*pow(m,2)+(2*l+3.)*m+2*pow(l,2)-l-7.))/(256*pow(l,5)-640*pow(l,4)-640*pow(l,3)+1600*pow(l,2)+144*l-360.))*1j) + \
            self.index(psi,l+-2,m+3)*((0.)+(-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*(4*m+2*l+5.))/(256*pow(l,5)-640*pow(l,4)-640*pow(l,3)+1600*pow(l,2)+144*l-360.))*1j) + \
            self.index(psi,l+-4,m+-1)*((0.)+(-(sqrt(2*l-7.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(128*pow(l,4)-1024*pow(l,3)+2752*pow(l,2)-2816*l+840.)))*1j) + \
            self.index(psi,l+-4,m+-3)*((0.)+((sqrt(2*l-7.)*sqrt(l-m)*sqrt(m+l-6.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(128*pow(l,4)-1024*pow(l,3)+2752*pow(l,2)-2816*l+840.)))*1j) + \
            self.index(psi,l+-4,m+1)*((0.)+(-(sqrt(2*l-7.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(128*pow(l,4)-1024*pow(l,3)+2752*pow(l,2)-2816*l+840.)))*1j) + \
            self.index(psi,l+-4,m+3)*((0.)+((sqrt(2*l-7.)*sqrt((-m)+l-6.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l))/(sqrt(2*l+1.)*(128*pow(l,4)-1024*pow(l,3)+2752*pow(l,2)-2816*l+840.)))*1j) + \
            self.index(psi,l+0,m+-1)*((0.)+((sqrt((-m)+l+1.)*sqrt(m+l)*(6*pow(m,3)-9*pow(m,2)+(2*pow(l,2)+2*l-9.)*m-pow(l,2)-l+6.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j) + \
            self.index(psi,l+0,m+-3)*((0.)+(-(sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(6*m-9.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j) + \
            self.index(psi,l+0,m+1)*((0.)+(-(sqrt(l-m)*sqrt(m+l+1.)*(6*pow(m,3)+9*pow(m,2)+(2*pow(l,2)+2*l-9.)*m+pow(l,2)+l-6.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j) + \
            self.index(psi,l+0,m+3)*((0.)+((sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*(6*m+9.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j) + \
            self.index(psi,l+2,m+-1)*((0.)+(-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*(4*pow(m,2)+(2*l-1.)*m+2*pow(l,2)+5*l-4.))/(256*pow(l,5)+1920*pow(l,4)+4480*pow(l,3)+2880*pow(l,2)-1136*l-840.))*1j) + \
            self.index(psi,l+2,m+-3)*((0.)+((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt(m+l)*(4*m+2*l-3.))/(256*pow(l,5)+1920*pow(l,4)+4480*pow(l,3)+2880*pow(l,2)-1136*l-840.))*1j) + \
            self.index(psi,l+2,m+1)*((0.)+(-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*(4*pow(m,2)+(1.-2*l)*m+2*pow(l,2)+5*l-4.))/(256*pow(l,5)+1920*pow(l,4)+4480*pow(l,3)+2880*pow(l,2)-1136*l-840.))*1j) + \
            self.index(psi,l+2,m+3)*((0.)+(-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*(4*m-2*l+3.))/(256*pow(l,5)+1920*pow(l,4)+4480*pow(l,3)+2880*pow(l,2)-1136*l-840.))*1j) + \
            self.index(psi,l+4,m+-1)*((0.)+((sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(sqrt(2*l+1.)*(128*pow(l,4)+1536*pow(l,3)+6592*pow(l,2)+11904*l+7560.)))*1j) + \
            self.index(psi,l+4,m+-3)*((0.)+(-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt((-m)+l+7.)*sqrt(m+l+1.))/(sqrt(2*l+1.)*(128*pow(l,4)+1536*pow(l,3)+6592*pow(l,2)+11904*l+7560.)))*1j) + \
            self.index(psi,l+4,m+1)*((0.)+((sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.))/(sqrt(2*l+1.)*(128*pow(l,4)+1536*pow(l,3)+6592*pow(l,2)+11904*l+7560.)))*1j) + \
            self.index(psi,l+4,m+3)*((0.)+(-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.)*sqrt(m+l+7.))/(sqrt(2*l+1.)*(128*pow(l,4)+1536*pow(l,3)+6592*pow(l,2)+11904*l+7560.)))*1j)
        
        return f
    
    def xy(self,psi,l,m):
    #Expansion in spherical harmonics of xy
    
    
        
        f=  self.index(psi,l-2,m-2)*((0.)+(-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(32*pow(l,3)-48*pow(l,2)-8*l+12.))*1j) + \
            self.index(psi,l-2,m+2)*((0.)+((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(32*pow(l,3)-48*pow(l,2)-8*l+12.))*1j) + \
            self.index(psi,l,m-2)*((0.)+((sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l))/(8*pow(l,2)+8*l-6.))*1j) + \
            self.index(psi,l,m+2)*((0.)+(-(sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(8*pow(l,2)+8*l-6.))*1j) + \
            self.index(psi,l+2,m-2)*((0.)+(-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.))/(32*pow(l,3)+144*pow(l,2)+184*l+60.))*1j) + \
            self.index(psi,l+2,m+2)*((0.)+((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(32*pow(l,3)+144*pow(l,2)+184*l+60.))*1j)
        
        
        return f
    
    def xyyz(self,psi,l,m):
    #Expansion in spherical harmonics of xyyz
    
    
        f=self.index(psi,l+-2,m+-1)*( (-(sqrt(2*l-3.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*pow(m,2)+((-2*l)-3.)*m+2*pow(l,2)-l-7.))/(sqrt(2*l+1.)*(128*pow(l,4)-384*pow(l,3)-128*pow(l,2)+864*l-360.)))+(0.)*1j )+ \
            self.index(psi,l+-2,m+-3)*( ((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l+1.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*m-2*l-5.))/(256*pow(l,5)-640*pow(l,4)-640*pow(l,3)+1600*pow(l,2)+144*l-360.))+(0.)*1j )+ \
            self.index(psi,l+-2,m+1)*( ((sqrt(2*l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l)*(4*pow(m,2)+(2*l+3.)*m+2*pow(l,2)-l-7.))/(sqrt(2*l+1.)*(128*pow(l,4)-384*pow(l,3)-128*pow(l,2)+864*l-360.)))+(0.)*1j )+ \
            self.index(psi,l+-2,m+3)*( ((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*(4*m+2*l+5.))/(256*pow(l,5)-640*pow(l,4)-640*pow(l,3)+1600*pow(l,2)+144*l-360.))+(0.)*1j )+ \
            self.index(psi,l+-4,m+-1)*( ((sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))+(0.)*1j )+ \
            self.index(psi,l+-4,m+-3)*( ((sqrt(2*l-7.)*sqrt(l-m)*sqrt(m+l-6.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(128*pow(l,4)-1024*pow(l,3)+2752*pow(l,2)-2816*l+840.)))+(0.)*1j )+ \
            self.index(psi,l+-4,m+1)*( (-(sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))+(0.)*1j )+ \
            self.index(psi,l+-4,m+3)*( (-(sqrt(2*l-7.)*sqrt((-m)+l-6.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l))/(sqrt(2*l+1.)*(128*pow(l,4)-1024*pow(l,3)+2752*pow(l,2)-2816*l+840.)))+(0.)*1j )+ \
            self.index(psi,l+0,m+-1)*( (-(sqrt((-m)+l+1.)*sqrt(m+l)*(6*pow(m,3)-9*pow(m,2)+(2*pow(l,2)+2*l-9.)*m-pow(l,2)-l+6.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j )+ \
            self.index(psi,l+0,m+-3)*( (-(sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(6*m-9.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j )+ \
            self.index(psi,l+0,m+1)*( (-(sqrt(l-m)*sqrt(m+l+1.)*(6*pow(m,3)+9*pow(m,2)+(2*pow(l,2)+2*l-9.)*m+pow(l,2)+l-6.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j )+ \
            self.index(psi,l+0,m+3)*( (-(sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*(6*m+9.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))+(0.)*1j )+ \
            self.index(psi,l+2,m+-1)*( ((sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*(4*pow(m,2)+(2*l-1.)*m+2*pow(l,2)+5*l-4.))/(sqrt(2*l+1.)*(128*pow(l,4)+896*pow(l,3)+1792*pow(l,2)+544*l-840.)))+(0.)*1j )+ \
            self.index(psi,l+2,m+-3)*( ((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt(m+l)*(4*m+2*l-3.))/(256*pow(l,5)+1920*pow(l,4)+4480*pow(l,3)+2880*pow(l,2)-1136*l-840.))+(0.)*1j )+ \
            self.index(psi,l+2,m+1)*( (-(sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*(4*pow(m,2)+(1.-2*l)*m+2*pow(l,2)+5*l-4.))/(sqrt(2*l+1.)*(128*pow(l,4)+896*pow(l,3)+1792*pow(l,2)+544*l-840.)))+(0.)*1j )+ \
            self.index(psi,l+2,m+3)*( ((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*(4*m-2*l+3.))/(256*pow(l,5)+1920*pow(l,4)+4480*pow(l,3)+2880*pow(l,2)-1136*l-840.))+(0.)*1j )+ \
            self.index(psi,l+4,m+-1)*( (-(sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))+(0.)*1j )+ \
            self.index(psi,l+4,m+-3)*( (-(sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt((-m)+l+7.)*sqrt(m+l+1.))/(sqrt(2*l+1.)*(128*pow(l,4)+1536*pow(l,3)+6592*pow(l,2)+11904*l+7560.)))+(0.)*1j )+ \
            self.index(psi,l+4,m+1)*( ((sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))+(0.)*1j )+ \
            self.index(psi,l+4,m+3)*( ((sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.)*sqrt(m+l+7.))/(sqrt(2*l+1.)*(128*pow(l,4)+1536*pow(l,3)+6592*pow(l,2)+11904*l+7560.)))+(0.)*1j )
        
        
        return f
    
    def xz(self,psi,l,m):
    #Expansion in spherical harmonics of xz
    
        f=  self.index(psi,l-2,m-1)*((-(sqrt(2*l-3.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(8*pow(l,2)-16*l+6.)))+(0.)*1j) + \
            self.index(psi,l-2,m+1)*(((sqrt(2*l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l))/(sqrt(2*l+1.)*(8*pow(l,2)-16*l+6.)))+(0.)*1j) + \
            self.index(psi,l,m-1)*((-(sqrt((-m)+l+1.)*sqrt(m+l)*(2*m-1.))/(8*pow(l,2)+8*l-6.))+(0.)*1j) + \
            self.index(psi,l,m+1)*((-(sqrt(l-m)*sqrt(m+l+1.)*(2*m+1.))/(8*pow(l,2)+8*l-6.))+(0.)*1j) + \
            self.index(psi,l+2,m-1)*(((sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.))/(sqrt(2*l+1.)*(8*pow(l,2)+32*l+30.)))+(0.)*1j) + \
            self.index(psi,l+2,m+1)*((-(sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(sqrt(2*l+1.)*(8*pow(l,2)+32*l+30.)))+(0.)*1j)
        
        
        return f
    
    def yy(self,psi,l,m):
    #Expansion in spherical harmonics of yy
    
    
        
        f=  self.index(psi,l-2,m-2)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(32*pow(l,3)-48*pow(l,2)-8*l+12.))+(0.)*1j) + \
            self.index(psi,l-2,m)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(16*pow(l,3)-24*pow(l,2)-4*l+6.))+(0.)*1j) + \
            self.index(psi,l-2,m+2)*((-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(32*pow(l,3)-48*pow(l,2)-8*l+12.))+(0.)*1j) + \
            self.index(psi,l,m-2)*(((sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l))/(8*pow(l,2)+8*l-6.))+(0.)*1j) + \
            self.index(psi,l,m)*(((pow(m,2)+pow(l,2)+l-1.)/(4*pow(l,2)+4*l-3.))+(0.)*1j) + \
            self.index(psi,l,m+2)*(((sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(8*pow(l,2)+8*l-6.))+(0.)*1j) + \
            self.index(psi,l+2,m-2)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.))/(32*pow(l,3)+144*pow(l,2)+184*l+60.))+(0.)*1j) + \
            self.index(psi,l+2,m)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(16*pow(l,3)+72*pow(l,2)+92*l+30.))+(0.)*1j) + \
            self.index(psi,l+2,m+2)*((-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(32*pow(l,3)+144*pow(l,2)+184*l+60.))+(0.)*1j)
        
        
        return f
    
    def yyyz(self,psi,l,m):
    #Expansion in spherical harmonics of yyyz
    
    
        f=self.index(psi,l+-2,m+-1)*( (0.)+((sqrt(2*l-3.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(12*pow(m,2)+((-6*l)-9.)*m+6*pow(l,2)-3*l-21.))/(sqrt(2*l+1.)*(128*pow(l,4)-384*pow(l,3)-128*pow(l,2)+864*l-360.)))*1j )+ \
            self.index(psi,l+-2,m+-3)*( (0.)+(-(sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l+1.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(4*m-2*l-5.))/(256*pow(l,5)-640*pow(l,4)-640*pow(l,3)+1600*pow(l,2)+144*l-360.))*1j )+ \
            self.index(psi,l+-2,m+1)*( (0.)+((sqrt(2*l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l)*(12*pow(m,2)+(6*l+9.)*m+6*pow(l,2)-3*l-21.))/(sqrt(2*l+1.)*(128*pow(l,4)-384*pow(l,3)-128*pow(l,2)+864*l-360.)))*1j )+ \
            self.index(psi,l+-2,m+3)*( (0.)+((sqrt(2*l-3.)*sqrt(2*l+1.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*(4*m+2*l+5.))/(256*pow(l,5)-640*pow(l,4)-640*pow(l,3)+1600*pow(l,2)+144*l-360.))*1j )+ \
            self.index(psi,l+-4,m+-1)*( (0.)+(-(3*sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))*1j )+ \
            self.index(psi,l+-4,m+-3)*( (0.)+(-(sqrt(2*l-7.)*sqrt(l-m)*sqrt(m+l-6.)*sqrt(m+l-5.)*sqrt(m+l-4.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(128*pow(l,4)-1024*pow(l,3)+2752*pow(l,2)-2816*l+840.)))*1j )+ \
            self.index(psi,l+-4,m+1)*( (0.)+(-(3*sqrt(2*l-7.)*sqrt(2*l+1.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(256*pow(l,5)-1920*pow(l,4)+4480*pow(l,3)-2880*pow(l,2)-1136*l+840.))*1j )+ \
            self.index(psi,l+-4,m+3)*( (0.)+(-(sqrt(2*l-7.)*sqrt((-m)+l-6.)*sqrt((-m)+l-5.)*sqrt((-m)+l-4.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l))/(sqrt(2*l+1.)*(128*pow(l,4)-1024*pow(l,3)+2752*pow(l,2)-2816*l+840.)))*1j )+ \
            self.index(psi,l+0,m+-1)*( (0.)+((sqrt((-m)+l+1.)*sqrt(m+l)*(18*pow(m,3)-27*pow(m,2)+(6*pow(l,2)+6*l-27.)*m-3*pow(l,2)-3*l+18.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j )+ \
            self.index(psi,l+0,m+-3)*( (0.)+((sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l)*(6*m-9.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j )+ \
            self.index(psi,l+0,m+1)*( (0.)+(-(sqrt(l-m)*sqrt(m+l+1.)*(18*pow(m,3)+27*pow(m,2)+(6*pow(l,2)+6*l-27.)*m+3*pow(l,2)+3*l-18.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j )+ \
            self.index(psi,l+0,m+3)*( (0.)+(-(sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*(6*m+9.))/(128*pow(l,4)+256*pow(l,3)-448*pow(l,2)-576*l+360.))*1j )+ \
            self.index(psi,l+2,m+-1)*( (0.)+(-(sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*(12*pow(m,2)+(6*l-3.)*m+6*pow(l,2)+15*l-12.))/(sqrt(2*l+1.)*(128*pow(l,4)+896*pow(l,3)+1792*pow(l,2)+544*l-840.)))*1j )+ \
            self.index(psi,l+2,m+-3)*( (0.)+(-(sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt(m+l)*(4*m+2*l-3.))/(256*pow(l,5)+1920*pow(l,4)+4480*pow(l,3)+2880*pow(l,2)-1136*l-840.))*1j )+ \
            self.index(psi,l+2,m+1)*( (0.)+(-(sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*(12*pow(m,2)+(3.-6*l)*m+6*pow(l,2)+15*l-12.))/(sqrt(2*l+1.)*(128*pow(l,4)+896*pow(l,3)+1792*pow(l,2)+544*l-840.)))*1j )+ \
            self.index(psi,l+2,m+3)*( (0.)+((sqrt(2*l+1.)*sqrt(2*l+5.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*(4*m-2*l+3.))/(256*pow(l,5)+1920*pow(l,4)+4480*pow(l,3)+2880*pow(l,2)-1136*l-840.))*1j )+ \
            self.index(psi,l+4,m+-1)*( (0.)+((3*sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))*1j )+ \
            self.index(psi,l+4,m+-3)*( (0.)+((sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.)*sqrt((-m)+l+5.)*sqrt((-m)+l+6.)*sqrt((-m)+l+7.)*sqrt(m+l+1.))/(sqrt(2*l+1.)*(128*pow(l,4)+1536*pow(l,3)+6592*pow(l,2)+11904*l+7560.)))*1j )+ \
            self.index(psi,l+4,m+1)*( (0.)+((3*sqrt(2*l+1.)*sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.))/(256*pow(l,5)+3200*pow(l,4)+14720*pow(l,3)+30400*pow(l,2)+27024*l+7560.))*1j )+ \
            self.index(psi,l+4,m+3)*( (0.)+((sqrt(2*l+9.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.)*sqrt(m+l+5.)*sqrt(m+l+6.)*sqrt(m+l+7.))/(sqrt(2*l+1.)*(128*pow(l,4)+1536*pow(l,3)+6592*pow(l,2)+11904*l+7560.)))*1j )
        
        return f
    
    def yz(self,psi,l,m):
    #Expansion in spherical harmonics of xz
    
    
        
        f=  self.index(psi,l-2,m-1)*((0.)+((sqrt(2*l-3.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l+1.)*(8*pow(l,2)-16*l+6.)))*1j) + \
            self.index(psi,l-2,m+1)*((0.)+((sqrt(2*l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l))/(sqrt(2*l+1.)*(8*pow(l,2)-16*l+6.)))*1j) + \
            self.index(psi,l,m-1)*((0.)+((sqrt((-m)+l+1.)*sqrt(m+l)*(2*m-1.))/(8*pow(l,2)+8*l-6.))*1j) + \
            self.index(psi,l,m+1)*((0.)+(-(sqrt(l-m)*sqrt(m+l+1.)*(2*m+1.))/(8*pow(l,2)+8*l-6.))*1j) + \
            self.index(psi,l+2,m-1)*((0.)+(-(sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.))/(sqrt(2*l+1.)*(8*pow(l,2)+32*l+30.)))*1j) + \
            self.index(psi,l+2,m+1)*((0.)+(-(sqrt(2*l+5.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(sqrt(2*l+1.)*(8*pow(l,2)+32*l+30.)))*1j)
        
        
        return f
    
    def zz(self,psi,l,m):
    #Expansion in spherical harmonics of zz
    
    
        
        f=  self.index(psi,l-2,m)*(((sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2*l-3.)*(2*l-1.)*sqrt(2*l+1.)))+(0.)*1j) + \
            self.index(psi,l,m)*((-(2*pow(m,2)-2*pow(l,2)-2*l+1.)/(4*pow(l,2)+4*l-3.))+(0.)*1j) + \
            self.index(psi,l+2,m)*(((sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(sqrt(2*l+1.)*(2*l+3.)*sqrt(2*l+5.)))+(0.)*1j)
            
        
        return f
    
    def xdx(self,psi,l,m):
        
        f=  self.index(psi,l+-2,m+-2)*((((l+1.)*sqrt(2.*l-3.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2.*l+1.)*(16.*pow(l,2)-32.*l+12.)))+(0.)*1j)+\
            self.index(psi,l+-2,m+0)*((-((l+1.)*sqrt(2.*l-3.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2.*l+1.)*(8.*pow(l,2)-16.*l+6.)))+(0.)*1j)+\
            self.index(psi,l+-2,m+2)*((((l+1.)*sqrt(2.*l-3.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(sqrt(2.*l+1.)*(16.*pow(l,2)-32.*l+12.)))+(0.)*1j)+\
            self.index(psi,l+0,m+-2)*((-(3.*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l))/(16.*pow(l,2)+16.*l-12.))+(0.)*1j)+\
            self.index(psi,l+0,m+0)*(((3.*pow(m,2)-pow(l,2)-l)/(8.*pow(l,2)+8.*l-6.))+(0.)*1j)+\
            self.index(psi,l+0,m+2)*((-(3.*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(16.*pow(l,2)+16.*l-12.))+(0.)*1j)+\
            self.index(psi,l+2,m+-2)*((-(l*sqrt(2.*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.))/(sqrt(2.*l+1.)*(16.*pow(l,2)+64.*l+60.)))+(0.)*1j)+\
            self.index(psi,l+2,m+0)*(((l*sqrt(2.*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(sqrt(2.*l+1.)*(8.*pow(l,2)+32.*l+30.)))+(0.)*1j)+\
            self.index(psi,l+2,m+2)*((-(l*sqrt(2.*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(sqrt(2.*l+1.)*(16.*pow(l,2)+64.*l+60.)))+(0.)*1j)
            
        return f
    
    def xdy(self,psi,l,m):
        
        f=  self.index(psi,l+-2,m+-2)*((0.)+(-((l+1.)*sqrt(2.*l-3.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2.*l+1.)*(16.*pow(l,2)-32.*l+12.)))*1j)+\
            self.index(psi,l+-2,m+2)*((0.)+(((l+1.)*sqrt(2.*l-3.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(sqrt(2.*l+1.)*(16.*pow(l,2)-32.*l+12.)))*1j)+\
            self.index(psi,l+0,m+-2)*((0.)+((3.*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l))/(16.*pow(l,2)+16.*l-12.))*1j)+\
            self.index(psi,l+0,m+0)*((0.)+(-m/2.)*1j)+\
            self.index(psi,l+0,m+2)*((0.)+(-(3.*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(16.*pow(l,2)+16.*l-12.))*1j)+\
            self.index(psi,l+2,m+-2)*((0.)+((l*sqrt(2.*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.))/(sqrt(2.*l+1.)*(16.*pow(l,2)+64.*l+60.)))*1j)+\
            self.index(psi,l+2,m+2)*((0.)+(-(l*sqrt(2.*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(sqrt(2.*l+1.)*(16.*pow(l,2)+64.*l+60.)))*1j)
            
        return f
    
    def xdz(self,psi,l,m):
        
        f=  self.index(psi,l+-2,m+-1)*((-((l+1.)*sqrt(2.*l-3.)*sqrt(2.*l+1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(16.*pow(l,3)-24.*pow(l,2)-4.*l+6.))+(0.)*1j)+\
            self.index(psi,l+-2,m+1)*((((l+1.)*sqrt(2.*l-3.)*sqrt(2.*l+1.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l))/(16.*pow(l,3)-24.*pow(l,2)-4.*l+6.))+(0.)*1j)+\
            self.index(psi,l+0,m+-1)*((-(sqrt((-m)+l+1.)*sqrt(m+l)*(3.*m-2.*pow(l,2)-2.*l))/(8.*pow(l,2)+8.*l-6.))+(0.)*1j)+\
            self.index(psi,l+0,m+1)*((-(sqrt(l-m)*sqrt(m+l+1.)*(3.*m+2.*pow(l,2)+2.*l))/(8.*pow(l,2)+8.*l-6.))+(0.)*1j)+\
            self.index(psi,l+2,m+-1)*((-(l*sqrt(2.*l+1.)*sqrt(2.*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.))/(16.*pow(l,3)+72.*pow(l,2)+92.*l+30.))+(0.)*1j)+\
            self.index(psi,l+2,m+1)*(((l*sqrt(2.*l+1.)*sqrt(2.*l+5.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(16.*pow(l,3)+72.*pow(l,2)+92.*l+30.))+(0.)*1j)

        return f
    
    def ydx(self,psi,l,m):
        
        f=  self.index(psi,l+-2,m+-2)*((0.)+(-((l+1.)*sqrt(2.*l-3.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2.*l+1.)*(16.*pow(l,2)-32.*l+12.)))*1j)+\
            self.index(psi,l+-2,m+2)*((0.)+(((l+1.)*sqrt(2.*l-3.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(sqrt(2.*l+1.)*(16.*pow(l,2)-32.*l+12.)))*1j)+\
            self.index(psi,l+0,m+-2)*((0.)+((3.*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l))/(16.*pow(l,2)+16.*l-12.))*1j)+\
            self.index(psi,l+0,m+0)*((0.)+(m/2.)*1j)+\
            self.index(psi,l+0,m+2)*((0.)+(-(3.*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(16.*pow(l,2)+16.*l-12.))*1j)+\
            self.index(psi,l+2,m+-2)*((0.)+((l*sqrt(2.*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.))/(sqrt(2.*l+1.)*(16.*pow(l,2)+64.*l+60.)))*1j)+\
            self.index(psi,l+2,m+2)*((0.)+(-(l*sqrt(2.*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(sqrt(2.*l+1.)*(16.*pow(l,2)+64.*l+60.)))*1j)
               
        return f
    
    def ydy(self,psi,l,m):
        
        f=  self.index(psi,l+-2,m+-2)*((-((l+1.)*sqrt(2.*l-3.)*sqrt(m+l-3.)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2.*l+1.)*(16.*pow(l,2)-32.*l+12.)))+(0.)*1j)+\
            self.index(psi,l+-2,m+0)*((-((l+1.)*sqrt(2.*l-3.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2.*l+1.)*(8.*pow(l,2)-16.*l+6.)))+(0.)*1j)+\
            self.index(psi,l+-2,m+2)*((-((l+1.)*sqrt(2.*l-3.)*sqrt((-m)+l-3.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m))/(sqrt(2.*l+1.)*(16.*pow(l,2)-32.*l+12.)))+(0.)*1j)+\
            self.index(psi,l+0,m+-2)*(((3.*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l-1.)*sqrt(m+l))/(16.*pow(l,2)+16.*l-12.))+(0.)*1j)+\
            self.index(psi,l+0,m+0)*(((3.*pow(m,2)-pow(l,2)-l)/(8.*pow(l,2)+8.*l-6.))+(0.)*1j)+\
            self.index(psi,l+0,m+2)*(((3.*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l+1.)*sqrt(m+l+2.))/(16.*pow(l,2)+16.*l-12.))+(0.)*1j)+\
            self.index(psi,l+2,m+-2)*(((l*sqrt(2.*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt((-m)+l+4.))/(sqrt(2.*l+1.)*(16.*pow(l,2)+64.*l+60.)))+(0.)*1j)+\
            self.index(psi,l+2,m+0)*(((l*sqrt(2.*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(sqrt(2.*l+1.)*(8.*pow(l,2)+32.*l+30.)))+(0.)*1j)+\
            self.index(psi,l+2,m+2)*(((l*sqrt(2.*l+5.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.)*sqrt(m+l+4.))/(sqrt(2.*l+1.)*(16.*pow(l,2)+64.*l+60.)))+(0.)*1j)
                        
        return f
    
    def ydz(self,psi,l,m):
        
        f=  self.index(psi,l+-2,m+-1)*((0.)+(((l+1.)*sqrt(2.*l-3.)*sqrt(2.*l+1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(16.*pow(l,3)-24.*pow(l,2)-4.*l+6.))*1j)+\
            self.index(psi,l+-2,m+1)*((0.)+(((l+1.)*sqrt(2.*l-3.)*sqrt(2.*l+1.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l))/(16.*pow(l,3)-24.*pow(l,2)-4.*l+6.))*1j)+\
            self.index(psi,l+0,m+-1)*((0.)+((sqrt((-m)+l+1.)*sqrt(m+l)*(3.*m-2.*pow(l,2)-2.*l))/(8.*pow(l,2)+8.*l-6.))*1j)+\
            self.index(psi,l+0,m+1)*((0.)+(-(sqrt(l-m)*sqrt(m+l+1.)*(3.*m+2.*pow(l,2)+2.*l))/(8.*pow(l,2)+8.*l-6.))*1j)+\
            self.index(psi,l+2,m+-1)*((0.)+((l*sqrt(2.*l+1.)*sqrt(2.*l+5.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.))/(16.*pow(l,3)+72.*pow(l,2)+92.*l+30.))*1j)+\
            self.index(psi,l+2,m+1)*((0.)+((l*sqrt(2.*l+1.)*sqrt(2.*l+5.)*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(16.*pow(l,3)+72.*pow(l,2)+92.*l+30.))*1j)
                    
        return f
    
    def zdx(self,psi,l,m):
        
        f=  self.index(psi,l+-2,m+-1)*((-((l+1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2.*l-3.)*sqrt(2.*l+1.)*(4.*l-2.)))+(0.)*1j)+\
            self.index(psi,l+-2,m+1)*((((l+1.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l))/(sqrt(2.*l-3.)*sqrt(2.*l+1.)*(4.*l-2.)))+(0.)*1j)+\
            self.index(psi,l+0,m+-1)*((-(sqrt((-m)+l+1.)*sqrt(m+l)*(3.*m+2.*pow(l,2)+2.*l-3.))/(8.*pow(l,2)+8.*l-6.))+(0.)*1j)+\
            self.index(psi,l+0,m+1)*((-(sqrt(l-m)*sqrt(m+l+1.)*(3.*m-2.*pow(l,2)-2.*l+3.))/(8.*pow(l,2)+8.*l-6.))+(0.)*1j)+\
            self.index(psi,l+2,m+-1)*((-(l*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.))/(sqrt(2.*l+1.)*sqrt(2.*l+5.)*(4.*l+6.)))+(0.)*1j)+\
            self.index(psi,l+2,m+1)*(((l*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(sqrt(2.*l+1.)*sqrt(2.*l+5.)*(4.*l+6.)))+(0.)*1j)
 
        return f
    
    def zdy(self,psi,l,m):
        
        f=  self.index(psi,l+-2,m+-1)*((0.)+(((l+1.)*sqrt(l-m)*sqrt(m+l-2.)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2.*l-3.)*sqrt(2.*l+1.)*(4.*l-2.)))*1j)+\
            self.index(psi,l+-2,m+1)*((0.)+(((l+1.)*sqrt((-m)+l-2.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l))/(sqrt(2.*l-3.)*sqrt(2.*l+1.)*(4.*l-2.)))*1j)+\
            self.index(psi,l+0,m+-1)*((0.)+((sqrt((-m)+l+1.)*sqrt(m+l)*(3.*m+2.*pow(l,2)+2.*l-3.))/(8.*pow(l,2)+8.*l-6.))*1j)+\
            self.index(psi,l+0,m+1)*((0.)+(-(sqrt(l-m)*sqrt(m+l+1.)*(3.*m-2.*pow(l,2)-2.*l+3.))/(8.*pow(l,2)+8.*l-6.))*1j)+\
            self.index(psi,l+2,m+-1)*((0.)+((l*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt((-m)+l+3.)*sqrt(m+l+1.))/(sqrt(2.*l+1.)*sqrt(2.*l+5.)*(4.*l+6.)))*1j)+\
            self.index(psi,l+2,m+1)*((0.)+((l*sqrt((-m)+l+1.)*sqrt(m+l+1.)*sqrt(m+l+2.)*sqrt(m+l+3.))/(sqrt(2.*l+1.)*sqrt(2.*l+5.)*(4.*l+6.)))*1j)
                    
        return f

    def zdz(self,psi,l,m):
        
        f=  self.index(psi,l+-2,m+0)*((((l+1.)*sqrt(2.*l+1.)*sqrt((-m)+l-1.)*sqrt(l-m)*sqrt(m+l-1.)*sqrt(m+l))/(sqrt(2.*l-3.)*(4.*pow(l,2)-1.)))+(0.)*1j)+\
            self.index(psi,l+0,m+0)*((-(3.*pow(m,2)-pow(l,2)-l)/(4.*pow(l,2)+4.*l-3.))+(0.)*1j)+\
            self.index(psi,l+2,m+0)*((-(l*sqrt(2.*l+1.)*sqrt((-m)+l+1.)*sqrt((-m)+l+2.)*sqrt(m+l+1.)*sqrt(m+l+2.))/(sqrt(2.*l+5.)*(4.*pow(l,2)+8.*l+3.)))+(0.)*1j)

        return f   
    
    def a2(self,f):
        if f.ndim==2:
            a=np.zeros((3,3,f.shape[1]),dtype=complex)
        else:
            a=np.zeros((3,3),dtype=complex)  
        a[0,0]=((0.33333333333333333333)+(0)*1j)*f[self.idx(0,0)]+ ((-0.14907119849998597976)+(0)*1j)*f[self.idx(2,0)]+ 2*((0.18257418583505537115)+(0)*1j)*f[self.idx(2,2)]
        a[0,1]=2*((0)+(0.18257418583505537115)*1j)*f[self.idx(2,2)]
        a[1,0]=a[0,1]
        a[0,2]=2*((-0.18257418583505537115)+(0)*1j)*f[self.idx(2,1)]
        a[2,0]=a[0,2]
        a[1,1]=((0.33333333333333333333)+(0)*1j)*f[self.idx(0,0)]+ ((-0.14907119849998597976)+(0)*1j)*f[self.idx(2,0)]+ 2*((-0.18257418583505537115)+(0)*1j)*f[self.idx(2,2)]
        a[1,2]=2*((0)+(-0.18257418583505537115)*1j)*f[self.idx(2,1)]
        a[2,1]=a[1,2]
        a[2,2]=((0.33333333333333333333)+(0)*1j)*f[self.idx(0,0)]+ ((0.29814239699997195952)+(0)*1j)*f[self.idx(2,0)]
        
        #a=a*sqrt(4*pi)
        return a
    
    def J(self,f):
        if f.ndim==2:
            J=np.zeros(f.shape[1])
            Sff = np.zeros(f.shape[1])
        else:
            J=0
            Sff=0
        
        for l in range(0,self.lmax+1,2):
            Sff = 0*Sff
            for m in range(0,l+1,1):
                Sff=Sff+np.abs(f[self.idx(l,abs(m))])**2
            J=J+Sff
        return J


        
        
    
    
    def a4(self,f):
        if f.ndim==2:
             a=np.zeros((3,3,3,3,f.shape[1]),dtype=complex)
        else:
             a=np.zeros((3,3,3,3),dtype=complex)
        a[0,0,0,0]=((0.2)+(0)*1j)*f[self.idx(0,0)]+ ((-0.12777531299998798265)+(0)*1j)*f[self.idx(2,0)]+ 2*((0.15649215928719031813)+(0)*1j)*f[self.idx(2,2)]\
            + ((0.028571428571428571429)+(0)*1j)*f[self.idx(4,0)]+ 2*((-0.030116930096841707924)+(0)*1j)*f[self.idx(4,2)]+ 2*((0.039840953644479787999)+(0)*1j)*f[self.idx(4,4)]
        
        a[1,0,0,0]=2*((0)+(0.078246079643595159065)*1j)*f[self.idx(2,2)]+ 2*((0)+(-0.015058465048420853962)*1j)*f[self.idx(4,2)]+ 2*((0)+(0.039840953644479787999)*1j)*f[self.idx(4,4)]
        a[0,1,0,0]=a[1,0,0,0]
        a[0,0,1,0]=a[1,0,0,0]
        a[0,0,0,1]=a[1,0,0,0]
        
        a[2,0,0,0]=2*((-0.078246079643595159065)+(0)*1j)*f[self.idx(2,1)]+ 2*((0.031943828249996995663)+(0)*1j)*f[self.idx(4,1)]+ 2*((-0.028171808490950552584)+(0)*1j)*f[self.idx(4,3)]
        a[0,2,0,0]=a[2,0,0,0]
        a[0,0,2,0]=a[2,0,0,0]
        a[0,0,0,2]=a[2,0,0,0]
        
        a[1,1,0,0]=((0.066666666666666666667)+(0)*1j)*f[self.idx(0,0)]+ ((-0.042591770999995994217)+(0)*1j)*f[self.idx(2,0)]+ ((0.0095238095238095238095)+(0)*1j)*f[self.idx(4,0)]+ 2*((-0.039840953644479787999)+(0)*1j)*f[self.idx(4,4)]
        a[1,0,1,0]=a[1,1,0,0]
        a[1,0,0,1]=a[1,1,0,0]
        a[0,1,1,0]=a[1,1,0,0]
        a[0,1,0,1]=a[1,1,0,0]
        a[0,0,1,1]=a[1,1,0,0]
        
        a[2,1,0,0]=2*((0)+(-0.026082026547865053022)*1j)*f[self.idx(2,1)]+ 2*((0)+(0.010647942749998998554)*1j)*f[self.idx(4,1)]+ 2*((0)+(-0.028171808490950552584)*1j)*f[self.idx(4,3)]
        a[2,0,1,0]=a[2,1,0,0]
        a[2,0,0,1]=a[2,1,0,0]
        a[1,2,0,0]=a[2,1,0,0]
        a[1,0,2,0]=a[2,1,0,0]
        a[1,0,0,2]=a[2,1,0,0]
        a[0,2,1,0]=a[2,1,0,0]
        a[0,2,0,1]=a[2,1,0,0]
        a[0,1,2,0]=a[2,1,0,0]
        a[0,1,0,2]=a[2,1,0,0]
        a[0,0,2,1]=a[2,1,0,0]
        a[0,0,1,2]=a[2,1,0,0]
        
        a[2,2,0,0]=((0.066666666666666666667)+(0)*1j)*f[self.idx(0,0)]+ ((0.021295885499997997109)+(0)*1j)*f[self.idx(2,0)]+ 2*((0.026082026547865053022)+(0)*1j)*f[self.idx(2,2)]\
            + ((-0.038095238095238095238)+(0)*1j)*f[self.idx(4,0)]+ 2*((0.030116930096841707924)+(0)*1j)*f[self.idx(4,2)]
        a[2,0,2,0]=a[2,2,0,0]
        a[2,0,0,2]=a[2,2,0,0]
        a[0,2,2,0]=a[2,2,0,0]
        a[0,2,0,2]=a[2,2,0,0]
        a[0,0,2,2]=a[2,2,0,0]
        
        a[1,1,1,0]=2*((0)+(0.078246079643595159065)*1j)*f[self.idx(2,2)]+ 2*((0)+(-0.015058465048420853962)*1j)*f[self.idx(4,2)]+ 2*((0)+(-0.039840953644479787999)*1j)*f[self.idx(4,4)]
        a[1,1,0,1]=a[1,1,1,0]
        a[1,0,1,1]=a[1,1,1,0]
        a[0,1,1,1]=a[1,1,1,0]
        
        a[2,1,1,0]=2*((-0.026082026547865053022)+(0)*1j)*f[self.idx(2,1)]+ 2*((0.010647942749998998554)+(0)*1j)*f[self.idx(4,1)]+ 2*((0.028171808490950552584)+(0)*1j)*f[self.idx(4,3)]
        a[2,1,0,1]=a[2,1,1,0]
        a[2,0,1,1]=a[2,1,1,0]
        a[1,2,1,0]=a[2,1,1,0]
        a[1,2,0,1]=a[2,1,1,0]
        a[1,1,2,0]=a[2,1,1,0]
        a[1,1,0,2]=a[2,1,1,0]
        a[1,0,2,1]=a[2,1,1,0]
        a[1,0,1,2]=a[2,1,1,0]
        a[0,2,1,1]=a[2,1,1,0]
        a[0,1,2,1]=a[2,1,1,0]
        a[0,1,1,2]=a[2,1,1,0]
        
        a[2,2,1,0]=2*((0)+(0.026082026547865053022)*1j)*f[self.idx(2,2)]+ 2*((0)+(0.030116930096841707924)*1j)*f[self.idx(4,2)]
        a[2,2,0,1]=a[2,2,1,0]
        a[2,1,2,0]=a[2,2,1,0]
        a[2,1,0,2]=a[2,2,1,0]
        a[2,0,2,1]=a[2,2,1,0]
        a[2,0,1,2]=a[2,2,1,0]
        a[1,2,2,0]=a[2,2,1,0]
        a[1,2,0,2]=a[2,2,1,0]
        a[1,0,2,2]=a[2,2,1,0]
        a[0,2,2,1]=a[2,2,1,0]
        a[0,2,1,2]=a[2,2,1,0]
        a[0,1,2,2]=a[2,2,1,0]
        
        a[2,2,2,0]=2*((-0.078246079643595159065)+(0)*1j)*f[self.idx(2,1)]+ 2*((-0.042591770999995994217)+(0)*1j)*f[self.idx(4,1)]
        a[2,2,0,2]=a[2,2,2,0]
        a[2,0,2,2]=a[2,2,2,0]
        a[0,2,2,2]=a[2,2,2,0]
        
        a[1,1,1,1]=((0.2)+(0)*1j)*f[self.idx(0,0)]+ ((-0.12777531299998798265)+(0)*1j)*f[self.idx(2,0)]+ 2*((-0.15649215928719031813)+(0)*1j)*f[self.idx(2,2)]\
            + ((0.028571428571428571429)+(0)*1j)*f[self.idx(4,0)]+ 2*((0.030116930096841707924)+(0)*1j)*f[self.idx(4,2)]+ 2*((0.039840953644479787999)+(0)*1j)*f[self.idx(4,4)]
        
        a[2,1,1,1]=2*((0)+(-0.078246079643595159065)*1j)*f[self.idx(2,1)]+ 2*((0)+(0.031943828249996995663)*1j)*f[self.idx(4,1)]+ 2*((0)+(0.028171808490950552584)*1j)*f[self.idx(4,3)]
        a[1,2,1,1]=a[2,1,1,1]
        a[1,1,2,1]=a[2,1,1,1]
        a[1,1,1,2]=a[2,1,1,1]
        
        a[2,2,1,1]=((0.066666666666666666667)+(0)*1j)*f[self.idx(0,0)]+ ((0.021295885499997997109)+(0)*1j)*f[self.idx(2,0)]+ 2*((-0.026082026547865053022)+(0)*1j)*f[self.idx(2,2)]+ ((-0.038095238095238095238)+(0)*1j)*f[self.idx(4,0)]+ 2*((-0.030116930096841707924)+(0)*1j)*f[self.idx(4,2)]
        a[2,1,2,1]=a[2,2,1,1]
        a[2,1,1,2]=a[2,2,1,1]
        a[1,2,2,1]=a[2,2,1,1]
        a[1,2,1,2]=a[2,2,1,1]
        a[1,1,2,2]=a[2,2,1,1]
        
        a[2,2,2,1]=2*((0)+(-0.078246079643595159065)*1j)*f[self.idx(2,1)]+ 2*((0)+(-0.042591770999995994217)*1j)*f[self.idx(4,1)]
        a[2,2,1,2]=a[2,2,2,1]
        a[2,1,2,2]=a[2,2,2,1]
        a[1,2,2,2]=a[2,2,2,1]
        
        a[2,2,2,2]=((0.2)+(0)*1j)*f[self.idx(0,0)]+ ((0.2555506259999759653)+(0)*1j)*f[self.idx(2,0)]+ ((0.076190476190476190476)+(0)*1j)*f[self.idx(4,0)]
        
        #a=a*sqrt(4*pi)
        return a




