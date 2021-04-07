#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import *	# numpy for arrays
from tqdm import tqdm
import SpecCAF.Solver as Solver
import SpecCAF.spherical as spherical
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize as norm
from scipy.interpolate import interp1d
from mpl_toolkits import mplot3d
import NumpyViscosities as visc
import colorcet
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 11,
    "figure.autolayout" : True,
    
})

sh = spherical.spherical(12)
y0 = True
colmap='viridis'

#Set initial Conditon
f0=sh.spec_array()
f0[0]=1

if y0:
    strainvec = linspace(0,2,100)
else:
    strainvec = linspace(0,0.5,100)
Tvec = linspace(-30,-10,3)
#Wvec = linspace(0,1,4)
Wvec = logspace(-1,1,5)
#Wvec = array([0.1, 0.5, 1, 3, 10])

Sgrid,Tgrid=meshgrid(Wvec,Tvec)
A=zeros((Tvec.size,Wvec.size,strainvec.size))


F = zeros((Tvec.size,Wvec.size,sh.nlm,strainvec.size))

#radius of pole figures
r=0.1

#fontsize
fs=11
# Grid for plotting pole figures -  map onto 0-1 space
Wlin = log10(Wvec)
yp = (Wlin-Wlin.min())/(Wlin.max()-Wlin.min())
xp = (Tvec-Tvec.min())/(Tvec.max()-Tvec.min())

[Xp,Yp] = meshgrid(xp,yp)

fig,mainax = plt.subplots(figsize=(8,8))
#mainax = fig.add_axes([0, 0, 1, 1])
#plt.plot(Wvec,Tvec,alpha=0)
# Padding
mainax.tick_params(axis='x', pad=55)
mainax.tick_params(axis='y', pad=55)
#Labels
mainax.set_ylabel('Vorticity number $\mathcal{W}$',fontsize=fs)
mainax.set_xlabel('$T(^{\circ}C)$',fontsize=fs)
mainax.grid(True)

#Limits
ylab = []
for i in range(Wvec.size):
    ylab.append('%s' % float('%.3g' % Wvec[i]))
    
xlab=[]
for i in range(Tvec.size):
    xlab.append('%s' % float('%.2g' % Tvec[i]))

mainax.set_yticks(yp)
mainax.set_yticklabels(ylab,fontsize=fs)
mainax.set_xticks(xp)
mainax.set_xticklabels(xlab,fontsize=fs)


ar=Wvec.size/Tvec.size
mainax.set_aspect(ar)

for i in tqdm(range(Tvec.size)):
    for j in range(Wvec.size):
        
        p=Solver.params(Tvec[i],strainvec,Wvec[j])
        sol=Solver.rk3solve(p,sh,f0)
        
        
        interp = interp1d(sol.t,sol.y)
        F[i,j,:,:] = interp(strainvec)
        A[i,j,:] = sh.a2(F[i,j,:,:])[2,2,:]
       
  
#mainax.contourf(Sgrid,Tgrid,A[:,:,k],50,vmin=0,vmax=1)
rhend,th = sh.y0synth(F[-1,Wvec.size//2,:,:])
vm = rhend.max().real

for i in range(Tvec.size):
    for j in range(Wvec.size):
    
        
        ax = mainax.inset_axes([Xp[j,i]-ar*r,Yp[j,i]-r,2*ar*r,2*r])
        rh,th = sh.y0synth(F[i,j,:,:])
        xx,yy,fgrid=sh.polefigure(F[i,j,:,-1])
        if y0:
            con=ax.contourf(strainvec,th*180/pi,rh,50,vmin=0,vmax=vm,cmap=colmap)
            for c in con.collections:
                c.set_edgecolor("face")
        else:
            con = ax.contourf(xx,yy,fgrid,50,vmin=0,vmax=vm,cmap=colmap)
        if y0:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

if y0:
    ax.axis('on')
    ax.set_xticks([0,strainvec[-1]])
    ax.set_yticks([-90,90])
    
    
    ax.set_xlabel('$\gamma$',labelpad=-6,backgroundcolor="white",fontsize=fs-2)
    ax.set_ylabel('$\\theta^{\circ}$',labelpad=-12,backgroundcolor="white",fontsize=fs-2)
    ax.set_xticklabels([0,strainvec[-1]],fontsize=fs-2)
    ax.set_yticklabels([-90,90],fontsize=fs-2)
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=0)
    fname = 'PS-SSy0confined.png'

else:
    mainax.set_title('$\gamma = ' + '{:.2f}$'.format(strainvec[-1]),y=1+r)
    fname = 'PS-SSpfconfined.png'
    
cbar_ax = fig.add_axes([0.1, -0.02, 0.8, 0.02])
cbar=plt.colorbar(cm.ScalarMappable(norm(0, vm),cmap=colmap),cax=cbar_ax,ticks=mgrid[0:vm:0.1],orientation='horizontal')
cbar.set_label('$\\rho^*$')
cbar.solids.set_edgecolor("face")
plt.show()

fig.savefig('fig06.pdf',format='pdf',bbox_inches='tight')
    
    
