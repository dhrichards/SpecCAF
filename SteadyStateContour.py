#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import *	# numpy for arrays
from tqdm import tqdm
import SpecCAF.Solver as Solver
import SpecCAF.spherical as spherical
from matplotlib import pyplot as plt
import SteadyStateTime as ss
from scipy.interpolate import interp1d
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 11,
    "axes.titlesize" : "medium",
    "figure.autolayout" : True,
    
})

sh = spherical.spherical(12)


#Set initial Conditon
f0=sh.spec_array()
f0[0]=1


strainvec = linspace(0,10,200)
Tvec = linspace(-30,-5,50)
#Wvec = linspace(0,1,4)
Wvec = logspace(-1,1,50)

Sgrid,Tgrid=meshgrid(Wvec,Tvec)
contSS = zeros((Tvec.size,Wvec.size))
contJ = zeros((Tvec.size,Wvec.size))
F = zeros((Tvec.size,Wvec.size,sh.nlm,strainvec.size))





for i in tqdm(range(Tvec.size)):
    for j in range(Wvec.size):
        
        p=Solver.params(Tvec[i],strainvec,Wvec[j])
        sol=Solver.rk3solve(p,sh,f0)
        
        
        interp = interp1d(sol.t,sol.y)
        F[i,j,:,:] = interp(strainvec)
        contSS[i,j] = ss.SteadyStateRM(strainvec, sh.J(F[i,j,:,:]),strainwindow=2,tolperc=5)
        contJ[i,j] = sh.J(F[i,j,:,-1])
    
    

#fig,ax = plt.subplots(3,2,figsize=(8,6),gridspec_kw={'height_ratios':[0.2,3,1]})
fig=plt.figure(figsize = (8,6))
gs = gridspec.GridSpec(2,2,figure=fig,height_ratios=[3.3,1])
gs.update(wspace=0.3, hspace=0.4)

gsl= gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,0],height_ratios=[0.07,1])
gsr= gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,1],height_ratios=[0.07,1])

cax1 = fig.add_subplot(gsl[0])
cax2 = fig.add_subplot(gsr[0])

ax1 = fig.add_subplot(gsl[1])
ax2 = fig.add_subplot(gsr[1])


gsb = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1,:],width_ratios=[0.4,1,0.4])
ax3 = fig.add_subplot(gsb[1])




#Labels
ax1.set_ylabel('Vorticity number $\mathcal{W}$')
ax1.set_xlabel('$T(^{\circ}C)$')
ax1.set_yscale('log')
ax2.set_ylabel('Vorticity number $\mathcal{W}$')
ax2.set_xlabel('$T(^{\circ}C)$')
ax2.set_yscale('log')

    #ax[1,i].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    #ax[1,i].set_yticks([0.1,1,10])
#ax[0,0].set_title('(a) Steady state strain')
#ax[0,1].set_title('(b) $J$ index at steady state')   

cs=ax1.contourf(Tgrid,Sgrid,contSS,50,cmap='inferno')
cb1=fig.colorbar(cs,cax=cax1,ticks=mgrid[floor(contSS.min()):ceil(contSS.max()):0.5],orientation='horizontal')
cb1.ax.set_title('Strain at steady state')
ax1.text(0.05, 0.9, '(a)',transform=ax1.transAxes,color='white')

cs2 = ax2.contourf(Tgrid,Sgrid,contJ,50,vmin=1,cmap='viridis')
cb2=fig.colorbar(cs2,cax=cax2,ticks=mgrid[1:ceil(contJ.max()):0.5],orientation='horizontal')
cb2.ax.set_title('$J$ index at steady-state')
ax2.text(0.05, 0.9, '(b)',transform=ax2.transAxes,color='white')

# Plot eigenvalues and steady state time
i=0
j=0
A = sh.a2(F[i,j,:,:])
A = moveaxis(A,-1,0)
eigA,w = linalg.eig(A)
eigA.sort()
J=sh.J(F[i,j,:,:])
ax3.plot(strainvec,J)
ax3.axvline(contSS[i,j],color='black',lw=2,ls='--')
yplot = (J.max()+1)/2
ax3.text(contSS[i,j]+0.25,yplot,' Strain at steady state')
ax3.set_xlabel('Strain $\gamma$')
ax3.set_ylabel('$J$')
ax3.set_xlim(0,10)
#ax[1,0].set_ylabel('Eigenvalues of $\mathbf{A^{(2)}}$'-2)
title = '(c) $J$ index at $T=-30^{\circ}C$, $\mathcal{W}=0$'
ax3.set_title(title)


# i=0
# j=-1
# ax[1,1].plot(strainvec,sh.J(F[i,j,:,:]))
# ax[1,1].axvline(contSS[i,j],color='black')

plt.show()
fname = 'steadystate3.png' 
fig.savefig(fname,dpi=600,format='png',bbox_inches='tight')

# plt.plot(Wvec,contJ[1,:])
# #plt.xscale('log')
# plt.grid()