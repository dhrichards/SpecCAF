#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np	# numpy for arrays
from tqdm import tqdm
import SpecCAF.Solver as Solver
import SpecCAF.spherical as spherical
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d,interp2d
from scipy.signal import find_peaks
from scipy.ndimage import median_filter,gaussian_filter1d
import matplotlib as mpl



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 12,
    "figure.autolayout" : True,
    
})

def peakfind(f,tol=0.1):
    rh,th = sh.y0synth(f)
#tol=0.1

#rh,th = sh.y0synth(sol.y)
    
    peaksth = np.zeros((2,rh.shape[1]))
    peaksrh = np.zeros((2,rh.shape[1]))
    cpotype = []
    
    for i in range(rh.shape[1]):
        peakstemp = find_peaks(rh[:,i])[0]
        if peakstemp.size==0:
            peaksth[:,i] = np.nan
            peaksrh[:,i] = np.mean(rh[:,i])
            cpotype.append('Isotropic')
            
        else:
                
            rhtemp = rh[peakstemp,i]
            sort = np.argsort(rhtemp)
            
            peaksth[0,i]=th[peakstemp[sort[-1]]]
            peaksrh[0,i]=rhtemp[sort[-1]]
            
            
            if peakstemp.size>1:
                peaksth[1,i]=th[peakstemp[sort[-2]]]
                peaksrh[1,i]=rhtemp[sort[-2]]
        
            if peaksrh[1,i]/peaksrh[0,i]>1-tol:
                cpotype.append('2D Cone')
            elif peaksrh[1,i]/peaksrh[0,i]<tol:
                cpotype.append('Single Maxima')
            else:
                cpotype.append('Secondary Cluster')
    return cpotype,peaksrh,peaksth


def cpoIdentify(ratios,tol=0.1):
    cpotype= np.zeros_like(ratios,dtype='object')
    
    cpotype[np.isnan(ratios)]='Isotropic'
    cpotype[ratios>1-tol]='2D Cone'
    cpotype[ratios<tol]='Single Maxima'
    cpotype[cpotype==0]='Secondary Cluster'
    
    return cpotype
        

newcols = np.zeros((256,4))
col1 = np.array([252/256, 141/256, 89/256, 1])
col2 = np.array([255/256, 255/256, 191/256, 1])
col3 = np.array([145/256, 191/256, 219/256, 1])

tol=0.1
newcols[:int(256*tol),:]=col1
newcols[int(256*tol):int((1-tol)*256),:]=col2
newcols[int((1-tol)*256):]=col3
newcols = gaussian_filter1d(newcols, 2.1,axis=0)
newcmp = mpl.colors.ListedColormap(newcols)


sh = spherical.spherical(12)


#Set initial Conditon
f0=sh.spec_array()
f0[0]=1


strainvec = np.linspace(0,10,200)
Tvec = np.linspace(-30,-5,100)
#Wvec = np.linspace(0,1,30)
Wvec = np.logspace(-1,1,100)
#Wvec = np.concatenate([np.linspace(0,0.1,10),np.logspace(-1,1,20)])


Wgrid,Tgrid,Sgrid=np.meshgrid(Wvec,Tvec,strainvec)
cpotypes = np.zeros((Tvec.size,Wvec.size,strainvec.size),dtype=object)
ratios = np.zeros((Tvec.size,Wvec.size,strainvec.size))
peakth = np.zeros_like(ratios)
F = np.zeros((Tvec.size,Wvec.size,sh.nlm,strainvec.size))


Whd = np.logspace(-1,1,1000)
Thd = np.linspace(-30,-5,1000)

# F = np.load('F100log.npy')
# ratios = np.load('ratios100log.npy')
# peakth = np.load('peakth100log.npy')


cpotypes = cpoIdentify(ratios,tol)
for i in tqdm(range(Tvec.size)):
    for j in range(Wvec.size):
        
        p=Solver.params(Tvec[i],strainvec,Wvec[j])
        sol=Solver.rk3solve(p,sh,f0)
        
        
        interp = interp1d(sol.t,sol.y)
        F[i,j,:,:] = interp(strainvec)
        
        cpotype,peaksrh,peaksth = peakfind(F[i,j,:,:])
        
        cpotypes[i,j,:] = cpotype
        ratios[i,j,:] = peaksrh[1,:]/peaksrh[0,:]
        peakth[i,j,:] = peaksth[0,:]
    

strainplots = np.array([0.3, 0.5, 1, 2,5,10])


r=0.07
ncol=2
nrow = strainplots.size//ncol
fig2,ax2 = plt.subplots(nrow,ncol,figsize=(8,9))
Tins = np.array([[-25,-10],[-25,-10],[-25,-15],[-25,-10],[-20,-15],[-20,-15]])
Wins = np.array([[0.3,2],[0.2,1],[1,0.2],[0.3,1],[0.2,5],[0.2,4]])
subfiglabels=('a','b','c','d','e','f')

i=0

#for i in range(strainplots.size):
for ax in ax2.flat:    
    strainind = np.argmin(np.abs(strainvec-strainplots[i]))

    
    R = interp2d(Tvec,Wvec,ratios[:,:,strainind])
    ratioshd = R(Thd,Whd).T
    med = median_filter(ratioshd,35)
    #cpotypeshd = cpoIdentify(med)
    
    #Zlocal = fv(cpotypeshd).T
    im2 = ax.contourf(Thd,Whd,med,vmin=0,vmax=1,cmap=newcmp)
    con = ax.contour(Tvec,Wvec,(180/np.pi)*peakth[:,:,strainind].T,[20,50],colors='black',lw=3)
    if i==1:
        ax.clabel(con,inline=1,fmt='%1.0f',inline_spacing=0,manual=[(-29,0.6),(-15,4)])   
    else:
        ax.clabel(con,inline=1,fmt='%1.0f',inline_spacing=0)

    ax.set_ylabel('$\mathcal{W}$')
    ax.set_xlabel('$T(^{\circ}C)$')
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_title('(' + subfiglabels[i] + '), $\gamma=' + ('%.2f' % strainplots[i]) + '$')
    ax.set_yticks(np.logspace(-1,1,5))
    #ax.grid(b=True, which='both',axis='both')
    ##Inset polefigures

    # for j in range(2):
    #     Wind = np.abs(Wvec-Wins[i,j]).argmin()
    #     Tind = np.abs(Tvec-Tins[i,j]).argmin()
    #     xx,yy,fgrid=sh.polefigure(F[Tind,Wind,:,strainind])
    #     axcentre = ax.transData.transform((Tvec[Tind],Wvec[Wind]))
    #     inv=ax.transAxes.inverted()
    #     axcentre = inv.transform(axcentre)
    #     ins = ax.inset_axes([axcentre[0]-r,axcentre[1]-r,2*r,2.926*r])
    #     cs=ins.contour(xx,yy,fgrid,levels=np.linspace(0,1,11),vmin=0)
    #     circle=plt.Circle((0,0),radius=0.98*np.pi/2,alpha=0.5,fc='w',ec='black')
    #     ins.add_patch(circle)
    #     ins.axis('off')
    i=i+1

legend_elements = [mpl.patches.Patch(facecolor=col1,label='Single Maxima'),
                   mpl.patches.Patch(facecolor=col2,label='Secondary Cluster'),
                   mpl.patches.Patch(facecolor=col3,label='Double Maxima')]
#leg_ax = fig2.add_axes([0.1, -0.05, 0.8, 0.05])
leg = fig2.legend(handles=legend_elements,bbox_to_anchor=(0.1,0,0.8,0.02),ncol=3,mode="expand")



fig2.savefig('regimediagram.png',dpi=600,format='png',bbox_inches='tight')

fig3,ax3= plt.subplots(nrow,ncol,figsize=(8,9))


i=0
for ax in ax3.flat:
    
    strainind = np.argmin(np.abs(strainvec-strainplots[i]))


    cs=ax.contourf(Tvec,Wvec,(180/np.pi)*peakth[:,:,strainind].T,[0, 10, 20, 30, 40, 50, 60, 70],cmap='magma')
    ax.set_aspect('auto')
    ax.set_yscale('log')
    ax.set_xlabel('$T(^{\circ}C)$')
    ax.set_ylabel('$\mathcal{W}$')
    ax.set_title('(' + subfiglabels[i] + '), $\gamma=' + ('%.2f' % strainplots[i]) + '$')
    
    i=i+1

cbar_ax = fig3.add_axes([0.1,-0.01, 0.8, 0.02])
cbar=fig3.colorbar(cs,cax=cbar_ax,orientation='horizontal')
cbar.set_label('Angle between primary cluster and compression axis $(^{\\circ})$')
#fig3.colorbar(cs,bbox_to_anchor=(0.1,0,0.8,0.02))

fig3.savefig('singlemaxtheta.png',dpi=600,format='png',bbox_inches='tight')