#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np	# numpy for arrays
from tqdm import tqdm
import SpecCAF.Solver as Solver
import SpecCAF.spherical as spherical
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d,interp2d
from scipy.signal import find_peaks
from scipy.ndimage import median_filter,gaussian_filter1d
import matplotlib as mpl
from matplotlib.colors import Normalize as norm


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 11,
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


strainvec = np.linspace(0,10,100)
Tvec = np.linspace(-30,-5,50)
#Wvec = np.linspace(0,1,30)
Wvec = np.logspace(-1,1,50)
#Wvec = np.concatenate([np.linspace(0,0.1,10),np.logspace(-1,1,20)])


Wgrid,Tgrid,Sgrid=np.meshgrid(Wvec,Tvec,strainvec)
cpotypes = np.zeros((Tvec.size,Wvec.size,strainvec.size),dtype=object)
ratios = np.zeros((Tvec.size,Wvec.size,strainvec.size))
peakth = np.zeros_like(ratios)
F = np.zeros((Tvec.size,Wvec.size,sh.nlm,strainvec.size))


Whd = np.logspace(-1,1,100)
Thd = np.linspace(-30,-5,100)

paramtype = 'min'
F = np.load('F' + str(Wvec.size) + paramtype + '.npy')


cpotypes = cpoIdentify(ratios,tol)
for i in tqdm(range(Tvec.size)):
    for j in range(Wvec.size):
        
        # gradu = Solver.gradufromW(Wvec[j])
        # p=Solver.params(Tvec[i],strainvec,gradu)
        # sol=Solver.rk3solve(p,sh,f0)
        
        
        # interp = interp1d(sol.t,sol.y)
        # F[i,j,:,:] = interp(strainvec)
        
        cpotype,peaksrh,peaksth = peakfind(F[i,j,:,:])
        
        cpotypes[i,j,:] = cpotype
        ratios[i,j,:] = peaksrh[1,:]/peaksrh[0,:]
        peakth[i,j,:] = peaksth[0,:]
    

strainplot = 2


r=0.1

fig2,ax = plt.subplots(figsize=(4,6))

Tinsv = np.linspace(-30,-10,3)
Winsv = np.logspace(-1,1,5)

Tins,Wins=np.meshgrid(Tinsv,Winsv)
strainind = np.argmin(np.abs(strainvec-strainplot))


R = interp2d(Tvec,Wvec,ratios[:,:,strainind])
ratioshd = R(Thd,Whd).T
med = median_filter(ratioshd,35)

im2 = ax.contourf(Thd,Whd,med,vmin=0,vmax=1,cmap=newcmp)

# Padding
ax.tick_params(axis='x', pad=30)
ax.tick_params(axis='y', pad=30)

ax.set_ylabel('$\mathcal{W}$')
ax.set_xlabel('$T(^{\circ}C)$')
ax.set_yscale('log')
ax.set_title('$\gamma=' + ('%.2f' % strainplot) + '$')
ax.set_yticks(np.logspace(-1,1,5))
ax.set_xlim([-30,-10])
#ax.grid(b=True, which='both',axis='both')
    #Inset polefigures
vm=0.8
xscale =2.338

shpf = spherical.spherical(12)
f0pf = shpf.spec_array()
f0pf[0]=1.
for i in range(Tinsv.size):
    for j in range(Winsv.size):

        # xx,yy,fgrid=sh.polefigure(F[Tind,Wind,:,strainind])
        Wind = np.abs(Wvec-Wins[j,i]).argmin()
        Tind = np.abs(Tvec-Tins[j,i]).argmin()
        
        gradu = Solver.gradufromW(Wins[j,i])
        p=Solver.params(Tins[j,i],np.linspace(0,strainplot,100),gradu)
        sol=Solver.rk3solve(p,shpf,f0pf)
        xx,yy,fgrid=shpf.polefigure(sol.y[:,-1])
        
        axcentre = ax.transData.transform((Tins[j,i],Wins[j,i]))
        inv=ax.transAxes.inverted()
        axcentre = inv.transform(axcentre)
        ins = ax.inset_axes([axcentre[0]-xscale*r,axcentre[1]-r,2*xscale*r,2*r])
        cs=ins.contourf(xx,yy,fgrid,10,vmin=0,vmax=vm,alpha=1,antialiased=True)
        cs=ins.contourf(xx,yy,fgrid,10,vmin=0,vmax=vm,alpha=1,antialiased=True)
        csl=ins.contour(xx,yy,fgrid,10,vmin=0,vmax=vm,linewidths=0.1,colors='white')
        ins.axis('off')
        # for c in cs.collections:
        #     c.set_edgecolor("face")
        # for c in csl.collections:
        #     c.set_edgecolor('face')


legend_elements = [mpl.patches.Patch(facecolor=col1,label='Single Maxima'),
                   mpl.patches.Patch(facecolor=col2,label='Secondary Cluster')]
#leg_ax = fig2.add_axes([0.1, -0.05, 0.8, 0.05])
leg = fig2.legend(handles=legend_elements,bbox_to_anchor=(0.1,0,0.9,0.02),ncol=2,mode="expand")

cbar=plt.colorbar(cm.ScalarMappable(norm(0, vm),cmap='viridis'),pad=0.18,ticks=np.mgrid[0:vm:0.1],aspect=35)

cbar.ax.set_title('$\\rho^*$')
cbar.solids.set_edgecolor("face")
fig2.savefig('fig12' + paramtype + '.png',format='png',dpi=400,bbox_inches='tight')




# %%
