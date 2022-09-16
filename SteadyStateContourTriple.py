#%%

import numpy as np	# numpy for arrays
from tqdm import tqdm
import SpecCAF.Solver as Solver
import SpecCAF.spherical as spherical
from matplotlib import pyplot as plt
import SteadyStateTime as ss
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
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

size=50
strainvec = np.linspace(0,30,300)
Tvec = np.linspace(-30,-5,size)
#Wvec = np.linspace(0,1,10)
Wvec = np.logspace(-1,1,size)
#Wvec = np.concatenate([np.linspace(0,1,size//2),np.logspace(0,1,size//2+1)[1:]])


strainplot=0.5
strainind = np.abs(strainvec - strainplot).argmin()

Sgrid,Tgrid=np.meshgrid(Wvec,Tvec)
contSS = np.zeros((Tvec.size,Wvec.size))
contJ  = np.zeros((Tvec.size,Wvec.size))
contJp5  = np.zeros((Tvec.size,Wvec.size))
F = np.zeros((Tvec.size,Wvec.size,sh.nlm,strainvec.size))
Jval = 1.5

paramtype = 'max'
#F = np.load('Fsssymlog' + str(Wvec.size) + paramtype + '.npy')
F = np.load('Fss' + str(Wvec.size) + paramtype + '.npy')
for i in tqdm(range(Tvec.size)):
    for j in range(Wvec.size):
        
        # gradu = Solver.gradufromW(Wvec[j])
        # p=Solver.params(Tvec[i],strainvec,gradu,paramtype)
        # sol=Solver.rk3solve(p,sh,f0)
        
        
        
        # interp = interp1d(sol.t,sol.y)
        # F[i,j,:,:] = interp(strainvec)
        #contSS[i,j] = ss.ToValue(strainvec, sh.J(F[i,j,:,:]),yroot=Jval)
        contSS[i,j] = ss.HalfwayCubic(strainvec, sh.J(F[i,j,:,:]),value=0.5)
        contJp5[i,j] = sh.J(F[i,j,:,strainind])
        contJ[i,j] = sh.J(F[i,j,:,-1])
    

np.save('Fsssymlog' + str(Wvec.size) + paramtype + '.npy',F)
#fig,ax = plt.subplots(3,2,figsize=(8,6),gridspec_kw={'height_ratios':[0.2,3,1]})
fig=plt.figure(figsize = (8,7))
gs = gridspec.GridSpec(2,2,figure=fig,height_ratios=[1,1])
gs.update(wspace=0.3, hspace=0.4)

gsl= gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0,1],height_ratios=[0.07,0.02,1])
gsr= gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0,0],height_ratios=[0.07,0.02,1])
gsp= gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1,0],height_ratios=[0.07,0.02,1])

cax1 = fig.add_subplot(gsl[0])
cax2 = fig.add_subplot(gsr[0])
cax3 = fig.add_subplot(gsp[0])

ax1 = fig.add_subplot(gsl[2])
ax2 = fig.add_subplot(gsr[2])
ax3 = fig.add_subplot(gsp[2])

gsb = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1,1],height_ratios=[0.1,1,0.1])
ax4 = fig.add_subplot(gsb[1])




#Labels
ax1.set_ylabel('Vorticity number $\mathcal{W}$')
ax1.set_xlabel('$T(^{\circ}C)$')
ax1.set_yscale('log')
ax2.set_ylabel('Vorticity number $\mathcal{W}$')
ax2.set_xlabel('$T(^{\circ}C)$')
ax2.set_yscale('log')
ax3.set_ylabel('Vorticity number $\mathcal{W}$')
ax3.set_xlabel('$T(^{\circ}C)$')
ax3.set_yscale('log')

#
    #ax[1,i].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    #ax[1,i].set_yticks([0.1,1,10])
#ax[0,0].set_title('(a) Steady state strain')
#ax[0,1].set_title('(b) $J$ index at steady state')   

def myfmt(x, pos):
    return '{0:.2f}'.format(x)

strainticks = np.linspace(0,np.nanmax(contSS),10)
#Jticks = np.arange(1,np.max(contJ)+Jval-1,Jval-1)
Jticks = np.linspace(1,np.max(contJ),10)
cs=ax1.contourf(Tgrid,Sgrid,contSS,levels=strainticks,cmap='inferno')
cb1=fig.colorbar(cs,cax=cax1,ticks=strainticks[::3],orientation='horizontal',format=ticker.FuncFormatter(myfmt))
cb1.ax.set_title('Strain at halfway to steady state')
ax1.text(0.05, 0.9, '(b)',transform=ax1.transAxes,color='white')

cs2 = ax2.contourf(Tgrid,Sgrid,contJ,levels=Jticks,vmin=1,cmap='viridis')
cb2=fig.colorbar(cs2,cax=cax2,ticks=Jticks[::3],orientation='horizontal',format=ticker.FuncFormatter(myfmt))
cb2.ax.set_title('$J$ index at steady state')
ax2.text(0.05, 0.9, '(a)',transform=ax2.transAxes,color='white')

cs3 = ax3.contourf(Tgrid,Sgrid,contJp5,levels=Jticks,vmin=1,cmap='viridis')
cb3=fig.colorbar(cs3,cax=cax3,ticks=Jticks[::3],orientation='horizontal',format=ticker.FuncFormatter(myfmt))
cb3.ax.set_title('$J$ index at $\gamma = 0.5$')
ax3.text(0.05, 0.9, '(c)',transform=ax3.transAxes,color='white')


# Plot eigenvalues and steady state time
i=-1
j=size//2
A = sh.a2(F[i,j,:,:])
A = np.moveaxis(A,-1,0)
eigA,w = np.linalg.eig(A)
eigA.sort()
J=sh.J(F[i,j,:,:])
ax4.plot(strainvec,J)
ax4.axvline(contSS[i,j],color='black',lw=1.5,ls='--')
ax4.axhline(contJp5[i,j],color='black',lw=1.5,ls='--')
yplot = (J.max()+1)/2
ax4.text(contSS[i,j]+0.1,1.1,'$\gamma$ at halfway \n to steady state')
ax4.text(2,0.9*contJp5[i,j],'$J$ at $\gamma=0.5$')
ax4.set_xlabel('Strain $\gamma$')
ax4.set_ylabel('$J$')
ax4.set_xlim(0,4)
#ax[1,0].set_ylabel('Eigenvalues of $\mathbf{A^{(2)}}$'-2)
Tplot = Tvec[i]
Wplot = Wvec[j]
Tstr = "{0:0.1f}".format(Tplot)
Wstr = "{0:.3g}".format(Wplot)
title = '(d) $J$ index at $T=' + Tstr +'^{\circ}C$, $\mathcal{W}=' + Wstr + '$'
ax4.set_title(title)




for c in cs.collections:
    c.set_edgecolor("face")
for c in cs2.collections:
    c.set_edgecolor("face")
for c in cs3.collections:
    c.set_edgecolor("face")
    
cb1.solids.set_edgecolor("face")
cb2.solids.set_edgecolor("face")
cb3.solids.set_edgecolor("face")
fig.savefig('fig13triple' + paramtype + '.pdf',format='pdf',bbox_inches='tight')

fig.savefig('steadystateval' + str(Jval) +'.png',dpi=300)
#save('Fsstest' + str(Wvec.size) + paramtype + '.npy',F)

# plt.plot(Wvec,contJ[1,:])
# #plt.xscale('log')
# plt.grid()
# %%
