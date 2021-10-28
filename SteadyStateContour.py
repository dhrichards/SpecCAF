#%%

from numpy import *	# numpy for arrays
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


strainvec = linspace(0,30,300)
Tvec = linspace(-30,-5,60)
#Wvec = linspace(0,3,50)
Wvec = logspace(-1,1,60)

Sgrid,Tgrid=meshgrid(Wvec,Tvec)
contSS = zeros((Tvec.size,Wvec.size))
contJ = zeros((Tvec.size,Wvec.size))
F = zeros((Tvec.size,Wvec.size,sh.nlm,strainvec.size))

paramtype = 'min'
F = load('Fss01' + str(Wvec.size) + paramtype + '.npy')
#F = load('Fss' + str(Wvec.size) + paramtype + '.npy')
for i in tqdm(range(Tvec.size)):
    for j in range(Wvec.size):
        
        # gradu = Solver.gradufromW(Wvec[j])
        # p=Solver.params(Tvec[i],strainvec,gradu,paramtype)
        # sol=Solver.rk3solve(p,sh,f0)
        
        
        # interp = interp1d(sol.t,sol.y)
        # F[i,j,:,:] = interp(strainvec)
        contSS[i,j] = ss.HalfwayCubic(strainvec, sh.J(F[i,j,:,:]),value=0.5)
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

def myfmt(x, pos):
    return '{0:.2f}'.format(x)


cs=ax1.contourf(Tgrid,Sgrid,contSS,levels=linspace(contSS.min(),contSS.max(),9),cmap='inferno')
cb1=fig.colorbar(cs,cax=cax1,ticks=linspace(contSS.min(),contSS.max(),5),orientation='horizontal',format=ticker.FuncFormatter(myfmt))
cb1.ax.set_title('Strain at halfway to steady state')
ax1.text(0.05, 0.9, '(a)',transform=ax1.transAxes,color='white')

cs2 = ax2.contourf(Tgrid,Sgrid,contJ,levels=linspace(1,contJ.max(),9),vmin=1,cmap='viridis')
cb2=fig.colorbar(cs2,cax=cax2,ticks=linspace(1,contJ.max(),5),orientation='horizontal',format=ticker.FuncFormatter(myfmt))
cb2.ax.set_title('$J$ index at steady state')
ax2.text(0.05, 0.9, '(b)',transform=ax2.transAxes,color='white')

# Plot eigenvalues and steady state time
i=25
j=25
A = sh.a2(F[i,j,:,:])
A = moveaxis(A,-1,0)
eigA,w = linalg.eig(A)
eigA.sort()
J=sh.J(F[i,j,:,:])
ax3.plot(strainvec,J)
ax3.axvline(contSS[i,j],color='black',lw=2,ls='--')
yplot = (J.max()+1)/2
ax3.text(contSS[i,j]+0.25,yplot,'Strain at halfway to steady state')
ax3.set_xlabel('Strain $\gamma$')
ax3.set_ylabel('$J$')
ax3.set_xlim(0,4)
#ax[1,0].set_ylabel('Eigenvalues of $\mathbf{A^{(2)}}$'-2)
Tplot = Tvec[i]
Wplot = Wvec[j]
Tstr = "{0:0.1f}".format(Tplot)
Wstr = "{0:.3g}".format(Wplot)
title = '(c) $J$ index at $T=' + Tstr +'^{\circ}C$, $\mathcal{W}=' + Wstr + '$'
ax3.set_title(title)




for c in cs.collections:
    c.set_edgecolor("face")
for c in cs2.collections:
    c.set_edgecolor("face")
    
cb1.solids.set_edgecolor("face")
cb2.solids.set_edgecolor("face")
fig.savefig('fig13halfway' + paramtype + '.pdf',format='pdf',bbox_inches='tight')


#save('Fsstest' + str(Wvec.size) + paramtype + '.npy',F)

# plt.plot(Wvec,contJ[1,:])
# #plt.xscale('log')
# plt.grid()
# %%
