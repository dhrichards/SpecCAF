#%%
import numpy as np
sys.path.insert(0, '../src/')
import spherical
import solver
from matplotlib import pyplot as plt

import SpecCAF
sh = spherical.spherical(12)


#Set initial Conditon
f0=sh.spec_array()
f0[0]=1


nt=1000
dt = 0.01

f = np.zeros((f0.size,nt),dtype='complex128')
f[:,0] = f0


gradu =  np.array([[-0.5, 0., 0.],\
                     [0., 1., 0.],\
                     [0.0, 0., -0.5]])
            

rk = solver.rk3iterate(-10,gradu,sh)

for i in range(nt-1):
    f[:,i+1]=rk.iterate(f[:,i],dt)


fig,ax = plt.subplots(figsize=(3,3))

xx,yy,fgrid=sh.polefigure(f[:,-1])
fgrid=fgrid.real
cs=ax.contourf(xx,yy,fgrid,20)
