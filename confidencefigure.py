#%%
import numpy as np
import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
import scipy as sp
import SpecCAF.confidence

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 12,
    "figure.autolayout" : True,
    
})

confidence = 0.8
confidence2 = 0.95
params = SpecCAF.confidence.parameters(confidence)
params2 = SpecCAF.confidence.parameters(confidence2)

iotacol = 'red'
betacol = 'blue'
lambcol = 'green'
fillalpha = 0.4
fillalpha2 = 0.15

Tvec = np.linspace(-30,-5,100)

fig,ax = plt.subplots(1,2,figsize = (8,5))

ax[0].scatter(params.rawT,params.rawiota,color=iotacol, marker = 'x')
ax[0].fill_between(Tvec,params.iotaLB(Tvec),params.iotaUB(Tvec),color=iotacol,alpha=fillalpha)
ax[0].plot(Tvec,params.iota(Tvec),color=iotacol)
ax[0].fill_between(Tvec,params2.iotaLB(Tvec),params2.iotaUB(Tvec),color=iotacol,alpha=fillalpha2)

ax[0].scatter(params.rawT,params.rawlamb,color=lambcol, marker = 'x')
ax[0].fill_between(Tvec,params.lambLB(Tvec),params.lambUB(Tvec),color=lambcol,alpha=fillalpha)
ax[0].plot(Tvec,params.lamb(Tvec),color=lambcol)
ax[0].fill_between(Tvec,params2.lambLB(Tvec),params2.lambUB(Tvec),color=lambcol,alpha=fillalpha2)

ax[1].scatter(params.rawT,params.rawbeta,color=betacol, marker='x',
             label = 'Points from inversion')
ax[1].fill_between(Tvec,params.betaLB(Tvec),params.betaUB(Tvec),color=betacol,alpha=fillalpha,
                label = '$' + str(int(confidence*100)) + '\%$ Confidence Interval')
ax[1].plot(Tvec,params.beta(Tvec),color=betacol, label = 'Linear fit')
ax[1].fill_between(Tvec,params2.betaLB(Tvec),params2.betaUB(Tvec),color=betacol,alpha=fillalpha2,
                label = '$' + str(int(confidence2*100)) + '\%$ Confidence Interval')


ax[0].set_ylabel('$\iota,\\tilde{\lambda}$')
ax[0].set_xlabel('$T(^{\circ}C)$')
ax[0].set_ylim(top=2.3)
ax[1].set_ylabel('$\\tilde{\\beta}$')
ax[1].set_xlabel('$T(^{\circ}C)$')

ax[1].legend(loc=4)


def myfmt(x):
    return '{0:.3g}'.format(x)

legend_elements = [mpl.lines.Line2D([0],[0],color=iotacol,label='$\iota = ' + \
                    myfmt(params.piota.tolist()[0]) + 'T + ' + \
                        myfmt(params.piota.tolist()[1]) + '$'),
                   mpl.lines.Line2D([0],[0],color=lambcol,label='$\\tilde{\lambda} = ' + \
                    myfmt(params.plamb.tolist()[0]) + 'T + ' + \
                        myfmt(params.plamb.tolist()[1]) + '$'),
                   mpl.lines.Line2D([0],[0],color=betacol,label='$\\tilde{\\beta} = ' + \
                    myfmt(params.pbeta.tolist()[0]) + 'T + ' + \
                        myfmt(params.pbeta.tolist()[1]) + '$')]
#leg = ax[0].legend(handles = legend_elements, ncol=3, loc = 2)
leg = fig.legend(handles=legend_elements,bbox_to_anchor=(0.1,0,0.8,0.02),ncol=3,mode="expand")

fig.savefig('figS1.pdf',format='pdf',bbox_inches='tight')




# %%
