
import numpy as np
import pkgutil
import io

I = np.eye(3)
    
def MatrixLoad(sh):
    
    filepath = 'data/MatricesL' + str(sh.lmax) + '.npz'
    fobj = io.BytesIO(pkgutil.get_data(__name__,filepath))
    npzfile = np.load(fobj)
    R=npzfile['R']
    Ri=npzfile['Ri']
    B=npzfile['B']
    Bi=npzfile['Bi']
    G=npzfile['G']
    Gi=npzfile['Gi']
    return R,Ri,B,Bi,G,Gi
    
class rk3iterate:
    def __init__(self,T,gradu,sh,lamfactor=1.0):
        self.sh=sh

        self.parameters = parameters()
        self.iota = self.parameters.iota(T)
        self.beta = self.parameters.beta(T)
        self.lamb = self.parameters.lamb(T)
        
        self.gradu = gradu
        self.D = 0.5*(self.gradu+self.gradu.T)
        self.W = 0.5*(self.gradu-self.gradu.T)
        self.D2 = np.einsum('ij,ji',self.D,self.D)
        self.effectiveSR = np.sqrt(0.5*self.D2)
        
        

        # self.iota = 0.02589*T + 1.78
        # self.lamb = 0.0003037*T + 0.161
        # self.beta = 0.1706*T + 5.898
        
        self.lamb=self.lamb*self.effectiveSR/lamfactor
        self.beta=self.beta*self.effectiveSR
        [self.R,self.Ri,self.B,self.Bi,self.G,self.Gi] = MatrixLoad(self.sh)
        
        self.M = self.linear()
        
    def linear(self):
        Mr =  + np.einsum('ijkl,kl->ij',self.R,self.gradu) - self.iota*np.einsum('ijkl,kl->ij',self.B,self.gradu)\
              - self.lamb*np.einsum('ij,j->ij',np.eye(self.sh.nlm),self.sh.l*(self.sh.l+1))
        Mi =  + np.einsum('ijkl,kl->ij',self.Ri,self.gradu) - self.iota*np.einsum('ijkl,kl->ij',self.Bi,self.gradu)
        
              
        return Mr.real - 1j*Mi.real
    
    def NL(self,f):
        a2=self.sh.a2(f)
        a4=self.sh.a4(f)
        
        DijDkl = np.einsum('ij,kl->ijkl',self.D,self.D)
        IikAjl = np.einsum('ik,jl->ijkl',np.eye(3),a2)
        Gu = np.einsum('ijklmn,klmn->ij',self.G,DijDkl)
        Gui = np.einsum('ijklmn,klmn->ij',self.Gi,DijDkl)
        
        return 5*self.beta*((np.einsum('ij,j->i',Gu,f.real) +np.einsum('ij,j->i',Gui,f.imag))\
                         - np.einsum('ijkl,ijkl,m->m',DijDkl,IikAjl-a4,f))/self.D2
            
            
            


            
    def iterate(self,f,dt):
        
        ak = (8/15,5/12,3/4)
        bk = (0, -17/60, -5/12)  
        fm = np.zeros_like(f)

        
        for rk in range(3):
            RH = np.einsum('ij,j->i',np.eye(self.sh.nlm) + 0.5*(ak[rk]+bk[rk])*dt*self.M,f) \
                    + ak[rk]*dt*self.NL(f) + bk[rk]*dt*self.NL(fm)
            fm = np.copy(f)
            f = np.linalg.solve(np.eye(self.sh.nlm) - 0.5*(ak[rk]+bk[rk])*dt*self.M,RH)
            
        return f
    


class parameters:
    def __init__(self):

        # Raw data from simple shear and compression inversion Richards et al. 2020
        self.rawT = np.array  ( [-30, -13.6, -10.2, -9.5, -30.3, -7, -5.5])
        self.rawlamb = 2*np.array( [0.173, 0.198, 0.126, 0.343, 0.153, 0.139, 0.178])
        self.rawbeta = 2*np.array([0.62, 4.25, 5.92, 2.75, 0.763, 4.12, 5.51])
        self.rawiota = np.array([1.23, 1.93, 1.54, 1.98, 0.993, 1.65, 1.59])

        self.plamb, self.lambcov = np.polyfit(self.rawT,self.rawlamb, 1, cov=True)
        self.pbeta, self.betacov = np.polyfit(self.rawT,self.rawbeta, 1, cov=True)
        self.piota, self.iotacov = np.polyfit(self.rawT,self.rawiota, 1, cov=True)


    def lamb(self,T):
        return np.polyval(self.plamb,T)

    def beta(self,T):
        return np.polyval(self.pbeta,T)

    def iota(self,T):
        return np.polyval(self.piota,T)

    