import shtns
import numpy as np



class solver:
    def __init__(self,lmax=20,mmax=4):
        
        self.sh = shtns.sht(lmax,mmax)

        self.nlats, self.nlons = self.sh.set_grid()
        self.theta_vals = np.arccos(self.sh.cos_theta)
        self.phi_vals = (2.0*np.pi/self.nlons)*np.arange(self.nlons)
        self.phi, self.theta = np.meshgrid(self.phi_vals, self.theta_vals)

        self.x = np.sin(self.theta)*np.cos(self.phi)
        self.y = np.sin(self.theta)*np.sin(self.phi)
        self.z = np.cos(self.theta)

        self.n = np.array([self.x,self.y,self.z])



    def v_star_cal(self,gradu):
        self.gradu = gradu
        self.D = 0.5*(gradu + gradu.T)
        self.W = 0.5*(gradu - gradu.T)
        self.D2 = np.einsum('ij,ji',self.D,self.D)
        self.effectiveSR = np.sqrt(0.5*self.D2)

        Dn = np.einsum('ij,jpq->ipq',self.D,self.n)
        Wn = np.einsum('ij,jpq->ipq',self.W,self.n)
        Dnn = np.einsum('ij,jpq,ipq->pq',self.D,self.n,self.n)

        self.v = Wn -self.iota*(Dn - Dnn*self.n)
        self.Dstar = 5*(np.einsum('ipq,ipq->pq',Dn,Dn) - Dnn**2)/self.D2
        


        

    def vec_cart2sph(self,v):
        vtheta = v[0,...]*np.cos(self.theta)*np.cos(self.phi) + v[1,...]*np.cos(self.theta)*np.sin(self.phi) - v[2,...]*np.sin(self.theta)
        vphi = -v[0,...]*np.sin(self.phi) + v[1,...]*np.cos(self.phi)
        return vtheta, vphi



    def div(self,vtheta,vphi):
        #Divergence of vector field on unit sphere, returned in spherical harmonics
        #See https://en.wikipedia.org/wiki/Vector_spherical_harmonics#Divergence
        S,T = self.sh.analys(vtheta, vphi)
        return -self.sh.l*(self.sh.l+1)*S

    def lap(self,f):
        #Laplacian of spherical harmonic array, returned in spherical harmonics
        #See https://en.wikipedia.org/wiki/Spherical_harmonics#Laplacian
        return -self.sh.l*(self.sh.l+1)*f
    


    def GBM(self,f):
        # return in harmonic from f*(D^* - <D^*>)
        f_spat = self.sh.synth(f)
        
        fDstar = self.sh.analys(f_spat*self.Dstar) # take advantage of spherical harmonic integration
        GBM_spat = f_spat*(self.Dstar - fDstar[0].real)
        return self.sh.analys(GBM_spat) 
    

    def RHS(self,f):
        #Right hand side of the ODE
        f_spat = self.sh.synth(f)
        fvth, fvph = self.vec_cart2sph(f_spat*self.v)
        divfv = self.div(fvth, fvph)
        return -divfv + self.effectiveSR*self.lamb*self.lap(f) +self.effectiveSR*self.beta*self.GBM(f)

    def RK4(self,f,dt):
        #4th order Runge-Kutta
        k1 = self.RHS(f)
        k2 = self.RHS(f + 0.5*dt*k1)
        k3 = self.RHS(f + 0.5*dt*k2)
        k4 = self.RHS(f + dt*k3)
        return f + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    
    def params(self,T):
        #Params from Richards et al. 2021, normalised to effective strain rate
        self.iota = 0.0259733*T + 1.95268104
        self.lamb = 0.00251776*T + 0.41244777
        self.beta = 0.35182521*T + 12.17066493
    

    def solve_constant(self,gradu,TorX,dt=0.01,tmax=1.0):
        self.dt = dt
        self.tmax = tmax
        self.t  = np.arange(0,tmax,dt)
        self.nsteps = len(self.t)
        self.f0 = self.sh.spec_array()
        self.f0[0] = 1.0

        self.f = np.zeros((self.nsteps,self.sh.nlm),dtype=np.complex128)
        self.f[0,...] = self.f0

        if np.isscalar(TorX):
            self.params(TorX)
        else:
            self.iota = TorX[0]
            self.lamb = TorX[1]
            self.beta = TorX[2]
        
        self.v_star_cal(gradu)
        for i in range(self.nsteps-1):
            self.f[i+1,...] = self.RK4(self.f[i,...],dt)

        return self.f

    def iterate(self,gradu,TorX,dt):
        if np.isscalar(TorX):
            self.params(TorX)
        else:
            self.iota = TorX[0]
            self.lamb = TorX[1]
            self.beta = TorX[2]
        
        self.v_star_cal(gradu)
        self.f = self.RK4(self.f,dt)
        return self.f
    

    def a2(self,f):
        if f.ndim==1:
            f = f.expand_dims(0)
        a=np.zeros((f.shape[0],3,3),dtype=complex) 

        a[:,0,0]=((0.33333333333333333333)+(0)*1j)*f[:,self.sh.idx(0,0)]+ ((-0.14907119849998597976)+(0)*1j)*f[:,self.sh.idx(2,0)]+ 2*((0.18257418583505537115)+(0)*1j)*f[:,self.sh.idx(2,2)]
        a[:,0,1]=2*((0)+(0.18257418583505537115)*1j)*f[:,self.sh.idx(2,2)]
        a[:,1,0]=a[:,0,1]
        a[:,0,2]=2*((-0.18257418583505537115)+(0)*1j)*f[:,self.sh.idx(2,1)]
        a[:,2,0]=a[:,0,2]
        a[:,1,1]=((0.33333333333333333333)+(0)*1j)*f[:,self.sh.idx(0,0)]+ ((-0.14907119849998597976)+(0)*1j)*f[:,self.sh.idx(2,0)]+ 2*((-0.18257418583505537115)+(0)*1j)*f[:,self.sh.idx(2,2)]
        a[:,1,2]=2*((0)+(-0.18257418583505537115)*1j)*f[:,self.sh.idx(2,1)]
        a[:,2,1]=a[:,1,2]
        a[:,2,2]=((0.33333333333333333333)+(0)*1j)*f[:,self.sh.idx(0,0)]+ ((0.29814239699997195952)+(0)*1j)*f[:,self.sh.idx(2,0)]
        
        a.squeeze()
        return a
    
    def a4(self,f):
        if f.ndim==1:
            f = f.expand_dims(0)
        a=np.zeros((f.shape[0],3,3,3,3),dtype=complex)

        a[:,0,0,0,0]=((0.2)+(0)*1j)*f[:,self.idx(0,0)]+ ((-0.12777531299998798265)+(0)*1j)*f[:,self.idx(2,0)]+ 2*((0.15649215928719031813)+(0)*1j)*f[:,self.idx(2,2)]\
            + ((0.028571428571428571429)+(0)*1j)*f[:,self.idx(4,0)]+ 2*((-0.030116930096841707924)+(0)*1j)*f[:,self.idx(4,2)]+ 2*((0.039840953644479787999)+(0)*1j)*f[:,self.idx(4,4)]
        
        a[:,1,0,0,0]=2*((0)+(0.078246079643595159065)*1j)*f[:,self.idx(2,2)]+ 2*((0)+(-0.015058465048420853962)*1j)*f[:,self.idx(4,2)]+ 2*((0)+(0.039840953644479787999)*1j)*f[:,self.idx(4,4)]
        a[:,0,1,0,0]=a[:,1,0,0,0]
        a[:,0,0,1,0]=a[:,1,0,0,0]
        a[:,0,0,0,1]=a[:,1,0,0,0]
        
        a[:,2,0,0,0]=2*((-0.078246079643595159065)+(0)*1j)*f[:,self.idx(2,1)]+ 2*((0.031943828249996995663)+(0)*1j)*f[:,self.idx(4,1)]+ 2*((-0.028171808490950552584)+(0)*1j)*f[:,self.idx(4,3)]
        a[:,0,2,0,0]=a[:,2,0,0,0]
        a[:,0,0,2,0]=a[:,2,0,0,0]
        a[:,0,0,0,2]=a[:,2,0,0,0]
        
        a[:,1,1,0,0]=((0.066666666666666666667)+(0)*1j)*f[:,self.idx(0,0)]+ ((-0.042591770999995994217)+(0)*1j)*f[:,self.idx(2,0)]+ ((0.0095238095238095238095)+(0)*1j)*f[:,self.idx(4,0)]+ 2*((-0.039840953644479787999)+(0)*1j)*f[:,self.idx(4,4)]
        a[:,1,0,1,0]=a[:,1,1,0,0]
        a[:,1,0,0,1]=a[:,1,1,0,0]
        a[:,0,1,1,0]=a[:,1,1,0,0]
        a[:,0,1,0,1]=a[:,1,1,0,0]
        a[:,0,0,1,1]=a[:,1,1,0,0]
        
        a[:,2,1,0,0]=2*((0)+(-0.026082026547865053022)*1j)*f[:,self.idx(2,1)]+ 2*((0)+(0.010647942749998998554)*1j)*f[:,self.idx(4,1)]+ 2*((0)+(-0.028171808490950552584)*1j)*f[:,self.idx(4,3)]
        a[:,2,0,1,0]=a[:,2,1,0,0]
        a[:,2,0,0,1]=a[:,2,1,0,0]
        a[:,1,2,0,0]=a[:,2,1,0,0]
        a[:,1,0,2,0]=a[:,2,1,0,0]
        a[:,1,0,0,2]=a[:,2,1,0,0]
        a[:,0,2,1,0]=a[:,2,1,0,0]
        a[:,0,2,0,1]=a[:,2,1,0,0]
        a[:,0,1,2,0]=a[:,2,1,0,0]
        a[:,0,1,0,2]=a[:,2,1,0,0]
        a[:,0,0,2,1]=a[:,2,1,0,0]
        a[:,0,0,1,2]=a[:,2,1,0,0]
        
        a[:,2,2,0,0]=((0.066666666666666666667)+(0)*1j)*f[:,self.idx(0,0)]+ ((0.021295885499997997109)+(0)*1j)*f[:,self.idx(2,0)]+ 2*((0.026082026547865053022)+(0)*1j)*f[:,self.idx(2,2)]\
            + ((-0.038095238095238095238)+(0)*1j)*f[:,self.idx(4,0)]+ 2*((0.030116930096841707924)+(0)*1j)*f[:,self.idx(4,2)]
        a[:,2,0,2,0]=a[:,2,2,0,0]
        a[:,2,0,0,2]=a[:,2,2,0,0]
        a[:,0,2,2,0]=a[:,2,2,0,0]
        a[:,0,2,0,2]=a[:,2,2,0,0]
        a[:,0,0,2,2]=a[:,2,2,0,0]
        
        a[:,1,1,1,0]=2*((0)+(0.078246079643595159065)*1j)*f[:,self.idx(2,2)]+ 2*((0)+(-0.015058465048420853962)*1j)*f[:,self.idx(4,2)]+ 2*((0)+(-0.039840953644479787999)*1j)*f[:,self.idx(4,4)]
        a[:,1,1,0,1]=a[:,1,1,1,0]
        a[:,1,0,1,1]=a[:,1,1,1,0]
        a[:,0,1,1,1]=a[:,1,1,1,0]
        
        a[:,2,1,1,0]=2*((-0.026082026547865053022)+(0)*1j)*f[:,self.idx(2,1)]+ 2*((0.010647942749998998554)+(0)*1j)*f[:,self.idx(4,1)]+ 2*((0.028171808490950552584)+(0)*1j)*f[:,self.idx(4,3)]
        a[:,2,1,0,1]=a[:,2,1,1,0]
        a[:,2,0,1,1]=a[:,2,1,1,0]
        a[:,1,2,1,0]=a[:,2,1,1,0]
        a[:,1,2,0,1]=a[:,2,1,1,0]
        a[:,1,1,2,0]=a[:,2,1,1,0]
        a[:,1,1,0,2]=a[:,2,1,1,0]
        a[:,1,0,2,1]=a[:,2,1,1,0]
        a[:,1,0,1,2]=a[:,2,1,1,0]
        a[:,0,2,1,1]=a[:,2,1,1,0]
        a[:,0,1,2,1]=a[:,2,1,1,0]
        a[:,0,1,1,2]=a[:,2,1,1,0]
        
        a[:,2,2,1,0]=2*((0)+(0.026082026547865053022)*1j)*f[:,self.idx(2,2)]+ 2*((0)+(0.030116930096841707924)*1j)*f[:,self.idx(4,2)]
        a[:,2,2,0,1]=a[:,2,2,1,0]
        a[:,2,1,2,0]=a[:,2,2,1,0]
        a[:,2,1,0,2]=a[:,2,2,1,0]
        a[:,2,0,2,1]=a[:,2,2,1,0]
        a[:,2,0,1,2]=a[:,2,2,1,0]
        a[:,1,2,2,0]=a[:,2,2,1,0]
        a[:,1,2,0,2]=a[:,2,2,1,0]
        a[:,1,0,2,2]=a[:,2,2,1,0]
        a[:,0,2,2,1]=a[:,2,2,1,0]
        a[:,0,2,1,2]=a[:,2,2,1,0]
        a[:,0,1,2,2]=a[:,2,2,1,0]
        
        a[:,2,2,2,0]=2*((-0.078246079643595159065)+(0)*1j)*f[:,self.idx(2,1)]+ 2*((-0.042591770999995994217)+(0)*1j)*f[:,self.idx(4,1)]
        a[:,2,2,0,2]=a[:,2,2,2,0]
        a[:,2,0,2,2]=a[:,2,2,2,0]
        a[:,0,2,2,2]=a[:,2,2,2,0]
        
        a[:,1,1,1,1]=((0.2)+(0)*1j)*f[:,self.idx(0,0)]+ ((-0.12777531299998798265)+(0)*1j)*f[:,self.idx(2,0)]+ 2*((-0.15649215928719031813)+(0)*1j)*f[:,self.idx(2,2)]\
            + ((0.028571428571428571429)+(0)*1j)*f[:,self.idx(4,0)]+ 2*((0.030116930096841707924)+(0)*1j)*f[:,self.idx(4,2)]+ 2*((0.039840953644479787999)+(0)*1j)*f[:,self.idx(4,4)]
        
        a[:,2,1,1,1]=2*((0)+(-0.078246079643595159065)*1j)*f[:,self.idx(2,1)]+ 2*((0)+(0.031943828249996995663)*1j)*f[:,self.idx(4,1)]+ 2*((0)+(0.028171808490950552584)*1j)*f[:,self.idx(4,3)]
        a[:,1,2,1,1]=a[:,2,1,1,1]
        a[:,1,1,2,1]=a[:,2,1,1,1]
        a[:,1,1,1,2]=a[:,2,1,1,1]
        
        a[:,2,2,1,1]=((0.066666666666666666667)+(0)*1j)*f[:,self.idx(0,0)]+ ((0.021295885499997997109)+(0)*1j)*f[:,self.idx(2,0)]+ 2*((-0.026082026547865053022)+(0)*1j)*f[:,self.idx(2,2)]+ ((-0.038095238095238095238)+(0)*1j)*f[:,self.idx(4,0)]+ 2*((-0.030116930096841707924)+(0)*1j)*f[:,self.idx(4,2)]
        a[:,2,1,2,1]=a[:,2,2,1,1]
        a[:,2,1,1,2]=a[:,2,2,1,1]
        a[:,1,2,2,1]=a[:,2,2,1,1]
        a[:,1,2,1,2]=a[:,2,2,1,1]
        a[:,1,1,2,2]=a[:,2,2,1,1]
        
        a[:,2,2,2,1]=2*((0)+(-0.078246079643595159065)*1j)*f[:,self.idx(2,1)]+ 2*((0)+(-0.042591770999995994217)*1j)*f[:,self.idx(4,1)]
        a[:,2,2,1,2]=a[:,2,2,2,1]
        a[:,2,1,2,2]=a[:,2,2,2,1]
        a[:,1,2,2,2]=a[:,2,2,2,1]
        
        a[:,2,2,2,2]=((0.2)+(0)*1j)*f[:,self.idx(0,0)]+ ((0.2555506259999759653)+(0)*1j)*f[:,self.idx(2,0)]+ ((0.076190476190476190476)+(0)*1j)*f[:,self.idx(4,0)]
        
        a = a.squeeze()
        return a
    


class plotting:
    def __init__(self,sh):
        from matplotlib import pyplot as plt
        self.sh = sh
        self.nlats, self.nlons = self.sh.set_grid(50,50,\
                shtns.SHT_PHI_CONTIGUOUS,1.e-10)
    
        self.theta_vals = np.arccos(self.sh.cos_theta)
        self.phi_vals = (2.0*np.pi/self.nlons)*np.arange(self.nlons)
        self.phi, self.theta = np.meshgrid(self.phi_vals, self.theta_vals)

        self.x = np.sin(self.theta)*np.cos(self.phi)
        self.y = np.sin(self.theta)*np.sin(self.phi)
        self.z = np.cos(self.theta)


    def polefigure(self,f):
        #plot pole figure of f
        fgrid = self.sh.synth(f)
        fgrid = fgrid.real

        # Equidistant 
        xx=self.theta*np.cos(self.phi)
        yy=self.theta*np.sin(self.phi)

        xx=xx[self.theta_vals<=np.pi/2,:]
        yy=yy[self.theta_vals<=np.pi/2,:]
        fgrid=fgrid[self.theta_vals<=np.pi/2,:]
        
        xx = np.hstack([xx, xx[:, 0][:, None]])
        yy = np.hstack([yy, yy[:, 0][:, None]])
        fgrid = np.hstack([fgrid, fgrid[:, 0][:, None]])

        return xx,yy,fgrid
    
    def sphericalplot(self,f):
        
        lon = self.phi*180/np.pi
        lon[lon>180] = lon[lon>180] - 360
        lat = -self.theta*180/np.pi + 90

        fgrid = self.sh.synth(f)

        lat = np.hstack([lat, lat[:, 0][:, None]])
        lon = np.hstack([lon, lon[:, 0][:, None]])
        fgrid = np.hstack([fgrid, fgrid[:, 0][:, None]])

        return lat,lon,fgrid
    
    def plot_polefigure(self,ax,f,**kwargs):

        xx,yy,fgrid = self.polefigure(f)
        ax.pcolormesh(xx,yy,fgrid,**kwargs)
        ax.set_aspect('equal')
        ax.axis('off')



    