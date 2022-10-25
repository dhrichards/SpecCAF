#%%
import numpy as np
import speccaf.spherical as spherical


def build(sh):
    

    
    # Initialise arrays
    
    R =  np.zeros((sh.nlm,sh.nlm,3,3),dtype=complex)
    Ri= np.zeros_like(R)
    B =  np.zeros((sh.nlm,sh.nlm,3,3),dtype=complex)
    Bi= np.zeros_like(B)
    G =  np.zeros((sh.nlm,sh.nlm,3,3,3,3),dtype=complex)
    Gi= np.zeros_like(G)
    
    
    
    def Bms(f,gradu):
        row =  np.zeros((sh.nlm),dtype=complex)
        D = 0.5*(gradu+gradu.T)
        
        
        
        for i in range(sh.nlm):
            l=sh.l[i]
            m=sh.m[i]
            
            row[i] = D[0,0]*sh.xdx(f,l,m) + D[0,1]*sh.xdy(f,l,m) + D[0,2]*sh.xdz(f,l,m)\
                    +D[1,0]*sh.ydx(f,l,m) + D[1,1]*sh.ydy(f,l,m) + D[1,2]*sh.ydz(f,l,m)\
                    +D[2,0]*sh.zdx(f,l,m) + D[2,1]*sh.zdy(f,l,m) + D[2,2]*sh.zdz(f,l,m)
        return row
    
    
    def Wms(f,gradu):
        row =  np.zeros((sh.nlm),dtype=complex)
        wx = gradu[2,1] - gradu[1,2]
        wy = gradu[0,2] - gradu[2,0]
        wz = gradu[1,0] - gradu[0,1]
        
        
        for t in range(sh.nlm):
            l=sh.l[t]
            m=sh.m[t]
            
            row[t] = 0.5*wx*( sh.ydz(f,l,m)-sh.zdy(f,l,m) )\
                    +0.5*wy*( sh.zdx(f,l,m)-sh.xdz(f,l,m) )\
                    +0.5*wz*( sh.xdy(f,l,m)-sh.ydx(f,l,m) )
        return row
        
    
    def Gms(f,Q):
        #Qijkl = DijDkl
        row =  np.zeros((sh.nlm),dtype=complex)
        
        for v in range(sh.nlm):
            l=sh.l[v]
            m=sh.m[v]
            row[v] =  ((Q[0,0,0,0]+Q[1,0,1,0]+Q[2,0,2,0])*sh.xx(f,l,m) + (Q[0,1,0,1]+Q[1,1,1,1]+Q[2,1,2,1])*sh.yy(f,l,m) + (Q[0,2,0,2]+Q[1,2,1,2]+Q[2,2,2,2])*sh.zz(f,l,m) \
                        +2*(Q[0,0,0,1] + Q[1,0,1,1] + Q[2,0,2,1])*sh.xy(f,l,m) + 2*(Q[0,0,0,2] + Q[1,0,1,2] + Q[2,0,2,2])*sh.xz(f,l,m) \
                        +2*(Q[0,1,0,2] + Q[1,1,1,2] + Q[2,1,2,2])*sh.yz(f,l,m))\
                        - ( Q[0,0,0,0]*sh.xxxx(f,l,m) + Q[1,1,1,1]*sh.yyyy(f,l,m) + Q[2,2,2,2]*sh.zzzz(f,l,m) + 4*Q[0,1,0,1]*sh.xxyy(f,l,m) + 4*Q[0,2,0,2]*sh.xxzz(f,l,m) + 4*Q[1,2,1,2]*sh.yyzz(f,l,m) + \
                        + 2*Q[0,0,1,1]*sh.xxyy(f,l,m) + 2*Q[0,0,2,2]*sh.xxzz(f,l,m) + 4*Q[0,0,0,1]*sh.xxxy(f,l,m) + 4*Q[0,0,0,2]*sh.xxxz(f,l,m) + 4*Q[0,0,1,2]*sh.xxyz(f,l,m) \
                        + 2*Q[1,1,2,2]*sh.yyzz(f,l,m) + 4*Q[1,1,0,1]*sh.xyyy(f,l,m) + 4*Q[1,1,0,2]*sh.xyyz(f,l,m) + 4*Q[1,1,1,2]*sh.yyyz(f,l,m) \
                        + 4*Q[2,2,0,1]*sh.xyzz(f,l,m) + 4*Q[2,2,0,2]*sh.xzzz(f,l,m) + 4*Q[2,2,1,2]*sh.yzzz(f,l,m) \
                        + 8*Q[0,1,0,2]*sh.xxyz(f,l,m) + 8*Q[0,1,1,2]*sh.xyyz(f,l,m) \
                        + 8*Q[0,2,1,2]*sh.xyzz(f,l,m))
        return row
    
    for j in range(sh.nlm):

        # Initilase spectral array and set fj=1, 0 otherwise
        f = sh.spec_array()
        g= sh.spec_array()
        f[j]=1.0
        g[j]=1j
        #Fill matrix
        
        for p in range(3):
            for q in range(3):
                
                gradu =  np.zeros((3,3))
                gradu[p,q]=1
                
                #R[:,j,p,q] = sh.analys(Rlam(theta,phi,fgrid,df_dtheta,df_dphi,gradu.flatten()))
                #B[:,j,p,q] = sh.analys(Blam(theta,phi,fgrid,df_dtheta,df_dphi,gradu.flatten()))
                R[:,j,p,q] = Wms(f,gradu)
                Ri[:,j,p,q] = Wms(g,gradu)
                B[:,j,p,q] = Bms(f,gradu)
                Bi[:,j,p,q] = Bms(g,gradu)
                for r in range(3):
                    for s in range(3):
                        
                        Q= np.zeros((3,3,3,3))
                        Q[p,q,r,s] = 1 
                        
                        #G[:,j,p,q,r,s] = sh.analys(fgrid*Deftop(theta,phi,Q))
                        G[:,j,p,q,r,s] = Gms(f,Q)
                        Gi[:,j,p,q,r,s] = Gms(g,Q)
    
    np.savez('MatricesL' + str(sh.lmax) +'.npz',R=R,Ri=Ri,B=B,Bi=Bi,G=G,Gi=Gi)    
    
    return R,Ri,B,Bi,G,Gi


L=20
sh = spherical.spherical(L)
R,Ri,B,Bi,G,Gi = build(sh)
