import numpy as np
from scipy.special import jv, spherical_jn
from scipy.ndimage import convolve1d
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-16
# Updated: 2021-11-17

twopi = 2*np.pi

def phi2func(eta):
    return np.piecewise(eta,[eta<=1e-3,eta>1e-3],[lambda eta: 1+eta**2/9,lambda eta: 1+(2*eta-eta**2+2*np.log(1-eta)*(1-eta))/(3*eta)])

def phi3func(eta):
    return np.piecewise(eta,[eta<=1e-3,eta>1e-3],[lambda eta: 1-4*eta/9,lambda eta: 1-(2*eta-3*eta**2+2*eta**3+2*np.log(1-eta)*(1-eta)**2)/(3*eta**2)])

def phi1func(eta):
    return np.piecewise(eta,[eta<=1e-3,eta>1e-3],[lambda eta: 1-2*eta/9-eta**2/18,lambda eta: 2*(eta+np.log(1-eta)*(1-eta)**2)/(3*eta**2)])

def dphi1dnfunc(eta):
    return np.piecewise(eta,[eta<=1e-3,eta>1e-3],[lambda eta: -2/9-eta/9-eta**2/15.0,lambda eta: (2*(eta-2)*eta+4*(eta-1)*np.log(1-eta))/(3*eta**3)])

def dphi2dnfunc(eta):
    return np.piecewise(eta,[eta<=1e-3,eta>1e-3],[lambda eta: 2*eta/9+eta**2/6.0,lambda eta: -(2*eta+eta**2+2*np.log(1-eta))/(3*eta**2)])

def dphi3dnfunc(eta):
    return np.piecewise(eta,[eta<=1e-3,eta>1e-3],[lambda eta: -4.0/9+eta/9,lambda eta: -2*(1-eta)*(eta*(2+eta)+2*np.log(1-eta))/(3*eta**3)])


####################################################
### The FMT Functional on 1d slab geometry       ###
####################################################
class FMTplanar():
    def __init__(self,N,delta,species=1,d=np.array([1.0]),method='WBI'):
        self.method = method
        self.N = N
        self.delta = delta
        self.L = N*delta
        self.d = d
        self.species = species

        self.n3 = np.empty(self.N,dtype=np.float32)
        self.n2 = np.empty(self.N,dtype=np.float32)
        self.n2vec = np.empty(self.N,dtype=np.float32)

        self.w3 = np.empty(self.species,dtype=object)
        self.w2 = np.empty(self.species,dtype=object)
        self.w2vec = np.empty(self.species,dtype=object)
        self.c1array = np.empty((self.species,self.N),dtype=np.float32)

        for i in range(self.species):
            nd = int(self.d[i]/self.delta)+1
            x = np.linspace(-0.5*self.d[i],0.5*self.d[i],nd)

            self.w3[i] = np.pi*((0.5*self.d[i])**2-x**2)
            self.w2[i] = self.d[i]*np.pi*np.ones(nd)
            self.w2vec[i] = twopi*x

    def weighted_densities(self,rho):
        if self.species == 1:
            self.n3[:] = convolve1d(rho, weights=self.w3[0], mode='nearest')*self.delta
            self.n2[:] = convolve1d(rho, weights=self.w2[0], mode='nearest')*self.delta
            self.n2vec[:] = convolve1d(rho, weights=self.w2vec[0], mode='nearest')*self.delta
            self.n1vec = self.n2vec/(twopi*self.d[0])
            self.n0 = self.n2/(np.pi*self.d[0]**2)
            self.n1 = self.n2/(twopi*self.d[0])

        else:
            self.n3[:] = convolve1d(rho[0], weights=self.w3[0], mode='nearest')*self.delta
            self.n2[:] = convolve1d(rho[0], weights=self.w2[0], mode='nearest')*self.delta
            self.n2vec[:] = convolve1d(rho[0], weights=self.w2vec[0], mode='nearest')*self.delta
            self.n1vec = self.n2vec/(twopi*self.d[0])
            self.n0 = self.n2/(np.pi*self.d[0]**2)
            self.n1 = self.n2/(twopi*self.d[0])
            
            for i in range(1,self.species):
                self.n3[:] += convolve1d(rho[i], weights=self.w3[i], mode='nearest')*self.delta
                n2 = convolve1d(rho[i], weights=self.w2[i], mode='nearest')*self.delta
                n2vec = convolve1d(rho[i], weights=self.w2vec[i], mode='nearest')*self.delta
                self.n2[:] += n2
                self.n2vec[:] += n2vec
                self.n1vec[:] += n2vec/(twopi*self.d[i])
                self.n0[:] += n2/(np.pi*self.d[i]**2)
                self.n1[:] += n2/(twopi*self.d[i])
            
        self.oneminusn3 = 1-self.n3

        if self.method == 'RF' or self.method == 'WBI': 
            self.phi2 = 1.0
            self.dphi2dn3 = 0.0
        elif self.method == 'WBII': 
            self.phi2 = phi2func(self.n3)
            self.dphi2dn3 = dphi2dnfunc(self.n3)

        if self.method == 'WBI': 
            self.phi3 = phi1func(self.n3)
            self.dphi3dn3 = dphi1dnfunc(self.n3)
        elif self.method == 'WBII': 
            self.phi3 = phi3func(self.n3)
            self.dphi3dn3 = dphi3dnfunc(self.n3)
        else:
            self.phi3 = 1.0
            self.dphi3dn3 = 0.0

    def Phi(self,rho):
        self.weighted_densities(rho)

        return -self.n0*np.log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec*self.n2vec)) + (self.phi3/(24*np.pi*self.oneminusn3**2))*(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec*self.n2vec))

    def F(self,rho):
        return np.sum(self.Phi(rho))*self.delta

    def c1(self,rho):
        self.weighted_densities(rho)

        self.dPhidn0 = -np.log(self.oneminusn3 )
        self.dPhidn1 = self.n2*self.phi2/self.oneminusn3
        self.dPhidn2 = self.n1*self.phi2/self.oneminusn3  + (3*self.n2*self.n2-3*(self.n2vec*self.n2vec))*self.phi3/(24*np.pi*self.oneminusn3**2)

        self.dPhidn3 = self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec*self.n2vec))*(self.dphi2dn3 + self.phi2/self.oneminusn3)/self.oneminusn3 + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec*self.n2vec))*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/(24*np.pi*self.oneminusn3**2)

        self.dPhidn1vec0 = -self.n2vec*self.phi2/self.oneminusn3 
        self.dPhidn2vec0 = -self.n1vec*self.phi2/self.oneminusn3  - self.n2*self.n2vec*self.phi3/(4*np.pi*self.oneminusn3**2)

        for i in range(self.species):
            self.c1array[i] = -convolve1d(self.dPhidn2 + self.dPhidn1/(twopi*self.d[i]) + self.dPhidn0/(np.pi*self.d[i]**2), weights=self.w2[i], mode='nearest')*self.delta - convolve1d(self.dPhidn3, weights=self.w3[i], mode='nearest')*self.delta + convolve1d(self.dPhidn2vec0+self.dPhidn1vec0/(twopi*self.d[i]), weights=self.w2vec[i], mode='nearest')*self.delta

        del self.dPhidn0,self.dPhidn1,self.dPhidn2,self.dPhidn3,self.dPhidn1vec0,self.dPhidn2vec0,

        if (self.species==1): return self.c1array[0]
        else: return self.c1array

    def mu(self,rhob):
        n3 = np.sum(rhob*np.pi*self.d**3/6)
        n2 = np.sum(rhob*np.pi*self.d**2)
        n1 = np.sum(rhob*self.d/2)
        n0 = np.sum(rhob)

        if self.method == 'RF' or self.method == 'WBI': 
            phi2 = 1.0
            dphi2dn3 = 0.0
        elif self.method == 'WBII': 
            phi2 = phi2func(n3)
            dphi2dn3 = dphi2dnfunc(n3)

        if self.method == 'WBI': 
            phi3 = phi1func(n3)
            dphi3dn3 = dphi1dnfunc(n3)
        elif self.method == 'WBII': 
            phi3 = phi3func(n3)
            dphi3dn3 = dphi3dnfunc(n3)
        else: 
            phi3 = 1.0
            dphi3dn3 = 0.0

        dPhidn0 = -np.log(1-n3)
        dPhidn1 = n2*phi2/(1-n3)
        dPhidn2 = n1*phi2/(1-n3) + (3*n2**2)*phi3/(24*np.pi*(1-n3)**2)
        dPhidn3 = n0/(1-n3) +(n1*n2)*(dphi2dn3 + phi2/(1-n3))/(1-n3) + (n2**3)*(dphi3dn3+2*phi3/(1-n3))/(24*np.pi*(1-n3)**2)

        if (self.species==1): return (dPhidn0+dPhidn1*self.d/2+dPhidn2*np.pi*self.d**2+dPhidn3*np.pi*self.d**3/6)[0]
        else: return (dPhidn0+dPhidn1*self.d/2+dPhidn2*np.pi*self.d**2+dPhidn3*np.pi*self.d**3/6)


####################################################
class FMTspherical():
    def __init__(self,N,delta,d=1.0,method='WBI'):
        self.method = method
        self.N = N
        self.delta = delta
        self.L = N*delta
        self.d = d

        self.n3 = np.empty(self.N,dtype=np.float32)
        self.n2 = np.empty(self.N,dtype=np.float32)
        self.n2vec = np.empty(self.N,dtype=np.float32)

        self.r = np.linspace(0,self.L,self.N)
        self.rmed = self.r + 0.1*self.delta

        self.nsig = int(self.d/self.delta)

        self.w3 = np.zeros(self.nsig,dtype=np.float32)
        self.w2 = np.zeros(self.nsig,dtype=np.float32)
        self.w2vec = np.zeros(self.nsig,dtype=np.float32)
        
        r = np.linspace(-0.5*self.d,0.5*self.d,self.nsig)
        
        self.w3[:] = np.pi*((0.5*self.d)**2-r**2)
        self.w2[:] = self.d*np.pi
        self.w2vec[:] = twopi*r

    def weighted_densities(self,rho):
        self.n3[:] = convolve1d(rho*self.r, weights=self.w3, mode='nearest')*self.delta/self.rmed
        self.n2[:] = convolve1d(rho*self.r, weights=self.w2, mode='nearest')*self.delta/self.rmed
        self.n2vec[:] = self.n3/self.rmed + convolve1d(rho*self.r, weights=self.w2vec, mode='nearest')*self.delta/self.rmed

        self.n1vec = self.n2vec/(twopi*self.d)
        self.n0 = self.n2/(np.pi*self.d**2)
        self.n1 = self.n2/(twopi*self.d)
        self.oneminusn3 = 1-self.n3

        if self.method == 'RF' or self.method == 'WBI': 
            self.phi2 = 1.0
            self.dphi2dn3 = 0.0
        elif self.method == 'WBII': 
            self.phi2 = phi2func(self.n3)
            self.dphi2dn3 = dphi2dnfunc(self.n3)

        if self.method == 'WBI': 
            self.phi3 = phi1func(self.n3)
            self.dphi3dn3 = dphi1dnfunc(self.n3)
        elif self.method == 'WBII': 
            self.phi3 = phi3func(self.n3)
            self.dphi3dn3 = dphi3dnfunc(self.n3)
        else:
            self.phi3 = 1.0
            self.dphi3dn3 = 0.0

    def F(self,rho):
        self.weighted_densities(rho)

        Phi = -self.n0*np.log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec*self.n2vec)) + (self.phi3/(24*np.pi*self.oneminusn3**2))*(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec*self.n2vec))

        return np.sum(Phi*4*np.pi*self.r**2)*self.delta

    def dPhidn(self,rho):
        self.weighted_densities(rho)

        self.dPhidn0 = -np.log(self.oneminusn3 )
        self.dPhidn1 = self.n2*self.phi2/self.oneminusn3
        self.dPhidn2 = self.n1*self.phi2/self.oneminusn3  + (3*self.n2*self.n2-3*(self.n2vec*self.n2vec))*self.phi3/(24*np.pi*self.oneminusn3**2)

        self.dPhidn3 = self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec*self.n2vec))*(self.dphi2dn3 + self.phi2/self.oneminusn3)/self.oneminusn3 + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec*self.n2vec))*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/(24*np.pi*self.oneminusn3**2)

        self.dPhidn1vec0 = -self.n2vec*self.phi2/self.oneminusn3 
        self.dPhidn2vec0 = -self.n1vec*self.phi2/self.oneminusn3  - self.n2*self.n2vec*self.phi3/(4*np.pi*self.oneminusn3**2)

        dPhidn = convolve1d((self.dPhidn2 + self.dPhidn1/(twopi*self.d) + self.dPhidn0/(np.pi*self.d**2))*self.r, weights=self.w2, mode='nearest')*self.delta/self.rmed
        dPhidn += convolve1d(self.dPhidn3*self.r, weights=self.w3, mode='nearest')*self.delta/self.rmed
        dPhidn += convolve1d((self.dPhidn2vec0+self.dPhidn1vec0/(twopi*self.d))*self.r, weights=self.w3, mode='nearest')*self.delta/(self.rmed)**2 - convolve1d((self.dPhidn2vec0+self.dPhidn1vec0/(twopi*self.d))*self.r, weights=self.w2vec, mode='nearest')*self.delta/self.rmed

        del self.dPhidn0,self.dPhidn1,self.dPhidn2,self.dPhidn3,self.dPhidn1vec0,self.dPhidn2vec0
        
        return dPhidn

    def c1(self,rho):
        return -self.dPhidn(rho)

    def mu(self,rhob):
        n3 = rhob*np.pi*self.d**3/6
        n2 = rhob*np.pi*self.d**2
        n1 = rhob*self.d/2
        n0 = rhob

        if self.method == 'RF' or self.method == 'WBI': 
            phi2 = 1.0
            dphi2dn3 = 0.0
        elif self.method == 'WBII': 
            phi2 = phi2func(n3)
            dphi2dn3 = dphi2dnfunc(n3)

        if self.method == 'WBI': 
            phi3 = phi1func(n3)
            dphi3dn3 = dphi1dnfunc(n3)
        elif self.method == 'WBII': 
            phi3 = phi3func(n3)
            dphi3dn3 = dphi3dnfunc(n3)
        else: 
            phi3 = 1.0
            dphi3dn3 = 0.0

        dPhidn0 = -np.log(1-n3)
        dPhidn1 = n2*phi2/(1-n3)
        dPhidn2 = n1*phi2/(1-n3) + (3*n2**2)*phi3/(24*np.pi*(1-n3)**2)
        dPhidn3 = n0/(1-n3) +(n1*n2)*(dphi2dn3 + phi2/(1-n3))/(1-n3) + (n2**3)*(dphi3dn3+2*phi3/(1-n3))/(24*np.pi*(1-n3)**2)

        return (dPhidn0+dPhidn1*self.d/2+dPhidn2*np.pi*self.d**2+dPhidn3*np.pi*self.d**3/6)

def Theta(x,y,z,xi,yi,zi):
    return np.where(((x-xi)**2+(y-yi)**2+(z-zi)**2)<=0.25,1.0,0.0)


##### Take a example using FMT ######
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    
    #############################
    test1 = False # hard wall (1D-planar)
    test2 = True # hard wall mixture (1D-planar)
    test3 = False # hard-sphere (1D-spherical)

    ######################################################
    if test1:
        delta = 0.01
        N = 600
        L = N*delta
        fmt = FMTplanar(N,delta)
        etaarray = np.array([0.4257,0.4783])

        nsig = int(0.5/delta)

        n = 0.8*np.ones((1,N),dtype=np.float32)
        n[0,:nsig] = 1.0e-12
        # n[N-nsig:] = 1.0e-12

        lnn = np.log(n)

        x = np.linspace(0,L,N)
            
        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            Fid = np.trapz(n[0]*(lnn[0]-1.0),dx=delta)
            Fex = np.trapz(fmt.Phi(n),dx=delta)
            Nmu = np.trapz(mu[0]*n[0],dx=delta)
            return (Fid+Fex-Nmu)/L

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            dphidn = -fmt.c1(n)
            return n*(lnn + dphidn - mu[0])*delta/L

        for eta in etaarray:

            rhob = eta/(np.pi/6.0)
            mu = np.array([np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)])
        
            [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,1.0e-10,0.05,True)

            n[:] = np.exp(nsol)
            nmean = np.trapz(n,dx=delta)/L
            print('rhob=',rhob,'\n nmean = ',nmean[0],'\n Omega/N =',Omegasol)

            # np.save('fmt-rf-slitpore-eta'+str(eta)+'-N-'+str(N)+'.npy',[x,n[:,N//2,N//2]/rhob])

            # [zRF,rhoRF] = np.load('fmt-wbii-slitpore-eta'+str(eta)+'-N-'+str(N)+'.npy') 

            data = np.loadtxt('../MCdataHS/hardwall-eta'+str(eta)+'.dat')
            [xdata, rhodata] = [data[:,0],data[:,1]]
            plt.scatter(xdata,rhodata,marker='o',edgecolors='C0',facecolors='none',label='MC')
            plt.plot(x,n[0],label='DFT')
            plt.xlabel(r'$x/\d$')
            plt.ylabel(r'$\rho(x) \d^3$')
            plt.xlim(0.5,3)
            # plt.ylim(0.0,7)
            plt.legend(loc='best')
            plt.savefig('hardwall-eta'+str(eta)+'.png', bbox_inches='tight')
            plt.show()
            plt.close()

    if test2:
        delta = 0.01
        d = np.array([1.0,3.0])
        L = 10*d[0]
        N = int(L/delta)

        fmt = FMTplanar(N,delta,species=2,d=d)

        n = np.empty((2,N),dtype=np.float32)
        nn = n.copy()
        nsig = np.array([int(0.5*d[0]/delta),int(0.5*d[1]/delta)])

        x = np.linspace(0,L,N)

        def Omega(lnn,mu):
            nn[:] = np.exp(lnn)
            Fid = np.sum(nn*(lnn-1.0))*delta
            Fex = np.sum(fmt.Phi(nn)*delta)
            return (Fid+Fex-np.sum(mu[:,np.newaxis]*nn)*delta)/L

        def dOmegadnR(lnn,mu):
            nn[:] = np.exp(lnn)
            c1fmt = fmt.c1(nn)
            return nn*(lnn -c1fmt - mu[:,np.newaxis])*delta/L

        def rhofunc(lnn,mu):
            nn = np.exp(lnn)
            c1hs = fmt.c1(nn)
            lnntilde = c1hs + mu[:,np.newaxis]
            lnntilde[0,:nsig[0]] = -16
            lnntilde[1,:nsig[1]] = -16
            return lnntilde

        eta = 0.39
        x1 = 0.25
        r = d[0]/d[1]
        rhob = np.array([eta/(np.pi*d[0]**3*(1+(1-x1)/x1/r**3)/6), eta/(np.pi*d[0]**3*(1+(1-x1)/x1/r**3)/6)*(1-x1)/x1])
        
        n[0,nsig[0]:] = rhob[0]
        n[1,nsig[1]:] = rhob[1]
        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16

        mu = np.log(rhob) + fmt.mu(rhob)

        print('mu =',mu)
        print('rhob=',rhob)
    
        var = np.log(n)
        [varsol,Omegasol,Niter] = optimize_anderson(var,rhofunc,Omega,dOmegadnR,mu,beta=0.25,atol=1e-7,logoutput=True)
        # [varsol,Omegasol,Niter] = optimize_fire2(var,Omega,dOmegadnR,mu,atol=1.0e-6,dt=40.0,logoutput=True)
        n[:] = np.exp(varsol)

        nmean = np.sum(n,axis=1)*delta/L
        print('mu =',mu)
        print('rhob=',rhob)
        print('nmean = ',nmean)
        print('Omega/L =',Omegasol)

        # data from: NOWORYTA, JERZY P., et al. “Hard Sphere Mixtures near a Hard Wall.” Molecular Physics, vol. 95, no. 3, Oct. 1998, pp. 415–24, doi:10.1080/00268979809483175.
        preamble = 'MCdata/'
        # name = 'hardwall-mixture-eta0.12-x1_0.83-ratio3'
        name = 'hardwall-mixture-eta0.39-x10.25-ratio3'
        data = np.loadtxt(preamble+name+'-small.dat')
        [xdata, rhodata] = [data[:,0],data[:,1]]
        plt.scatter(xdata,rhodata,marker='o',edgecolors='C0',facecolors='none',label='MC')
        plt.plot(x,n[0]/rhob[0],label='DFT')
        plt.xlabel(r'$x/d_1$')
        plt.ylabel(r'$\rho(x)/\rho_b$')
        plt.xlim(0.5,8)
        plt.ylim(0.0,3.0)
        plt.legend(loc='upper right')
        plt.savefig(name+'-small.png', bbox_inches='tight')
        plt.show()
        plt.close()

        data = np.loadtxt(preamble+name+'-big.dat')
        [xdata, rhodata] = [data[:,0],data[:,1]]
        plt.scatter(xdata,rhodata,marker='o',edgecolors='C1',facecolors='none',label='MC')
        plt.plot(x,n[1]/rhob[1],'C1',label='DFT')
        plt.xlabel(r'$x/d_1$')
        plt.ylabel(r'$\rho(x)/\rho_b$')
        plt.xlim(1.5,8)
        plt.ylim(0.0,7)
        plt.legend(loc='upper right')
        plt.savefig(name+'-big.png', bbox_inches='tight')
        plt.show()
        plt.close()

    ################################################################
    if test3:
        delta = 0.01
        N = 300
        L = N*delta
        fmt = FMTspherical(N,delta)
        rhobarray = np.array([0.7,0.8,0.9])

        nsig = int(1.5/delta)

        n = 1.0e-16*np.ones(N,dtype=np.float32)
        n[:nsig] = 3/np.pi
        # n[N-nsig:] = 1.0e-12

        r = 0.5*delta+np.linspace(0,L,N)
        Vol = 4*np.pi*L**3/3
            
        def Omega(n,mu):
            phi = fmt.Phi(n)
            Omegak = n*(np.log(n)-1.0) + phi - mu*n
            return np.sum(4*np.pi*r**2*Omegak*delta)/Vol

        def dOmegadnR(n,mu):
            dphidn = fmt.dPhidn(n)
            return 4*np.pi*r**2*(np.log(n) + dphidn - mu)*delta/Vol

        for rhob in rhobarray:

            eta = rhob*(np.pi/6.0)
            mu = np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)
        
            [nsol,Omegasol,Niter] = optimize_fire2(n,Omega,dOmegadnR,mu,1.0e-9,0.001,True)

            nmean = np.sum(nsol*4*np.pi*r**2*delta)/Vol
            print('rhob=',rhob,'\n nmean = ',nmean,'\n Omega/N =',Omegasol)

            # data = np.loadtxt('../MCdataHS/hardwall-eta'+str(eta)+'.dat')
            # [xdata, rhodata] = [data[:,0],data[:,1]]
            # plt.scatter(xdata,rhodata,marker='o',edgecolors='C0',facecolors='none',label='MC')
            plt.plot(r,n/rhob,label='DFT')
            plt.xlabel(r'$r/\d$')
            plt.ylabel(r'$g(r)$')
            plt.xlim(1.0,2.2)
            plt.ylim(0.5,6)
            plt.legend(loc='best')
            plt.savefig('hardsphere-eta'+str(eta)+'.png', bbox_inches='tight')
            plt.show()
            plt.close()
            