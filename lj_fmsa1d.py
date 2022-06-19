#!/usr/bin/env python3

# This script is the python implementation of the Density Functional Theory
# for the multiple Yukawa hard-core fluids in the presence of an external potential
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-06-09
# Updated: 2022-06-09
# Version: 1.0
#
import numpy as np
from scipy.ndimage import convolve1d
from lj_eos import LJEOS
from yk_eos import *

" The DFT model for Lennard-Jones fluids"

class LJPlanar():
    def __init__(self,N,delta,sigma=1.0,epsilon=1.0,rhob=0.1,kT=1.0,method='MBWR'):
        self.N = N
        self.delta = delta
        self.L = delta*N

        self.sigma = sigma
        self.epsilon = epsilon
        self.method = method
        self.rhob = rhob 
        self.kT = kT
        self.beta = 1/self.kT

        # Baker-Henderson effective diameter
        kTstar = kT/self.epsilon
        self.d = sigma*(1+0.2977*kTstar)/(1+0.33163*kTstar+1.0477e-3*kTstar**2)

        # Two Yukawa parameters of LJ direct correlation function
        l = np.array([2.64279,14.9677])*self.d/self.sigma
        eps = 1.94728*self.epsilon*(self.sigma/self.d)*np.array([1,-1])*np.exp(l*(self.sigma/self.d-1))
        # l = np.array([2.9637,14.0167])*self.d/self.sigma
        # eps = 2.1714*self.epsilon*(self.sigma/self.d)*np.array([1,-1])*np.exp(l*(self.sigma/self.d-1))

        if self.method == 'FMSA':
            ljeos = YKEOS(sigma=self.d,epsilon=eps,l=l)
            self.f = ljeos.f(rhob,kT)
            self.mu = ljeos.mu(rhob,kT)

        elif self.method == 'MBWR':
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon,method='MBWR')
            self.f = ljeos.fatt(rhob,kT)
            self.mu = ljeos.muatt(rhob,kT)
        
        self.rc = 5.0*self.sigma # cutoff radius
        nphi = int(2*self.rc/self.delta)
        self.c2 = np.zeros(nphi,dtype=np.float32)

        eta = np.pi*self.rhob*self.d**3/6 # the effective packing fraction
        denom = ((1-eta)**4*l**6*Qfunc(l,eta)**2)
        A0 = -24*eta*Sfunc(l,eta)*Lfunc(l,eta)/denom
        A1 = 24*eta*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        A2 = -12*eta*(Sfunc(l,eta)*Lfunc(l,eta)*l**2-(1-eta)**2*(1+0.5*eta)*l**6)/denom
        A4 = 12*eta**2*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        C1 = -Sfunc(l,eta)**2/denom
        C2 = -144*eta**2*Lfunc(l,eta)**2/denom

        rc = self.rc/self.sigma
        x = np.linspace(-rc,rc,nphi)

        for i in range(l.size):
            self.c2[:] += 2*np.pi*self.beta*eps[i]*((np.exp(-l[i]*(np.abs(x)-1))-np.exp(-l[i]*(rc-1)))/l[i] + np.exp(-l[i]*(self.rc/self.sigma-1))*(np.abs(x)**2-rc**2)/rc) + self.beta*eps[i]*np.piecewise(x,[(np.abs(x)<=1),(np.abs(x)>1)],[lambda x: C1[i]*2*np.pi*(np.exp(-l[i]*(np.abs(x)-1))-1)/l[i]-C2[i]*2*np.pi*(np.exp(l[i]*(np.abs(x)-1))-1)/l[i]-A4[i]*0.4*np.pi*(np.abs(x)**5-1)-A2[i]*(2/3.0)*np.pi*(np.abs(x)**3-1) - A1[i]*np.pi*(np.abs(x)**2-1)-A0[i]*2*np.pi*(np.abs(x)-1),0.0] )
        
        del x, A0, A1, A2, A4, C1, C2, eta, denom, nphi

    def F(self,rho):
        Phi = self.f*np.ones(self.N,dtype=np.float32)
        Phi[:] += self.mu*(rho-self.rhob)
        Phi[:] += -self.kT*0.5*(rho-self.rhob)*convolve1d((rho-self.rhob), weights=self.c2, mode='nearest')*self.delta
        return np.sum(Phi)*self.delta

    def c1(self,rho):
        cc = -self.beta*self.mu*np.ones(self.N,dtype=np.float32)
        cc[:] += convolve1d((rho-self.rhob), weights=self.c2, mode='nearest')*self.delta
        return cc

class LJSpherical():
    def __init__(self,N,delta,sigma=1.0,epsilon=np.array([1.0]),rhob=0.1,kT=1.0,method='MBWR'):
        self.N = N
        self.delta = delta
        self.L = delta*N

        # the spherical coordinates
        self.r = np.arange(0,self.L,self.delta)+ 0.5*self.delta

        self.sigma = sigma
        self.epsilon = epsilon
        self.method = method
        self.rhob = rhob 
        self.kT = kT
        self.beta = 1/self.kT

        # Baker-Henderson effective diameter
        kTstar = kT/self.epsilon
        self.d = sigma*(1+0.2977*kTstar)/(1+0.33163*kTstar+1.0477e-3*kTstar**2)

        # Two Yukawa parameters of LJ direct correlation function
        l = np.array([2.64279,14.9677])*self.d/self.sigma
        eps = 1.94728*self.epsilon*(self.sigma/self.d)*np.array([1,-1])*np.exp(l*(self.sigma/self.d-1))
        # l = np.array([2.9637,14.0167])*self.d/self.sigma
        # eps = 2.1714*self.epsilon*(self.sigma/self.d)*np.array([1,-1])*np.exp(l*(self.sigma/self.d-1))

        if self.method == 'FMSA':
            ljeos = YKEOS(sigma=self.d,epsilon=eps,l=l)
            self.f = ljeos.f(rhob,kT)
            self.mu = ljeos.mu(rhob,kT)

        elif self.method == 'MBWR':
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon,method='MBWR')
            self.f = ljeos.fatt(rhob,kT)
            self.mu = ljeos.muatt(rhob,kT)
        
        self.rc = 5.0*self.sigma # cutoff radius
        nphi = int(2*self.rc/self.delta)
        self.c2 = np.zeros(nphi,dtype=np.float32)

        eta = np.pi*self.rhob*self.d**3/6 # the effective packing fraction
        denom = ((1-eta)**4*l**6*Qfunc(l,eta)**2)
        A0 = -24*eta*Sfunc(l,eta)*Lfunc(l,eta)/denom
        A1 = 24*eta*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        A2 = -12*eta*(Sfunc(l,eta)*Lfunc(l,eta)*l**2-(1-eta)**2*(1+0.5*eta)*l**6)/denom
        A4 = 12*eta**2*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        C1 = -Sfunc(l,eta)**2/denom
        C2 = -144*eta**2*Lfunc(l,eta)**2/denom

        r = np.linspace(-self.rc/self.sigma,self.rc/self.sigma,nphi)

        for i in range(eps.size):
            self.c2[:] += 2*np.pi*self.beta*eps[i]*np.exp(-l[i]*(np.abs(r)-1))/l[i] + self.beta*eps[i]*np.piecewise(r,[(np.abs(r)<=1),(np.abs(r)>1)],[lambda r: C1[i]*2*np.pi*(np.exp(-l[i]*(np.abs(r)-1))-1)/l[i]-C2[i]*2*np.pi*(np.exp(l[i]*(np.abs(r)-1))-1)/l[i]-A4[i]*0.4*np.pi*(np.abs(r)**5-1)-A2[i]*(2/3.0)*np.pi*(np.abs(r)**3-1) - A1[i]*np.pi*(np.abs(r)**2-1)-A0[i]*2*np.pi*(np.abs(r)-1),0.0] )
        
        del r, A0, A1, A2, A4, C1, C2, eta, denom, nphi

    def F(self,rho):
        Phi = self.f*np.ones_like(rho)
        Phi[:] += self.mu*(rho-self.rhob)
        Phi[:] += -self.kT*0.5*(rho-self.rhob)*convolve1d((rho-self.rhob)*self.r, weights=self.c2, mode='nearest')*self.delta/self.r
        return np.sum(Phi*4*np.pi*self.r**2)*self.delta

    def c1(self,rho):
        cc = -self.beta*self.mu*np.ones_like(rho)
        cc[:] += convolve1d((rho-self.rhob)*self.r, weights=self.c2, mode='nearest')*self.delta/self.r
        return cc

def yukpotential(r,params,sigma=1.0):
    [eps,l,rc] = params
    pot = np.zeros_like(r)
    for i in range(eps.size):
        pot += np.where(r<sigma,0.0,-eps[i]*sigma*np.exp(-l[i]*(r/sigma-1))/r+eps[i]*sigma*np.exp(-l[i]*(rc/sigma-1))/rc)
    return pot

def ljpotential(r,eps,sigma=1.0):
    return 4*eps*((sigma/r)**12-(sigma/r)**6)

################################################################################
if __name__ == "__main__":
    test1 = False # lennard-jones fluid with hardwall
    test2 = False # lj fluid inside a pore
    test3 = True # lj radialdistribution function

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    from fmt1d import FMTplanar, FMTspherical

    if test1: 
        
        epsilon = 1.0 
        sigma = 1.0
        delta = 0.01*sigma
        L = 6.0
        N = int(L/delta)

        rhob = 0.82
        kT = 1.35
        beta = 1.0/kT

        x = np.linspace(0,L,N)

        n = rhob*np.ones(N,dtype=np.float32)
        nsig = int(0.5*sigma/delta)
        n[:nsig] = 1.0e-16

        # Now we will solve the DFT with electrostatic correlations
        nn = n.copy()

        lj = LJPlanar(N,delta,sigma=sigma,epsilon=epsilon,rhob=rhob,kT=kT)
        fmt = FMTplanar(N,delta,species=1,d=np.array([lj.d]))

        mu = np.log(rhob) + fmt.mu(rhob) + beta*lj.mu

        # Now we will solve the DFT equations
        def Omega(var,mu):
            nn[:] = np.exp(var)
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = fmt.F(nn)
            Flj = beta*lj.F(nn)
            return (Fid+Fhs+Flj-np.sum(mu*nn*delta))/L

        def dOmegadnR(var,mu):
            nn[:] = np.exp(var)
            return nn*(var -fmt.c1(nn) -lj.c1(nn) - mu)*delta/L

        var = np.log(n)
        [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,mu,rtol=1.0e-6,dt=10.0,logoutput=True)
        n[:] = np.exp(varsol)

        MCdata = np.loadtxt('MCdata/lj-hardwall-rhob0.82-T1.35.dat')
        xMC,rhoMC = MCdata[:,0], MCdata[:,1]

        plt.scatter(xMC+0.5,rhoMC)
        plt.plot(x/sigma,n*sigma**3)
        plt.xlim(0.5,5.0)
        plt.ylim(0,3)
        plt.show()

        np.save('results/profiles-lennardjones-hardwall-rhob='+str(rhob)+'-T='+str(kT)+'.npy',[x,n])

    if test2: 
        
        epsilon = 1.0 
        sigma = 1.0
        delta = 0.01*sigma
        L = 1.8
        N = int(L/delta)

        rhob = 0.5925
        kT = 1.2
        beta = 1.0/kT

        sigmaw, epsw, Delta = sigma, 6.283*epsilon, 0.7071*sigma

        x = np.arange(0,L,delta)+0.5*delta

        def Vsteele(z,sigmaw,epsw,Delta):
            return epsw*(0.4*(sigmaw/z)**10-(sigmaw/z)**4-sigmaw**4/(3*Delta*(z+0.61*Delta)**3))

        Vext = beta*Vsteele(x,sigmaw,epsw,Delta)+beta*Vsteele(L-x,sigmaw,epsw,Delta)

        n = rhob*np.ones(N,dtype=np.float32)
        nsig = int(0.5*sigma/delta)
        # n[-nsig:] = 1.0e-16
        # n[:nsig] = 1.0e-16
        n[Vext>6.0] = 1.0e-16
        # Vext[-nsig:] = 1.0e-16
        # Vext[:nsig] = 1.0e-16
        Vext[Vext>6.0] = 0.0

        # Now we will solve the DFT with electrostatic correlations
        nn = n.copy()

        lj = LJPlanar(N,delta,sigma=sigma,epsilon=epsilon,rhob=rhob,kT=kT)
        fmt = FMTplanar(N,delta,species=1,d=np.array([lj.d]))

        mu = np.log(rhob) + fmt.mu(rhob) + beta*lj.mu

        # Now we will solve the DFT equations
        def Omega(var,mu):
            nn[:] = np.exp(var)
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = fmt.F(nn)
            Flj = beta*lj.F(nn)
            return (Fid+Fhs+Flj-np.sum((mu-Vext)*nn*delta))/L

        def dOmegadnR(var,mu):
            nn[:] = np.exp(var)
            return nn*(var -fmt.c1(nn) -lj.c1(nn) - mu + Vext)*delta/L

        var = np.log(n)
        [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,mu,rtol=1.0e-5,dt=0.2,logoutput=True)
        n[:] = np.exp(varsol)

        plt.plot(x/sigma,n*sigma**3)
        plt.xlim(0,L)
        plt.ylim(0,4)
        plt.show()

        # np.save('results/profiles-lennardjones-slitpore-rhob='+str(rhob)+'-T='+str(kT)+'-H='+str(L)+'.npy',[x,n])
        np.save('results/profiles-lennardjones-slitpore-rhob='+str(rhob)+'-T='+str(kT)+'-H='+str(L)+'-Elvisparameters.npy',[x,n])

    if test3:       
        rhob = 0.84
        kT = 0.75
        beta = 1.0/kT

        epsilon = 1.0

        sigma = 1.0
        delta = 0.01*sigma
        N = 900
        L = N*delta

        lj = LJSpherical(N,delta,sigma=sigma,epsilon=epsilon,rhob=rhob,kT=kT)
        fmt = FMTspherical(N,delta,d=lj.d)

        nsig = int(0.5*lj.d/delta)
        r = np.arange(0.0,L,delta) + 0.5*delta

        n = rhob*np.ones(N,dtype=np.float32)
        n[:nsig] = 1.0e-16
        Vext = np.zeros(N,dtype=np.float32)
        Vext[:] = beta*ljpotential(r,epsilon)
        # n[Vext>6.0] = 1.0e-16
        # Vext[Vext>6.0] = 0.0

        Vol = 4*np.pi*L**3/3

        mu = np.log(rhob) + fmt.mu(rhob) + beta*lj.mu
            
        def Omega(lnn,mu):
            n[nsig:-2*nsig] = np.exp(lnn)
            Fhs = fmt.F(n)
            Fid = np.sum(4*np.pi*r**2*n*(np.log(n)-1.0))*delta 
            Fyk = beta*lj.F(n)
            muN = np.sum(4*np.pi*r**2*(mu-Vext)*n)*delta 
            return (Fid+Fhs+Fyk-muN)/Vol

        def dOmegadnR(lnn,mu):
            n[nsig:-2*nsig] = np.exp(lnn)
            dOmegadrho = n*4*np.pi*r**2*(np.log(n) -fmt.c1(n) -lj.c1(n)- mu + Vext)*delta
            return dOmegadrho[nsig:-2*nsig]/Vol

        lnn = np.log(n[nsig:-2*nsig])
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,rtol=1.0e-5,dt=4.0,logoutput=True)

        n[nsig:-2*nsig] = np.exp(nsol)
        nmean = np.sum(n*4*np.pi*r**2*delta)/Vol

        print('rhob=',rhob,'\n nmean = ',nmean,'\n Omega/N =',Omegasol)

        plt.plot(r/sigma,n/rhob)
        plt.plot(r/sigma,0.5*kT*Vext,'--',color='grey')
        plt.xlim(0,3)
        plt.ylim(-0.5,3.5)
        plt.show()

        np.save('results/radialdistribution-lennardjones-rhob='+str(rhob)+'-T='+str(kT)+'-Elvisparameters.npy',[r,n/rhob])
        # np.save('results/radialdistribution-lennardjones-rhob='+str(rhob)+'-T='+str(kT)+'.npy',[r,n/rhob])