#!/usr/bin/env python3

# This script is the python implementation of the Density Functional Theory
# for the multiple Yukawa hard-core fluids in the presence of an external potential
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-05-13
# Updated: 2022-05-16
# Version: 1.0
#
import numpy as np
from scipy.ndimage import convolve1d
from yk_eos import *

" The DFT model for multiple yukawa hard-core fluids"

class YukawaPlanar():
    def __init__(self,N,delta,sigma=1.0,epsilon=np.array([1.0]),l=np.array([1.0]),rhob=0.1,kT=1.0,method='FMSA'):
        self.N = N
        self.delta = delta
        self.L = delta*N

        self.sigma = sigma
        self.epsilon = epsilon
        self.l = l*sigma
        self.method = method
        self.rhob = rhob 
        self.kT = kT
        self.beta = 1/self.kT

        ykeos = YKEOS(sigma=self.sigma,epsilon=self.epsilon,l=self.l,method=self.method)

        self.f = ykeos.f(rhob,kT)
        self.mu = ykeos.mu(rhob,kT)
        
        self.rc = 5.0*self.sigma # cutoff radius
        nphi = int(2*self.rc/self.delta)
        self.c2 = np.zeros(nphi,dtype=np.float32)

        eta = np.pi*self.rhob*self.sigma**3/6
        denom = ((1-eta)**4*l**6*Qfunc(l,eta)**2)
        A0 = -24*eta*Sfunc(l,eta)*Lfunc(l,eta)/denom
        A1 = 24*eta*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        A2 = -12*eta*(Sfunc(l,eta)*Lfunc(l,eta)*l**2-(1-eta)**2*(1+0.5*eta)*l**6)/denom
        A4 = 12*eta**2*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        C1 = -Sfunc(l,eta)**2/denom
        C2 = -144*eta**2*Lfunc(l,eta)**2/denom

        rc = self.rc/self.sigma
        x = np.linspace(-self.rc/self.sigma,self.rc/self.sigma,nphi)

        for i in range(self.epsilon.size):
            self.c2[:] += 2*np.pi*self.beta*self.epsilon[i]*((np.exp(-self.l[i]*(np.abs(x)-1))-np.exp(-self.l[i]*(rc-1)))/self.l[i] + np.exp(-self.l[i]*(self.rc/self.sigma-1))*(np.abs(x)**2-rc**2)/rc) + self.beta*self.epsilon[i]*np.piecewise(x,[(np.abs(x)<=1),(np.abs(x)>1)],[lambda x: C1[i]*2*np.pi*(np.exp(-self.l[i]*(np.abs(x)-1))-1)/self.l[i]-C2[i]*2*np.pi*(np.exp(self.l[i]*(np.abs(x)-1))-1)/self.l[i]-A4[i]*0.4*np.pi*(np.abs(x)**5-1)-A2[i]*(2/3.0)*np.pi*(np.abs(x)**3-1) - A1[i]*np.pi*(np.abs(x)**2-1)-A0[i]*2*np.pi*(np.abs(x)-1),0.0] )
        
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

class YukawaSpherical():
    def __init__(self,N,delta,sigma=1.0,epsilon=np.array([1.0]),l=np.array([1.0]),rhob=0.1,kT=1.0,method='FMSA'):
        self.N = N
        self.delta = delta
        self.L = delta*N

        # the spherical coordinates
        self.r = np.linspace(0,self.L,self.N)
        self.rmed = self.r + 0.1*self.delta

        self.sigma = sigma
        self.epsilon = epsilon
        self.l = l*sigma
        self.method = method
        self.rhob = rhob 
        self.kT = kT
        self.beta = 1/self.kT

        ykeos = YKEOS(sigma=self.sigma,epsilon=self.epsilon,l=self.l,method=self.method)

        self.f = ykeos.f(rhob,kT)
        self.mu = ykeos.mu(rhob,kT)
        
        self.rc = 5.0*self.sigma # cutoff radius
        nphi = int(2*self.rc/self.delta)
        self.c2 = np.zeros(nphi,dtype=np.float32)

        eta = np.pi*self.rhob*self.sigma**3/6
        denom = ((1-eta)**4*l**6*Qfunc(l,eta)**2)
        A0 = -24*eta*Sfunc(l,eta)*Lfunc(l,eta)/denom
        A1 = 24*eta*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        A2 = -12*eta*(Sfunc(l,eta)*Lfunc(l,eta)*l**2-(1-eta)**2*(1+0.5*eta)*l**6)/denom
        A4 = 12*eta**2*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        C1 = -Sfunc(l,eta)**2/denom
        C2 = -144*eta**2*Lfunc(l,eta)**2/denom

        r = np.linspace(-self.rc/self.sigma,self.rc/self.sigma,nphi)

        for i in range(self.epsilon.size):
            self.c2[:] += 2*np.pi*self.beta*self.epsilon[i]*np.exp(-self.l[i]*(np.abs(r)-1))/self.l[i] + self.beta*self.epsilon[i]*np.piecewise(r,[(np.abs(r)<=1),(np.abs(r)>1)],[lambda r: C1[i]*2*np.pi*(np.exp(-self.l[i]*(np.abs(r)-1))-1)/self.l[i]-C2[i]*2*np.pi*(np.exp(self.l[i]*(np.abs(r)-1))-1)/self.l[i]-A4[i]*0.4*np.pi*(np.abs(r)**5-1)-A2[i]*(2/3.0)*np.pi*(np.abs(r)**3-1) - A1[i]*np.pi*(np.abs(r)**2-1)-A0[i]*2*np.pi*(np.abs(r)-1),0.0] )
        
        del r, A0, A1, A2, A4, C1, C2, eta, denom, nphi

    def F(self,rho):
        Phi = self.f*np.ones_like(rho)
        Phi[:] += self.mu*(rho-self.rhob)
        Phi[:] += -self.kT*0.5*(rho-self.rhob)*convolve1d((rho-self.rhob)*self.r, weights=self.c2, mode='nearest')*self.delta/self.rmed
        return np.sum(Phi*4*np.pi*self.r**2)*self.delta

    def c1(self,rho):
        cc = -self.beta*self.mu*np.ones_like(rho)
        cc[:] += convolve1d((rho-self.rhob)*self.r, weights=self.c2, mode='nearest')*self.delta/self.rmed
        return cc

def yuk(r,params,sigma=1.0):
    [eps,l,rc] = params
    pot = np.zeros_like(r)
    for i in range(eps.size):
        pot += np.where(r<sigma,0.0,-eps[i]*sigma*np.exp(-l[i]*(r/sigma-1))/r+eps[i]*sigma*np.exp(-l[i]*(rc/sigma-1))/rc)
    return pot

################################################################################
if __name__ == "__main__":
    test1 = False # attractive yukawa hardwall
    test2 = True # attractive yukawa radial distribution

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    from fmt1d import FMTplanar, FMTspherical

    if test1: 
        
        sigma = 1.0
        delta = 0.01*sigma
        L = 5.0
        N = int(L/delta)

        rhob = 0.7
        eps = np.array([1.0])
        l = np.array([1.8])

        kT = 2.0
        beta = 1.0/kT

        x = np.linspace(0,L,N)

        n = rhob*np.ones(N,dtype=np.float32)
        nsig = int(0.5*sigma/delta)
        n[:nsig] = 1.0e-16

        # Now we will solve the DFT with electrostatic correlations
        nn = n.copy()

        fmt = FMTplanar(N,delta,species=1,d=np.array([sigma]))
        yk = YukawaPlanar(N,delta,sigma=sigma,epsilon=eps,l=l,rhob=rhob,kT=kT)

        mu = np.log(rhob) + fmt.mu(rhob) + beta*yk.mu

        # Now we will solve the DFT equations
        def Omega(var,mu):
            nn[:] = np.exp(var)
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = fmt.F(nn)
            Fyk = beta*yk.F(nn)
            return (Fid+Fhs+Fyk-np.sum(mu*nn*delta))/L

        def dOmegadnR(var,mu):
            nn[:] = np.exp(var)
            return nn*(var -fmt.c1(nn) -yk.c1(nn) - mu)*delta/L

        var = np.log(n)
        [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,mu,rtol=1.0e-6,dt=10.0,logoutput=True)
        n[:] = np.exp(varsol)

        plt.plot(x/sigma,n*sigma**3)
        plt.xlim(0.5,3.5)
        plt.ylim(0,2.5)
        plt.show()

        np.save('results/profiles-yukawa-hardwall-rhob='+str(rhob)+'-T='+str(kT)+'-epsw=0.0-lambda'+str(l[0])+'.npy',[x,n])

    if test2:       
        rhob = 0.3
        kT = 2.0
        beta = 1.0/kT

        l = np.array([1.8])
        eps = np.array([1.0])

        sigma = 1.0
        delta = 0.02*sigma
        N = 300
        L = N*delta

        fmt = FMTspherical(N,delta,d=sigma)
        yk = YukawaSpherical(N,delta,sigma=sigma,epsilon=eps,l=l,rhob=rhob,kT=kT)

        mu = np.log(rhob) + fmt.mu(rhob) + beta*yk.mu

        nsig = int(sigma/delta)
        r = np.linspace(0.0,L,N)

        n = rhob*np.ones(N,dtype=np.float32)
        n[:nsig] = 1.0e-16
        Vext = np.zeros(N,dtype=np.float32)
        Vext[:] = beta*yuk(r,[eps,l,5.0])

        # plt.plot(r,Vext)
        # plt.show()

        Vol = 4*np.pi*L**3/3
            
        def Omega(lnn,param):
            [mu,kT,rhob] = param 
            n[nsig:] = np.exp(lnn)
            n[:nsig] = 1.0e-16
            n[-nsig:] = rhob
            Fhs = fmt.F(n)
            Fid = np.sum(4*np.pi*r**2*n*(np.log(n)-1.0))*delta 
            Fyk = beta*yk.F(n)
            muN = np.sum(4*np.pi*r**2*(mu-Vext)*n)*delta 
            return (Fid+Fhs+Fyk-muN)/Vol

        def dOmegadnR(lnn,param):
            [mu,kT,rhob] = param 
            n[nsig:] = np.exp(lnn)
            n[:nsig] = 1.0e-16
            n[-nsig:] = rhob
            dOmegadrho = n*4*np.pi*r**2*(np.log(n) -fmt.c1(n) -yk.c1(n)- mu + Vext)*delta
            dOmegadrho[-nsig:] = 0.0
            return dOmegadrho[nsig:]/Vol

        lnn = np.log(n[nsig:])
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,[mu,kT,rhob],rtol=1.0e-5,dt=2.0,logoutput=True)

        n[nsig:]  = np.exp(nsol)
        nmean = np.sum(n*4*np.pi*r**2*delta)/Vol

        print('rhob=',rhob,'\n nmean = ',nmean,'\n Omega/N =',Omegasol)

        plt.plot(r/sigma,n/rhob)
        # plt.xlim(1,3)
        plt.ylim(0,5)
        plt.show()

        np.save('results/radialdistribution-yukawa-rhob'+str(rhob)+'-T'+str(kT)+'-lambda'+str(l[0])+'.npy',[r,n/rhob])