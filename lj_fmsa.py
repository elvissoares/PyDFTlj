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
# from scipy.fftpack import fftn, ifftn
import pyfftw
from pyfftw.interfaces.scipy_fftpack import fftn, ifftn, ifft
from lj_eos import LJEOS

from yk_eos import *

pyfftw.config.NUM_THREADS = 4
pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(30)

" The DFT model for multiple yukawa hard-core fluids"

twopi = 2*np.pi

def sigmaLancsozFT(kx,ky,kz,kcut):
    return np.sinc(kx/kcut[0])*np.sinc(ky/kcut[1])*np.sinc(kz/kcut[2])

def translationFT(kx,ky,kz,a):
    return np.exp(1.0j*(kx*a[0]+ky*a[1]+kz*a[2]))

def A0funcFT(k,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[2*np.pi*sigma**3,lambda k: 4*np.pi*sigma*(1-np.cos(k*sigma))/k**2])

def A1funcFT(k,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[4*np.pi*sigma**3/3,lambda k: 4*np.pi*(np.sin(k*sigma)-(k*sigma)*np.cos(k*sigma))/k**3])

def A2funcFT(k,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[np.pi*sigma**3,lambda k: (4*np.pi/sigma)*(2*k*sigma*np.sin(k*sigma)+(2-k**2*sigma**2)*np.cos(k*sigma)-2)/k**4])

def A4funcFT(k,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[2*np.pi*sigma**3/3,lambda k: (4*np.pi/sigma**3)*(4*k*sigma*(k**2*sigma**2-6)*np.sin(k*sigma)-(24-12*k**2*sigma**2+k**4*sigma**4)*np.cos(k*sigma)+24)/k**6])

def YKcoreFT(k,l,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[-4*np.pi*sigma**3*(1+l-np.exp(l))/l**2,lambda k: -4*np.pi*sigma**2*(l*np.sin(k*sigma)+k*sigma*np.cos(k*sigma)-k*sigma*np.exp(l))/(k*(l**2+k**2*sigma**2))])

def YKcutoffFT(k,l,rc=5.0,sigma=1.0):
    return 4*np.pi*sigma*np.exp(l)*np.piecewise(k,[k<=1e-6,k>1e-6],[(np.exp(-l*rc/sigma)*(rc**3*l**2/3+rc**2*l*sigma+rc*sigma**2)-rc*sigma**2)/(rc*l**2),lambda k: (np.exp(-l*rc/sigma)*((l**2+k**2*rc*l*sigma+k**2*sigma**2)*np.sin(k*rc)-k*rc*l**2*np.cos(k*rc))-k**3*rc*sigma**2)/(k**3*rc*(l**2+k**2*sigma**2))])

class LennardJones():
    def __init__(self,N,delta,sigma=1.0,epsilon=1.0,rhob=0.1,kT=1.0,method='MBWR'):
        if N.size == 3: 
            self.N = N
            self.delta = delta
        else: 
            self.N = np.array([N,N,N])
            self.delta = np.array([delta,delta,delta])
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
            self.pexc = ljeos.p(rhob,kT)

        elif self.method == 'MBWR':
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon,method='MBWR')
            self.f = ljeos.fatt(rhob,kT)
            self.mu = ljeos.muatt(rhob,kT)
            self.pexc = ljeos.p(rhob,kT)
        
        self.rc = 5.0*self.sigma # cutoff radius

        kx = np.fft.fftfreq(self.N[0], d=self.delta[0])*twopi
        ky = np.fft.fftfreq(self.N[1], d=self.delta[1])*twopi
        kz = np.fft.fftfreq(self.N[2], d=self.delta[2])*twopi
        kcut = np.pi/self.delta
        Kx,Ky,Kz = np.meshgrid(kx,ky,kz,indexing ='ij')
        K = np.sqrt(Kx**2 + Ky**2 + Kz**2)
        del kx,ky,kz

        self.rhodiff_hat = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)
        self.c2_hat = np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)

        eta = np.pi*self.rhob*self.d**3/6
        denom = ((1-eta)**4*l**6*Qfunc(l,eta)**2)
        A0 = -24*eta*Sfunc(l,eta)*Lfunc(l,eta)/denom
        A1 = 24*eta*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        A2 = -12*eta*(Sfunc(l,eta)*Lfunc(l,eta)*l**2-(1-eta)**2*(1+0.5*eta)*l**6)/denom
        A4 = 0.5*eta*A1
        C1 = -Sfunc(l,eta)**2/denom
        C2 = -144*eta**2*Lfunc(l,eta)**2/denom

        for i in range(eps.size):
            self.c2_hat[:] += -self.beta*eps[i]*YKcutoffFT(K,l[i],rc=self.rc/self.sigma,sigma=self.d) 
            self.c2_hat[:] += self.beta*eps[i]*(A0[i]*A0funcFT(K,sigma=self.d)+A1[i]*A1funcFT(K,sigma=self.d)+A2[i]*A2funcFT(K,sigma=self.d)+A4[i]*A4funcFT(K,sigma=self.d)+C1[i]*YKcoreFT(K,l[i],sigma=self.d)+C2[i]*YKcoreFT(K,-l[i],sigma=self.d))

        self.c2_hat[:] *= sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L) # to avoid Gibbs phenomenum 
        
        del Kx,Ky,Kz,K,kcut, A0, A1, A2, A4, C1, C2, eta, denom

    def free_energy_density(self,rho):
        self.rhodiff_hat[:] = fftn(rho-self.rhob)
        Phi = self.f*np.ones_like(rho)
        Phi[:] += self.mu*(rho-self.rhob)
        Phi[:] += -self.kT*0.5*(rho-self.rhob)*ifftn(self.rhodiff_hat*self.c2_hat).real
        return Phi

    def c1(self,rho):
        self.rhodiff_hat[:] = fftn(rho-self.rhob)
        cc = -self.beta*self.mu*np.ones_like(rho)
        cc[:] += ifftn(self.rhodiff_hat*self.c2_hat).real
        return cc

def ljpotential(r,eps,sigma=1.0):
    return  4*eps*((sigma/r)**(12)-(sigma/r)**(6))

################################################################################
if __name__ == "__main__":
    test1 = True # lennard-jones radial distribution function

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    from fmt import FMT

    if test1:
        
        # parameters of the fluid
        epsilon = 1.0
        sigma = 1.0
        rhob = 0.84
        kT = 0.75
        beta = 1.0/kT

        d = sigma*(1+0.2977*kT)/(1+0.33163*kT+0.0010477*kT**2)

        # parameters of the gridsize
        # N, delta = 32, 0.4*d # 32³ grid
        # N, delta = 64, 0.2*d # 64³ grid
        N, delta = 128, 0.1*d # 128³ grid
        # N, delta = 256, 0.05*d # 256³ grid

        L = N*delta
        z = np.arange(0,L,delta)
        Narray = np.array([N,N,N])
        deltaarray = np.array([delta,delta,delta])

        lj = LennardJones(Narray,deltaarray,sigma=sigma,epsilon=epsilon,rhob=rhob,kT=kT)
        fmt = FMT(Narray,deltaarray,sigma=lj.d)

        n0 = rhob*np.ones((N,N,N),dtype=np.float32)
        Vext = np.zeros((N,N,N),dtype=np.float32)

        X,Y,Z = np.meshgrid(z,z,z,indexing ='ij')
        R2 = (X-L/2)**2 + (Y-L/2)**2 + (Z-L/2)**2
        mask = R2<(0.5*d)**2
        n0[mask] = 1.e-16
        mask = R2>(0.5*d)**2
        Vext[mask] = beta*ljpotential(np.sqrt(R2[mask]),epsilon,sigma=sigma)
        mask = Vext>10.0
        Vext[mask] = 10.0
        # n0[mask] = 1.e-16
        del X,Y,Z,R2,mask

        # plt.plot(z,Vext[:,N//2,N//2])
        # # plt.xlim(0.0,L/2)
        # # plt.ylim(0,5)
        # plt.show()

        # plt.plot(z,n0[:,N//2,N//2]/rhob)
        # # plt.xlim(0.0,L/2)
        # # plt.ylim(0,5)
        # plt.show()

        n = n0.copy() # auxiliary variable

        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            fhs = fmt.Phi(n)
            fyk = beta*lj.free_energy_density(n)
            Omegak = n*(lnn-1.0) + fhs + fyk + (Vext- mu)*n
            return Omegak.sum()*delta**3/L**3

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            c1hs = fmt.c1(n)
            c1yk = lj.c1(n)
            return n*(lnn -c1hs - c1yk - mu + Vext)*delta**3/L**3

        mu = np.log(rhob) + fmt.mu(rhob) + beta*lj.mu      
        lnn = np.log(n)
        
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,alpha0=0.62,rtol=1e-4,dt=40.0,logoutput=True)

        print('Niter = ',Niter)

        n[:] = np.exp(nsol)

        # plt.plot(z-L/2,n[:,N//2,N//2]/rhob)
        # plt.xlim(0.5,5)
        # plt.ylim(-0.2,3.2)
        # plt.show()

        np.save('results/densityfield-lj-fmsa-rhostar='+str(rhob)+'-Tstar='+str(kT)+'-N'+str(N)+'-delta'+'{0:.1f}'.format(delta/d)+'.npy',n)        