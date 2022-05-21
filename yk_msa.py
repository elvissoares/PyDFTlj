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
from pyfftw.interfaces.scipy_fftpack import fftn, ifftn
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

class Yukawa():
    def __init__(self,N,delta,sigma=1.0,epsilon=np.array([1.0]),l=np.array([1.0]),rc=5.0,rhob=0.1,kT=1.0,method='FMSA'):
        if N.size == 3: 
            self.N = N
            self.delta = delta
        else: 
            self.N = np.array([N,N,N])
            self.delta = np.array([delta,delta,delta])
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

        kx = np.fft.fftfreq(self.N[0], d=self.delta[0])*twopi
        ky = np.fft.fftfreq(self.N[1], d=self.delta[1])*twopi
        kz = np.fft.fftfreq(self.N[2], d=self.delta[2])*twopi
        kcut = np.pi/self.delta
        Kx,Ky,Kz = np.meshgrid(kx,ky,kz,indexing ='ij')
        K = np.sqrt(Kx**2 + Ky**2 + Kz**2)
        del kx,ky,kz
        
        self.rc = rc*self.sigma # cutoff radius

        self.rhodiff_hat = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)
        self.c2_hat = np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)

        eta = np.pi*self.rhob*self.sigma**3/6
        denom = ((1-eta)**4*l**6*Qfunc(l,eta)**2)
        A0 = -24*eta*Sfunc(l,eta)*Lfunc(l,eta)/denom
        A1 = 24*eta*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        A2 = -12*eta*(Sfunc(l,eta)*Lfunc(l,eta)*l**2-(1-eta)**2*(1+0.5*eta)*l**6)/denom
        A4 = 12*eta**2*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
        C1 = -Sfunc(l,eta)**2/denom
        C2 = -144*eta**2*Lfunc(l,eta)**2/denom

        for i in range(self.epsilon.size):
            self.c2_hat[:] += -self.beta*self.epsilon[i]*YKcutoffFT(K,self.l[i],rc=self.rc,sigma=self.sigma) 
            self.c2_hat[:] += self.beta*self.epsilon[i]*(A0[i]*A0funcFT(K,sigma=self.sigma)+A1[i]*A1funcFT(K,sigma=self.sigma)+A2[i]*A2funcFT(K,sigma=self.sigma)+A4[i]*A4funcFT(K,sigma=self.sigma)+C1[i]*YKcoreFT(K,self.l[i],sigma=self.sigma)+C2[i]*YKcoreFT(K,-self.l[i],sigma=self.sigma))

        self.c2_hat[:] *= sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L) # to avoid Gibbs phenomenum 
        
        del Kx,Ky,Kz,K, A0, A1, A2, A4, C1, C2, eta, denom

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

def yuk(r,params,sigma=1.0):
    [eps,l,rc] = params
    pot = np.zeros_like(r)
    for i in range(eps.size):
        pot += np.where(r<sigma,0.0,-eps[i]*sigma*np.exp(-l[i]*(r/sigma-1))/r+eps[i]*sigma*np.exp(-l[i]*(rc/sigma-1))/rc)
    return pot

def lj(r,params,sigma=1.0):
    [eps,rc] = params
    return np.piecewise(r,[r<rc, r>=rc],[lambda r: 4*eps*((sigma/r)**(12)-(sigma/r)**(6))-4*eps*((sigma/rc)**(12)-(sigma/rc)**(6)),0.0])

################################################################################
if __name__ == "__main__":
    test1 = False # attractive yukawa radial distribution function
    test2 = True # lennard-jones radial distribution function

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    from fmt import FMT

    if test1:
        sigma = 1.0

        # parameters of the gridsize
        # N, delta = 32, 0.4*sigma # 32³ grid
        # N, delta = 64, 0.2*sigma # 64³ grid
        N, delta = 128, 0.1*sigma # 128³ grid
        # N, delta = 256, 0.05*sigma # 256³ grid
        L = N*delta
        z = np.arange(0,L,delta)
        Narray = np.array([N,N,N])
        deltaarray = np.array([delta,delta,delta])

        # parameters of Yukawa potential
        eps = np.array([1.0])
        l = np.array([1.8])
        rc = 5.0

        # parameters of the fluid
        rhob = 0.3
        kT = 2.0
        beta = 1.0/kT

        n0 = 1.0e-16*np.ones((N,N,N),dtype=np.float32)
        Vext = np.zeros((N,N,N),dtype=np.float32)

        X,Y,Z = np.meshgrid(z,z,z,indexing ='ij')
        R2 = (X-L/2)**2 + (Y-L/2)**2 + (Z-L/2)**2
        mask = R2>=sigma**2
        n0[mask] = rhob
        Vext[mask] = beta*yuk(np.sqrt(R2[mask]),[eps,l,rc],sigma=sigma)
        del X,Y,Z,R2,mask

        # for i in range(N):
        #     for j in range(N):
        #         for k in range(N):
        #             r2 = delta**2*((i-N/2)**2+(j-N/2)**2+(k-N/2)**2)
        #             Vext[i,j,k] = beta*yuk(np.sqrt(r2),[eps,l,rc],sigma=sigma)
        #             if r2>=sigma**2: n0[i,j,k] = rhob

        print('Vext(r=sigma) = ',beta*yuk(1.0,[eps,l,rc],sigma=sigma))
        print('Vext min = ',np.min(Vext))


        # plt.plot(z,Vext[:,N//2,N//2])
        # plt.xlim(0.0,L/2)
        # # plt.ylim(0,5)
        # plt.show()

        # plt.plot(z,n0[:,N//2,N//2]/rhob)
        # plt.xlim(0.0,L/2)
        # # plt.ylim(0,5)
        # plt.show()
        ################################################################################

        fmt = FMT(Narray,deltaarray)
        yk = Yukawa(Narray,deltaarray,sigma=sigma,epsilon=eps,l=l,rc=rc,rhob=rhob,kT=kT)

        # c2yk = ifftn(yk.c2_hat).real

        # plt.plot(z,c2yk[:,N//2,N//2])
        # # plt.xlim(0.0,L/2)
        # # plt.ylim(0,5)
        # plt.show()

        n = n0.copy() # auxiliary variable

        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            fhs = fmt.Phi(n)
            fyk = beta*yk.free_energy_density(n)
            Omegak = n*(lnn-1.0) + fhs + fyk + (Vext- mu)*n
            return Omegak.sum()*delta**3/L**3

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            c1hs = fmt.c1(n)
            c1yk = yk.c1(n)
            return n*(lnn -c1hs - c1yk - mu + Vext)*delta**3/L**3

        mu = np.log(rhob) + fmt.mu(rhob) + beta*yk.mu
        
        lnn = np.log(n)
        
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,alpha0=0.62,rtol=1e-4,dt=40.0,logoutput=False)

        print('Niter = ',Niter)

        n[:] = np.exp(nsol)

        # plt.plot(z,n[:,N//2,N//2]/rhob)
        # # plt.xlim(1.0,L/2)
        # plt.ylim(0,5)
        # plt.show()

        np.save('results/densityfield-yukawa-fmsa-rhob'+str(rhob)+'-N'+str(N)+'-delta'+str(delta)+'.npy',n)

        plt.imshow(n[:,:,N//2]/rhob, cmap='Greys_r')
        plt.colorbar(label=r'$\rho(x,y,0)/\rho_b$')
        plt.xlabel('$x/\\sigma$')
        plt.ylabel('$y/\\sigma$')
        plt.savefig('figures/densitymap-yukawa-fmsa-rhob'+str(rhob)+'-N'+str(N)+'-delta'+str(delta)+'.pdf', bbox_inches='tight')
        # plt.show()
        plt.close()
    
    if test2:
        
        # parameters of the fluid
        rhob = 0.84
        kT = 0.71
        beta = 1.0/kT

        sigma = 1.0
        d = sigma*(1+0.2977*kT)/(1+0.33163*kT+0.0010477*kT**2)
        print('d/sigma = ',d)

        # parameters of the gridsize
        # N, delta = 32, 0.4*d # 32³ grid
        # N, delta = 64, 0.2*d # 64³ grid
        # N, delta = 128, 0.1*d # 128³ grid
        N, delta = 256, 0.05*d # 256³ grid

        L = N*delta
        z = np.arange(0,L,delta)
        Narray = np.array([N,N,N])
        deltaarray = np.array([delta,delta,delta])

        # parameters of Lennard-Jones potential described by two Yukawa
        l = np.array([2.9637,14.0167])/sigma
        eps = 2.1714*sigma*np.array([1,-1])*np.exp(l*(sigma-d))
        rc = 5.0

        n0 = rhob*np.ones((N,N,N),dtype=np.float32)
        Vext = np.zeros((N,N,N),dtype=np.float32)

        X,Y,Z = np.meshgrid(z,z,z,indexing ='ij')
        R2 = (X-L/2)**2 + (Y-L/2)**2 + (Z-L/2)**2
        mask = R2<(0.75*d)**2
        n0[mask] = 1.e-16
        mask = R2>(0.75*d)**2
        Vext[mask] = beta*lj(np.sqrt(R2[mask]),[1.0,rc],sigma=sigma)
        mask = Vext>10
        Vext[mask] = 0.0
        n0[mask] = 1.e-16
        del X,Y,Z,R2,mask

        # plt.plot(z,Vext[:,N//2,N//2])
        # # plt.xlim(0.0,L/2)
        # # plt.ylim(0,5)
        # plt.show()

        # plt.plot(z,n0[:,N//2,N//2]/rhob)
        # # plt.xlim(0.0,L/2)
        # # plt.ylim(0,5)
        # plt.show()

        print('Vext min = ',np.min(Vext))

        fmt = FMT(Narray,deltaarray,sigma=d)
        yk = Yukawa(Narray,deltaarray,sigma=d,epsilon=eps,l=l,rc=rc,rhob=rhob/d**3,kT=kT)

        n = n0.copy() # auxiliary variable

        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            fhs = fmt.Phi(n)
            fyk = beta*yk.free_energy_density(n)
            Omegak = n*(lnn-1.0) + fhs + fyk + (Vext- mu)*n
            return Omegak.sum()*delta**3/L**3

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            c1hs = fmt.c1(n)
            c1yk = yk.c1(n)
            return n*(lnn -c1hs - c1yk - mu + Vext)*delta**3/L**3

        mu = np.log(rhob) + fmt.mu(rhob/d**3) + beta*yk.mu
        
        lnn = np.log(n)
        
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,alpha0=0.62,rtol=1e-4,dt=40.0,logoutput=True)

        print('Niter = ',Niter)

        n[:] = np.exp(nsol)

        plt.plot(z-L/2,n[:,N//2,N//2]/rhob)
        plt.xlim(0.5,5)
        plt.ylim(-0.2,3.2)
        plt.show()

        np.save('results/densityfield-lj-fmsa-rhostar='+str(rhob)+'-Tstar='+str(kT)+'-N'+str(N)+'-delta'+str(delta/d)+'.npy',n)

        plt.imshow(n[:,:,N//2]/rhob, cmap='Greys_r')
        plt.colorbar(label=r'$\rho(x,y,0)/\rho_b$')
        plt.xlabel('$x/\\sigma$')
        plt.ylabel('$y/\\sigma$')
        plt.savefig('figures/densitymap-lj-fmsa-rhostar='+str(rhob)+'-Tstar='+str(kT)+'-N'+str(N)+'-delta'+str(delta/d)+'.pdf', bbox_inches='tight')
        # plt.show()
        plt.close()

        