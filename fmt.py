import numpy as np
from scipy.special import spherical_jn
# from scipy.fftpack import fftn, ifftn
import pyfftw
from pyfftw.interfaces.scipy_fftpack import fftn, ifftn
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-16
# Updated: 2022-06-24
# pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.NUM_THREADS = 4
pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(30)

twopi = 2*np.pi

def sigmaLancsozFT(kx,ky,kz,kcut):
    return np.sinc(kx/kcut[0])*np.sinc(ky/kcut[1])*np.sinc(kz/kcut[2])

def translationFT(kx,ky,kz,a):
    return np.exp(1.0j*(kx*a[0]+ky*a[1]+kz*a[2]))

def w3FT(k,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[np.pi*sigma**3/6,lambda k: (np.pi*sigma**2/k)*spherical_jn(1,0.5*sigma*k)])

def w2FT(k,sigma=1.0):
    return np.pi*sigma**2*spherical_jn(0,0.5*sigma*k)

def wtensFT(k,sigma=1.0):
    return np.pi*sigma**2*spherical_jn(2,0.5*sigma*k)

def wtensoverk2FT(k,sigma=1.0):
    return np.piecewise(k,[k*sigma<=1e-3,k*sigma>1e-3],[np.pi*sigma**4/60,lambda k:(np.pi*sigma**2/k**2)*spherical_jn(2,0.5*sigma*k)])

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

# The available methods are
# RF: Rosenfeld Functional
# WBI: White Bear version I (default method)
# WBII: White Bear version II

class FMT():
    def __init__(self,N,delta,sigma=1.0,method='WBI'):
        self.method = method 
        if N.size == 3: 
            self.N = N
            self.delta = delta
        else: 
            self.N = np.array([N,N,N])
            self.delta = np.array([delta,delta,delta])
        self.L = self.N*self.delta
        self.sigma = sigma

        self.n_hat = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)

        self.w3_hat = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)
        self.w2_hat = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)
        self.w2vec_hat = np.empty((3,self.N[0],self.N[1],self.N[2]),dtype=np.complex64)

        self.n3 = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.float32)
        self.n2 = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.float32)
        self.n2vec = np.empty((3,self.N[0],self.N[1],self.N[2]),dtype=np.float32)
        self.n1vec = np.empty((3,self.N[0],self.N[1],self.N[2]),dtype=np.float32)
        
        kx = np.fft.fftfreq(self.N[0], d=self.delta[0])*twopi
        ky = np.fft.fftfreq(self.N[1], d=self.delta[1])*twopi
        kz = np.fft.fftfreq(self.N[2], d=self.delta[2])*twopi
        kcut = np.array([kx.max(),ky.max(),kz.max()])
        # kcut = np.pi/delta
        Kx,Ky,Kz = np.meshgrid(kx,ky,kz,indexing ='ij')
        K = np.sqrt(Kx**2 + Ky**2 + Kz**2)
        del kx,ky,kz

        self.w3_hat[:] = w3FT(K,sigma=self.sigma)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)

        self.w2_hat[:] = w2FT(K,sigma=self.sigma)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)

        self.w2vec_hat[0] = -1.0j*Kx*self.w3_hat
        self.w2vec_hat[1] = -1.0j*Ky*self.w3_hat
        self.w2vec_hat[2] = -1.0j*Kz*self.w3_hat

        if self.method == 'WB-tensor':

            self.w2tens_hat = np.empty((3,3,self.N[0],self.N[1],self.N[2]),dtype=np.complex64)

            self.n2tens = np.empty((3,3,self.N[0],self.N[1],self.N[2]),dtype=np.float32)

            w2tens_hat = wtensFT(K,sigma=self.sigma)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)
            w2tensoverk2_hat = wtensoverk2FT(K,sigma=self.sigma)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)

            self.w2tens_hat[0,0] = -Kx*Kx*w2tensoverk2_hat+(1/3.0)*w2tens_hat
            self.w2tens_hat[0,1] = -Kx*Ky*w2tensoverk2_hat
            self.w2tens_hat[0,2] = -Kx*Kz*w2tensoverk2_hat
            self.w2tens_hat[1,1] = -Ky*Ky*w2tensoverk2_hat+(1/3.0)*w2tens_hat
            self.w2tens_hat[1,0] = -Ky*Kx*w2tensoverk2_hat
            self.w2tens_hat[1,2] = -Ky*Kz*w2tensoverk2_hat
            self.w2tens_hat[2,0] = -Kz*Kx*w2tensoverk2_hat
            self.w2tens_hat[2,1] = -Kz*Ky*w2tensoverk2_hat
            self.w2tens_hat[2,2] = -Kz*Kz*w2tensoverk2_hat+(1/3.0)*w2tens_hat

            del w2tens_hat, w2tensoverk2_hat

        del Kx,Ky,Kz,K

    def weighted_densities(self,n):
        self.n_hat[:] = fftn(n)

        self.n3[:] = ifftn(self.n_hat*self.w3_hat).real
        self.n2[:] = ifftn(self.n_hat*self.w2_hat).real
        self.n2vec[0] = ifftn(self.n_hat*self.w2vec_hat[0]).real
        self.n2vec[1] = ifftn(self.n_hat*self.w2vec_hat[1]).real
        self.n2vec[2] = ifftn(self.n_hat*self.w2vec_hat[2]).real
        self.n1vec[0] = self.n2vec[0]/(twopi*self.sigma)
        self.n1vec[1] = self.n2vec[1]/(twopi*self.sigma)
        self.n1vec[2] = self.n2vec[2]/(twopi*self.sigma)

        if self.method == 'WB-tensor':
            self.n2tens[0,0] = ifftn(self.n_hat*self.w2tens_hat[0,0]).real
            self.n2tens[0,1] = ifftn(self.n_hat*self.w2tens_hat[0,1]).real
            self.n2tens[0,2] = ifftn(self.n_hat*self.w2tens_hat[0,2]).real
            self.n2tens[1,0] = ifftn(self.n_hat*self.w2tens_hat[1,0]).real
            self.n2tens[1,1] = ifftn(self.n_hat*self.w2tens_hat[1,1]).real
            self.n2tens[1,2] = ifftn(self.n_hat*self.w2tens_hat[1,2]).real
            self.n2tens[2,0] = ifftn(self.n_hat*self.w2tens_hat[2,0]).real
            self.n2tens[2,1] = ifftn(self.n_hat*self.w2tens_hat[2,1]).real
            self.n2tens[2,2] = ifftn(self.n_hat*self.w2tens_hat[2,2]).real

        self.n0 = self.n2/(np.pi*self.sigma**2)
        self.n1 = self.n2/(twopi*self.sigma)
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
        

    def free_energy_density(self,n):
        self.weighted_densities(n)

        phi = -self.n0*np.log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2])) + (self.phi3/(24*np.pi*self.oneminusn3**2))*(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))

        if self.method == 'WB-tensor':

            vTv = self.n2vec[0]*(self.n2tens[0,0]*self.n2vec[0]+self.n2tens[0,1]*self.n2vec[1]+self.n2tens[0,2]*self.n2vec[2])+self.n2vec[1]*(self.n2tens[1,0]*self.n2vec[0]+self.n2tens[1,1]*self.n2vec[1]+self.n2tens[1,2]*self.n2vec[2])+self.n2vec[2]*(self.n2tens[2,0]*self.n2vec[0]+self.n2tens[2,1]*self.n2vec[1]+self.n2tens[2,2]*self.n2vec[2])
            trT3 = self.n2tens[0,0]**3+self.n2tens[1,1]**3+self.n2tens[2,2]**3 + 3*self.n2tens[0,0]*self.n2tens[0,1]*self.n2tens[1,0]+ 3*self.n2tens[0,0]*self.n2tens[0,2]*self.n2tens[2,0]+ 3*self.n2tens[0,1]*self.n2tens[1,1]*self.n2tens[1,0]+ 3*self.n2tens[0,2]*self.n2tens[2,1]*self.n2tens[1,0]+ 3*self.n2tens[0,1]*self.n2tens[1,2]*self.n2tens[2,0]+ 3*self.n2tens[0,2]*self.n2tens[2,2]*self.n2tens[2,0]+ 3*self.n2tens[1,2]*self.n2tens[2,2]*self.n2tens[2,1]+ 3*self.n2tens[2,1]*self.n2tens[1,1]*self.n2tens[1,2]

            phi += (self.phi3/(24*np.pi*self.oneminusn3**2))*(9*vTv-4.5*trT3) 
        
        return phi

    def c1(self,n):
        self.weighted_densities(n)

        self.dPhidn0 = fftn(-np.log(self.oneminusn3 ))
        self.dPhidn1 = fftn(self.n2*self.phi2/self.oneminusn3 )
        self.dPhidn2 = fftn(self.n1*self.phi2/self.oneminusn3  + (3*self.n2*self.n2-3*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))*self.phi3/(24*np.pi*self.oneminusn3**2) )

        self.dPhidn3 = fftn(self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2]))*(self.dphi2dn3 + self.phi2/self.oneminusn3)/self.oneminusn3 + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/(24*np.pi*self.oneminusn3**2) ) 

        self.dPhidn1vec0 = fftn( -self.n2vec[0]*self.phi2/self.oneminusn3 )
        self.dPhidn1vec1 = fftn( -self.n2vec[1]*self.phi2/self.oneminusn3 )
        self.dPhidn1vec2 = fftn( -self.n2vec[2]*self.phi2/self.oneminusn3 )
        self.dPhidn2vec0 = fftn( -self.n1vec[0]*self.phi2/self.oneminusn3 + (- 6*self.n2*self.n2vec[0])*self.phi3/(24*np.pi*self.oneminusn3**2))
        self.dPhidn2vec1 = fftn(-self.n1vec[1]*self.phi2/self.oneminusn3 + (- 6*self.n2*self.n2vec[1])*self.phi3/(24*np.pi*self.oneminusn3**2))
        self.dPhidn2vec2 = fftn(-self.n1vec[2]*self.phi2/self.oneminusn3 +(-6*self.n2*self.n2vec[2])*self.phi3/(24*np.pi*self.oneminusn3**2))

        c1_hat = -(self.dPhidn2 + self.dPhidn1/(twopi*self.sigma) + self.dPhidn0/(np.pi*self.sigma**2))*self.w2_hat
        c1_hat -= self.dPhidn3*self.w3_hat
        c1_hat += (self.dPhidn2vec0+self.dPhidn1vec0/(twopi*self.sigma))*self.w2vec_hat[0] +(self.dPhidn2vec1+self.dPhidn1vec1/(twopi*self.sigma))*self.w2vec_hat[1] + (self.dPhidn2vec2+self.dPhidn1vec2/(twopi*self.sigma))*self.w2vec_hat[2]

        if self.method == 'WB-tensor':
            vTv = self.n2vec[0]*(self.n2tens[0,0]*self.n2vec[0]+self.n2tens[0,1]*self.n2vec[1]+self.n2tens[0,2]*self.n2vec[2])+self.n2vec[1]*(self.n2tens[1,0]*self.n2vec[0]+self.n2tens[1,1]*self.n2vec[1]+self.n2tens[1,2]*self.n2vec[2])+self.n2vec[2]*(self.n2tens[2,0]*self.n2vec[0]+self.n2tens[2,1]*self.n2vec[1]+self.n2tens[2,2]*self.n2vec[2])
            trT3 = self.n2tens[0,0]**3+self.n2tens[1,1]**3+self.n2tens[2,2]**3 + 3*self.n2tens[0,0]*self.n2tens[0,1]*self.n2tens[1,0]+ 3*self.n2tens[0,0]*self.n2tens[0,2]*self.n2tens[2,0]+ 3*self.n2tens[0,1]*self.n2tens[1,1]*self.n2tens[1,0]+ 3*self.n2tens[0,2]*self.n2tens[2,1]*self.n2tens[1,0]+ 3*self.n2tens[0,1]*self.n2tens[1,2]*self.n2tens[2,0]+ 3*self.n2tens[0,2]*self.n2tens[2,2]*self.n2tens[2,0]+ 3*self.n2tens[1,2]*self.n2tens[2,2]*self.n2tens[2,1]+ 3*self.n2tens[2,1]*self.n2tens[1,1]*self.n2tens[1,2]

            self.dPhidn3 = fftn( (9*vTv-4.5*trT3)*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/(24*np.pi*self.oneminusn3**2) ) 
            self.dPhidn2vec0 = fftn( (18*self.n2tens[0,0]*self.n2vec[0]+ 18*self.n2tens[0,1]*self.n2vec[1]+ 18*self.n2tens[0,2]*self.n2vec[2]+self.n2vec[1]*self.n2tens[1,0]+ 18*self.n2vec[2]*self.n2tens[2,0])*self.phi3/(24*np.pi*self.oneminusn3**2))
            self.dPhidn2vec1 = fftn(( 18*self.n2vec[0]*self.n2tens[0,1]+ 18*self.n2vec[1]*self.n2tens[1,0]+ 18*self.n2tens[1,1]*self.n2vec[1]+ 18*self.n2tens[1,2]*self.n2vec[2]+ 18*self.n2vec[2]*self.n2tens[2,1])*self.phi3/(24*np.pi*self.oneminusn3**2))
            self.dPhidn2vec2 = fftn(( 18*self.n2vec[0]*self.n2tens[0,2]+ 18*self.n2vec[1]*self.n2tens[1,2]+ 18*self.n2tens[2,0]*self.n2vec[0]+ 18*self.n2tens[2,1]*self.n2vec[1]+ 18*self.n2tens[2,2]*self.n2vec[2])*self.phi3/(24*np.pi*self.oneminusn3**2))
        
            self.dPhidn2tens00 = fftn((9*self.n2vec[0]*self.n2vec[0]-13.5*(self.n2tens[0,1]*self.n2tens[1,0]+self.n2tens[0,0]*self.n2tens[0,0]+self.n2tens[0,2]*self.n2tens[2,0]))*self.phi3/(24*np.pi*self.oneminusn3**2))
            self.dPhidn2tens01 = fftn((9*self.n2vec[0]*self.n2vec[1]-13.5*(self.n2tens[0,1]*self.n2tens[1,1]+self.n2tens[0,0]*self.n2tens[0,1]+self.n2tens[0,2]*self.n2tens[2,1]))*self.phi3/(24*np.pi*self.oneminusn3**2))
            self.dPhidn2tens02 = fftn((9*self.n2vec[0]*self.n2vec[2]-13.5*(self.n2tens[0,1]*self.n2tens[1,2]+self.n2tens[0,0]*self.n2tens[0,2]+self.n2tens[0,2]*self.n2tens[2,2]))*self.phi3/(24*np.pi*self.oneminusn3**2))
            self.dPhidn2tens10 = fftn((9*self.n2vec[1]*self.n2vec[0]-13.5*(self.n2tens[1,1]*self.n2tens[1,0]+self.n2tens[1,0]*self.n2tens[0,0]+self.n2tens[1,2]*self.n2tens[2,0]))*self.phi3/(24*np.pi*self.oneminusn3**2))
            self.dPhidn2tens11 = fftn((9*self.n2vec[1]*self.n2vec[1]-13.5*(self.n2tens[1,1]*self.n2tens[1,1]+self.n2tens[1,0]*self.n2tens[0,1]+self.n2tens[1,2]*self.n2tens[2,1]))*self.phi3/(24*np.pi*self.oneminusn3**2))
            self.dPhidn2tens12 = fftn((9*self.n2vec[1]*self.n2vec[2]-13.5*(self.n2tens[1,1]*self.n2tens[1,2]+self.n2tens[1,0]*self.n2tens[0,2]+self.n2tens[1,2]*self.n2tens[2,2]))*self.phi3/(24*np.pi*self.oneminusn3**2))
            self.dPhidn2tens20 = fftn((9*self.n2vec[2]*self.n2vec[0]-13.5*(self.n2tens[2,1]*self.n2tens[1,0]+self.n2tens[2,0]*self.n2tens[0,0]+self.n2tens[2,2]*self.n2tens[2,0]))*self.phi3/(24*np.pi*self.oneminusn3**2))
            self.dPhidn2tens21 = fftn((9*self.n2vec[2]*self.n2vec[1]-13.5*(self.n2tens[2,1]*self.n2tens[1,1]+self.n2tens[2,0]*self.n2tens[0,1]+self.n2tens[2,2]*self.n2tens[2,1]))*self.phi3/(24*np.pi*self.oneminusn3**2))
            self.dPhidn2tens22 = fftn((9*self.n2vec[2]*self.n2vec[2]-13.5*(self.n2tens[2,1]*self.n2tens[1,2]+self.n2tens[2,0]*self.n2tens[0,2]+self.n2tens[2,2]*self.n2tens[2,2]))*self.phi3/(24*np.pi*self.oneminusn3**2))

            c1_hat -= self.dPhidn3*self.w3_hat
            c1_hat += (self.dPhidn2vec0+self.dPhidn1vec0/(twopi*self.sigma))*self.w2vec_hat[0] +(self.dPhidn2vec1+self.dPhidn1vec1/(twopi*self.sigma))*self.w2vec_hat[1] + (self.dPhidn2vec2+self.dPhidn1vec2/(twopi*self.sigma))*self.w2vec_hat[2]
            c1_hat -= self.dPhidn2tens00*self.w2tens_hat[0,0] + self.dPhidn2tens01*self.w2tens_hat[0,1] + self.dPhidn2tens02*self.w2tens_hat[0,2] + self.dPhidn2tens10*self.w2tens_hat[1,0] + self.dPhidn2tens11*self.w2tens_hat[1,1] + self.dPhidn2tens12*self.w2tens_hat[1,2] + self.dPhidn2tens20*self.w2tens_hat[2,0] + self.dPhidn2tens21*self.w2tens_hat[2,1] + self.dPhidn2tens22*self.w2tens_hat[2,2]

            del self.dPhidn2tens00, self.dPhidn2tens01, self.dPhidn2tens02, self.dPhidn2tens10, self.dPhidn2tens11, self.dPhidn2tens12, self.dPhidn2tens20, self.dPhidn2tens21, self.dPhidn2tens22

        del self.dPhidn0,self.dPhidn1,self.dPhidn2,self.dPhidn3,self.dPhidn1vec0,self.dPhidn1vec1,self.dPhidn1vec2,self.dPhidn2vec0,self.dPhidn2vec1,self.dPhidn2vec2
        
        return ifftn(c1_hat).real

    def mu(self,rhob):
        n3 = np.sum(rhob*np.pi*self.sigma**3/6)
        n2 = np.sum(rhob*np.pi*self.sigma**2)
        n1 = np.sum(rhob*self.sigma/2)
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

        return (dPhidn0+dPhidn1*self.sigma/2+dPhidn2*np.pi*self.sigma**2+dPhidn3*np.pi*self.sigma**3/6)

##### Take a example using FMT ######
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    
    #############################
    test1 = False
    test2 = True # calculate the radial distribution function
    test5 = False # solid-fluid phase diagram (3D)
    test6 = False # free minimization a la Lutsko

    if test1:
        delta = 0.1
        N = 128
        L = N*delta
        Narray = np.array([N,N,N])
        deltaarray = np.array([delta,delta,delta])
        fmt = FMT(Narray,deltaarray)

        w = ifftn(fmt.w3_hat).real
        x = np.arange(-L/2,L/2,delta)
        y = np.arange(-L/2,L/2,delta)
        X, Y = np.meshgrid(x,y)

        print(w.sum(),np.pi/6,np.abs(w.sum()-np.pi/6)/np.pi/6)

        wmax = np.max(w[:,:,N//2])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        cmap = plt.get_cmap('jet')
        cp = ax.contourf(X, Y, w[:,:,N//2]/wmax,20, cmap=cmap)
        # ax.set_title(r'$\omega_3(r)=\Theta(\sigma/2-r)$')
        fig.colorbar(cp,ticks=[0,0.2,0.4,0.6,0.8,1.0]) 
        ax.set_xlabel(r'$x/\sigma$')
        ax.set_ylabel(r'$y/\sigma$')
        fig.savefig('omega3-N%d.pdf'% N, bbox_inches='tight')
        plt.show()
        plt.close()

        w = ifftn(fmt.w2_hat).real

        wmax = np.max(w[:,:,N//2])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        print(w.sum(),np.pi,np.abs(np.pi-w.sum())/np.pi)
        cp2 = ax.contourf(X, Y, w[:,:,N//2]/wmax,20, cmap=cmap)
        # ax.set_title(r'$\omega_2(r)=\delta(\sigma/2-r)$')
        fig.colorbar(cp2,ticks=[0,0.2,0.4,0.6,0.8,1.0]) 
        ax.set_xlabel(r'$x/\sigma$')
        ax.set_ylabel(r'$y/\sigma$')
        fig.savefig('omega2-N%d.pdf'% N, bbox_inches='tight')
        plt.show()
        plt.close()
            

    if test2:
        sigma = 1.0
        # parameters of the gridsize
        N, delta = 32, 0.4*sigma # 32³ grid
        # N, delta = 64, 0.2*sigma # 64³ grid
        # N, delta = 128, 0.1*sigma # 128³ grid
        # N, delta = 256, 0.05*sigma # 256³ grid

        L = N*delta
        z = np.linspace(-L/2,L/2,N,endpoint=False)
        Narray = np.array([N,N,N])
        deltaarray = np.array([delta,delta,delta])

        fmt = FMT(Narray,deltaarray)

        n0 = np.ones((N,N,N),dtype=np.float32)

        X,Y,Z = np.meshgrid(z,z,z,indexing ='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)
        mask = R<sigma
        n0[mask] = 1.0e-16

        del X,Y,Z,R,mask
        
        n = n0.copy()

        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            fexc = fmt.free_energy_density(n)
            Omegak = n*(lnn-1.0) + fexc - mu*n
            return Omegak.sum()*delta**3/L**3

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            c1exc = fmt.c1(n)
            return n*(lnn -c1exc - mu)*delta**3/L**3

        rhobarray = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

        for rhob in rhobarray:
            print("Doing the rhob =",rhob) 

            mu = np.log(rhob) + fmt.mu(rhob)
            
            lnn = np.log(n0) + np.log(rhob)
            
            [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,alpha0=0.62,rtol=1e-7,dt=40.0,logoutput=True)

            n[:] = np.exp(nsol)

            np.save('results/densityfield-fmt-wbi-rhob'+str(rhob)+'-N'+str(N)+'-delta'+str(delta)+'.npy',n)
    
    if test5: 
        # a = 1.0 #sc
        a = np.sqrt(2) # fcc
        # a = 2*np.sqrt(3)/3 #bcc
        a = 1.0 #hcp
        N = 64
        Narray = np.array([N,N,N])
        delta = 0.05
        # delta = 2*a/N
        
        deltaarray = np.array([delta,delta,delta])

        L = deltaarray*Narray
        Vol = L[0]*L[1]*L[2]

        FMT = FMT(Narray,deltaarray)

        print('The fluid-solid phase diagram')
        print('N=',N)
        print('L=',L)
        print('delta=',delta)

        # define the variables to the gaussian parametrization
        # R = a*np.array([[1,0,0],[0,1,0],[0,0,1]]) #sc
        # R = 0.5*a*np.array([[0,1,1],[1,0,1],[1,1,0]]) #fcc
        # R = 0.5*a*np.array([[1,1,-1],[-1,1,1],[1,-1,1]]) #bcc
        R = 0.5*a*np.array([[0,0,2],[1,np.sqrt(3),0],[-1,np.sqrt(3),0]]) #hcp
        def gaussian(alpha,x,y,z):
            rho = 1e-16*np.ones((N,N,N),dtype=np.float32)
            for n1 in range(-3,4):
                for n2 in range(-3,4):
                    for n3 in range(-3,4):
                        # rho += np.power(alpha/np.pi,1.5)*np.exp(-alpha*((x-n1*R[0,0]-n2*R[1,0]-n3*R[2,0])**2+(y-n1*R[0,1]-n2*R[1,1]-n3*R[2,1])**2+(z-n1*R[0,2]-n2*R[1,2]-n3*R[2,2])**2))
                        rho += (6/np.pi)*Theta(x,y,z,n1*R[0,0]+n2*R[1,0]+n3*R[2,0],n1*R[0,1]+n2*R[1,1]+n3*R[2,1],n1*R[0,2]+n2*R[1,2]+n3*R[2,2])
            return rho 

        n = np.empty((N,N,N),dtype=np.float32)
        n_hat = np.empty((N,N,N),dtype=np.complex64)

        x = np.linspace(-L[0]/2,L[0]/2,N)
        X,Y,Z = np.meshgrid(x,x,x)
        
        n[:] = gaussian(50.0,X,Y,Z)
        rhomean = n.sum()*delta**3/Vol 
        print(rhomean)
        nsig = int(0.5*a/delta)
        plt.imshow(n[N//2], cmap='viridis')
        plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.show()

        lnnsol = np.log(n)

        # lnnflu = np.log(n) - np.log(rhomean) + np.log(0.7)
        
        lnnflu = np.log(0.7)*np.ones((N,N,N),dtype=np.float32)

        n[:] = np.exp(lnnflu)
        rhomean = n.sum()*delta**3/Vol 
        print(rhomean)
        plt.imshow(n[N-1], cmap='viridis')
        plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.show()

        rhob = 0.8
        eta = rhob*np.pi/6.0
        mumin = np.log(rhob) + FMT.mu(rhob)

        rhob = 1.0
        eta = rhob*np.pi/6.0
        mumax = np.log(rhob) + FMT.mu(rhob)

        muarray = np.linspace(15.75,17,10,endpoint=True)
        # muarray = np.array([15.74,15.75,15.76])

        output = True

        ## The Grand Canonical Potential
        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fftn(n)
            phi = FMT.Phi(self.n_hat)
            FHS = np.sum(phi)*delta**3
            Fid = np.sum(n*(lnn-1.0))*delta**3
            N = n.sum()*delta**3
            return (Fid+FHS-mu*N)/Vol

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fftn(n)
            dphidn = FMT.dPhidn(self.n_hat)
            return n*(lnn + dphidn - mu)*delta**3/Vol

        print('#######################################')
        print("mu\trho\trho2\tOmega1\tOmega2")

        for i in range(muarray.size):

            mu = muarray[i]

            [lnn1,Omegasol,Niter] = optimize_fire2(lnnflu,Omega,dOmegadnR,mu,5.0e-10,1.0,output)

            plt.imshow(n[N//2], cmap='Greys_r')
            plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.show()

            [lnn2,Omegasol2,Niter] = optimize_fire2(lnnsol,Omega,dOmegadnR,mu,5.0e-10,1.0,output)

            rhomean = np.exp(lnn1).sum()*delta**3/Vol
            rhomean2 = np.exp(lnn2).sum()*delta**3/Vol

            plt.imshow(n[N//2], cmap='Greys_r')
            plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.show()

            print(mu,rhomean,rhomean2,Omegasol,Omegasol2)

    if test6: 
        # a = 1.0 #sc
        a = np.sqrt(2) # fcc
        # a = 2*np.sqrt(3)/3 #bcc
        # a = 1.0 #hcp
        
        delta = 0.01
        deltaarray = np.array([delta,delta,delta])
        Narr = np.array([127,128,129,130,131,132,133,134,135])

        print('#######################################')
        print("mu\trho\tOmega")

        for N in Narr: 

            Narray = np.array([N,N,N])

            L = deltaarray*Narray
            Vol = L[0]*L[1]*L[2]

            n = np.empty((N,N,N),dtype=np.float32)
            n_hat = np.empty((N,N,N),dtype=np.complex64)

            fmt = FMT(Narray,deltaarray)

            # print('The fluid-solid phase diagram')
            print('N=',N)
            # print('L=',L)
            # print('delta=',delta)

            # define the variables to the gaussian parametrization
            # R = a*np.array([[1,0,0],[0,1,0],[0,0,1]]) #sc
            R = 0.5*a*np.array([[0,1,1],[1,0,1],[1,1,0]]) #fcc
            # R = 0.5*a*np.array([[1,1,-1],[-1,1,1],[1,-1,1]]) #bcc
            # R = 0.5*a*np.array([[0,0,2],[1,np.sqrt(3),0],[-1,np.sqrt(3),0]]) #hcp
            def gaussian(alpha,x,y,z):
                rho = 1e-16*np.ones((N,N,N),dtype=np.float32)
                for n1 in range(-3,4):
                    for n2 in range(-3,4):
                        for n3 in range(-3,4):
                            # rho += np.power(alpha/np.pi,1.5)*np.exp(-alpha*((x-n1*R[0,0]-n2*R[1,0]-n3*R[2,0])**2+(y-n1*R[0,1]-n2*R[1,1]-n3*R[2,1])**2+(z-n1*R[0,2]-n2*R[1,2]-n3*R[2,2])**2))
                            rho += (6/np.pi)*Theta(x,y,z,n1*R[0,0]+n2*R[1,0]+n3*R[2,0],n1*R[0,1]+n2*R[1,1]+n3*R[2,1],n1*R[0,2]+n2*R[1,2]+n3*R[2,2])
                return rho 

            x = np.linspace(-L[0]/2,L[0]/2,N)
            X,Y,Z = np.meshgrid(x,x,x)
            
            n[:] = gaussian(50.0,X,Y,Z)
            # rhomean = n.sum()*delta**3/Vol 
            # print(rhomean)
            # nsig = int(0.5*a/delta)
            plt.imshow(n[0], cmap='viridis')
            plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.show()

            lnnsol = np.log(n)

            # muarray = np.linspace(15.75,17,10,endpoint=True)
            muarray = np.array([15.75])

            output = False

            ## The Grand Canonical Potential
            def Omega(lnn,mu):
                n[:] = np.exp(lnn)
                n_hat[:] = fftn(n)
                phi = fmt.Phi(self.n_hat)
                FHS = np.sum(phi)*delta**3
                Fid = np.sum(n*(lnn-1.0))*delta**3
                N = n.sum()*delta**3
                return (Fid+FHS-mu*N)/Vol

            def dOmegadnR(lnn,mu):
                n[:] = np.exp(lnn)
                n_hat[:] = fftn(n)
                dphidn = fmt.dPhidn(self.n_hat)
                return n*(lnn + dphidn - mu)*delta**3/Vol

            for i in range(muarray.size):

                mu = muarray[i]

                [lnn2,Omegasol2,Niter] = optimize_fire2(lnnsol,Omega,dOmegadnR,mu,5.0e-10,0.2,output)

                rhomean2 = np.exp(lnn2).sum()*delta**3/Vol

                plt.imshow(n[N//2], cmap='Greys_r')
                plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
                plt.xlabel('$x$')
                plt.ylabel('$y$')
                plt.show()

                print(mu,rhomean2,Omegasol2)