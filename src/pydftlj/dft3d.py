import numpy as np
from numba import jit, njit
from numpy import pi, exp, log, sqrt, round, sum, abs, allclose, float32, complex64, linspace, isscalar, array, zeros, zeros_like, empty, empty_like, meshgrid
import timeit
from eos import LJEOS, BHdiameter
from dcf import DCF3dFT, ljBH3dFT
from fmtaux import sigmaLancsozFT,translationFT, w3FT, w2FT, phi2func, phi3funcWBII, phi3funcWBI, dphi3dnfuncWBI, dphi2dnfunc, dphi3dnfuncWBII
from multiprocessing import Pool
from scipy.fftpack import fftn, ifftn
try:
    import pyfftw
    from pyfftw.interfaces.scipy_fftpack import fftn, ifftn
    # import multiprocessing
    # pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    # pyfftw.config.NUM_THREADS = 2
    pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)
except ImportError: 
    pass # Rely on scipy.fftpack routines

# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-09-23
# Updated: 2023-02-27

# Define the function to be parallelized
def calc_convolution(rho_hat, w_hat):
    return ifftn(rho_hat * w_hat).real

" The DFT model for Lennard-Jones fluid on 3d geometries"

" The hard-sphere FMT functional implemented are the following: "
" fmtmethod = RF (Rosenfeld functional) "
"           = WBI (White Bear version I) "
"           = WBII (White Bear version II) "

class dft3d():
    def __init__(self,gridsize,fmtfunctional='antisymmetrized',fmtmethod='WBI',ljmethod='MMFA'):
        self.fmtfunctional = fmtfunctional
        self.fmtmethod = fmtmethod 
        self.ljmethod = ljmethod 
        self.Ngrid = gridsize 
        self.Ngridtot = self.Ngrid[0]*self.Ngrid[1]*self.Ngrid[2]

    def Set_Geometry(self,L):
        if isscalar(L): 
            self.L = array([L,L,L])
        else: 
            self.L = L
        self.Vol = self.L[0]*self.L[1]*self.L[2]

        self.x = linspace(0.0,self.L[0],self.Ngrid[0])
        self.y = linspace(0.0,self.L[1],self.Ngrid[1])
        self.z = linspace(0.0,self.L[2],self.Ngrid[2])

        self.delta = array([self.x[1]-self.x[0],self.y[1]-self.y[0],self.z[1]-self.z[0]])

    def Set_FluidProperties(self,sigma=1.0,epsilon=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
    
    def Set_Temperature(self,kT):

        self.kT = kT
        self.beta = 1/self.kT
        if self.ljmethod == 'MFA' or self.ljmethod == 'BFD' or self.ljmethod == 'WDA' or self.ljmethod == 'MMFA':
            self.d = round(BHdiameter(self.kT,sigma=self.sigma,epsilon=self.epsilon),3)
            
        else:
            self.d = self.sigma        

        self.rho = zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=float32)
        self.rho_hat = empty((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=complex64)
        if self.ljmethod == 'BFD':
            self.rhodiff_hat = empty_like(self.rho_hat)
        elif self.ljmethod == 'WDA':
            self.rhobar = empty_like(self.rho)
            self.mu_hat = empty_like(self.rho_hat)
        elif self.ljmethod == 'MMFA':
            self.rhobar = empty_like(self.rho)
            self.uint = empty_like(self.rho)
            self.mucore_hat = empty_like(self.rho_hat)
        self.Vext = zeros_like(self.rho)

        self.c1 = empty_like(self.rho)
        self.c1hs = empty_like(self.rho)
        self.c1att = empty_like(self.rho)

        self.w3_hat = empty_like(self.rho_hat)
        self.w2_hat = empty_like(self.rho_hat)
        self.w2vec_hat = empty((3,self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=complex64)

        self.n0 = empty_like(self.rho)
        self.n1 = empty_like(self.rho)
        self.n3 = empty_like(self.rho)
        self.n2 = empty_like(self.rho)
        self.n2vec = empty((3,self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=float32)
        self.n1vec = empty_like(self.n2vec)
        
        kx = np.fft.fftfreq(self.Ngrid[0], d=self.delta[0])*2*pi
        ky = np.fft.fftfreq(self.Ngrid[1], d=self.delta[1])*2*pi
        kz = np.fft.fftfreq(self.Ngrid[2], d=self.delta[2])*2*pi
        self.kcut = array([kx.max(),ky.max(),kz.max()])
        self.Kx,self.Ky,self.Kz = meshgrid(kx,ky,kz,indexing ='ij')
        self.K = sqrt(self.Kx**2 + self.Ky**2 + self.Kz**2)
        del kx, ky, kz

        self.dV = self.delta[0]*self.delta[1]*self.delta[2]

        # Defining the weight functions
        self.w3_hat[:] = w3FT(self.K,sigma=self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)*translationFT(self.Kx,self.Ky,self.Kz,0.5*self.L)
        self.w2_hat[:] = w2FT(self.K,sigma=self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)*translationFT(self.Kx,self.Ky,self.Kz,0.5*self.L)
        self.w2vec_hat[0] = -1.0j*self.Kx*self.w3_hat
        self.w2vec_hat[1] = -1.0j*self.Ky*self.w3_hat
        self.w2vec_hat[2] = -1.0j*self.Kz*self.w3_hat

        if self.ljmethod == 'BFD':
            self.c2_hat = DCF3dFT(self.K,self.rhob,self.kT,sigma=self.sigma,epsilon=self.epsilon)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)*translationFT(self.Kx,self.Ky,self.Kz,0.5*self.L) # to avoid Gibbs phenomenum 
        elif self.ljmethod == 'WDA':
            psi = 1.3862
            self.w_hat = w3FT(self.K,sigma=2*psi*self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)*translationFT(self.Kx,self.Ky,self.Kz,0.5*self.L)/(4*pi*(psi*self.d)**3/3)
        elif self.ljmethod == 'MMFA':
            self.amft = -32*pi*self.epsilon*self.sigma**3/9
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)
            self.fcore = lambda rr: ljeos.fatt(rr,self.kT) - 0.5*self.amft*rr**2
            self.mucore = lambda rr: ljeos.muatt(rr,self.kT) - self.amft*rr
            self.ulj_hat = ljBH3dFT(self.K,self.sigma,self.epsilon)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)*translationFT(self.Kx,self.Ky,self.Kz,0.5*self.L) # to avoid Gibbs phenomenum

    def Set_BulkDensity(self,rhob):

        self.rhob = rhob 

        if self.ljmethod == 'BFD':
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)
            self.flj = ljeos.fatt(self.rhob,self.kT)
            self.mulj = ljeos.muatt(self.rhob,self.kT)
        elif self.ljmethod == 'WDA':
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)
            self.mulj = ljeos.muatt(self.rhob,self.kT)
            self.fdisp = lambda rr: ljeos.fatt(rr,self.kT)
            self.mudisp = lambda rr: ljeos.muatt(rr,self.kT) 
        elif self.ljmethod == 'MMFA':
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)
            self.mulj = ljeos.muatt(self.rhob,self.kT)
            
        self.Calculate_mu()
        
    def Set_External_Potential(self,Vext):
        self.Vext[:] = Vext
        self.mask = self.Vext>=16128
        self.Vext[self.mask] = 16128

    def Set_InitialCondition(self):
        self.rho[:] = self.rhob
        self.rho[self.mask] = 1.e-16
        self.Update_System()

    def GetSystemInformation(self):
        print('============== The DFT 3D for LJ fluids ==============')
        print('Methods:')
        print('FMT :', self.fmtmethod, '-', self.fmtfunctional )
        print('Attractive :', self.ljmethod)
        print('The grid is',self.Ngrid)
        print('--- Geometry properties ---')
        print('Lx =', self.L[0], ' A')
        print('Ly =', self.L[1], ' A')
        print('Lz =', self.L[2], ' A')
        print('Vol =',self.Vol, ' A³')
    
    def GetFluidInformation(self):
        print('--- Fluid properties ---')
        print('epsilon/kB =', self.epsilon, ' K')
        print('sigma =', self.sigma, ' A')

    def GetFluidTemperatureInformation(self):
        print('Temperature =', self.kT, ' K')
        print('Baker-Henderson diameter =', self.d, ' A')

    def GetFluidDensityInformation(self):
        print('Bulk Density:',self.rhob, ' particles/A³')
        print('muid:',self.muid.round(3))
        print('muhs:',self.muhs.round(3))
        print('muatt:',self.muatt.round(3))

    def Update_System(self):
        self.Calculate_weighted_densities()
        self.Calculate_c1()
        self.Calculate_Omega()

    def Calculate_weighted_densities(self):
        self.rho_hat[:] = fftn(self.rho)

        # Set up the multiprocessing pool
        pool = Pool()

        # Define the inputs for the parallel function
        inputs = [(self.rho_hat, self.w3_hat), (self.rho_hat, self.w2_hat),(self.rho_hat,self.w2vec_hat[0]),(self.rho_hat,self.w2vec_hat[1]),(self.rho_hat,self.w2vec_hat[2])]

        # Map the function to the inputs using the multiprocessing pool
        results = pool.starmap(calc_convolution, inputs)

        # Unpack the results and assign to self.n 
        self.n3[:] = results[0]
        self.n2[:] = results[1]
        self.n2vec[0] = results[2]
        self.n2vec[1] = results[3]
        self.n2vec[2] = results[4]

        self.n3[self.n3>=1.0] = 1.0-1e-4 # to avoid Nan on some calculations
        self.xi = np.sum(self.n2vec*self.n2vec,axis=0)/(self.n2**2)
        self.xi[self.xi>=1.0] = 1.0

        self.n1vec[:] = self.n2vec/(2*pi*self.d)

        self.n0[:] = self.n2/(pi*self.d**2)
        self.n1[:] = self.n2/(2*pi*self.d)
        self.oneminusn3 = 1-self.n3

        if self.fmtmethod == 'RF': 
            self.phi2 = 1.0
            self.dphi2dn3 = 0.0
            self.phi3 = 1.0
            self.dphi3dn3 = 0.0
        elif self.fmtmethod == 'WBI': 
            self.phi2 = 1.0
            self.dphi2dn3 = 0.0
            self.phi3 = phi3funcWBI(self.n3)
            self.dphi3dn3 = dphi3dnfuncWBI(self.n3)
        elif self.fmtmethod == 'WBII': 
            self.phi2 = phi2func(self.n3)
            self.dphi2dn3 = dphi2dnfunc(self.n3)
            self.phi3 = phi3funcWBII(self.n3)
            self.dphi3dn3 = dphi3dnfuncWBII(self.n3)
        
        if self.ljmethod == 'BFD':
            self.rhodiff_hat[:] = fftn(self.rho-self.rhob)
        elif self.ljmethod == 'WDA':
            self.rhobar[:] = ifftn(self.rho_hat*self.w_hat).real
            self.mu_hat[:] =  fftn(self.mudisp(self.rhobar))
        elif self.ljmethod == 'MMFA':
            self.rhobar[:] = ifftn(self.rho_hat*self.w3_hat/(pi*self.d**3/6)).real
            self.uint[:] = ifftn(self.rho_hat*self.ulj_hat).real
            self.mucore_hat[:] = fftn(self.mucore(self.rhobar))

    def Calculate_Free_energy(self):
        self.Fid = self.kT*sum(self.rho*(log(self.rho)-1.0))*self.dV

        if self.fmtfunctional == 'symmetrized':
            phi = -self.n0*log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2])) + (self.phi3/(24*pi*self.oneminusn3**2))*(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))
        elif self.fmtfunctional == 'antisymmetrized':
            phi = -self.n0*log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2])) + (self.phi3/(24*pi*self.oneminusn3**2))*self.n2*self.n2*self.n2*(1-self.xi)**3
        
        self.Fhs = self.kT*sum(phi)*self.dV

        if self.ljmethod == 'BFD':
            phi[:] = self.flj + self.mulj*(self.rho-self.rhob) -self.kT*0.5*(self.rho-self.rhob)*ifftn(self.rhodiff_hat*self.c2_hat).real
        elif self.ljmethod == 'WDA':
            phi[:] = self.fdisp(self.rhobar)
        elif self.ljmethod == 'MMFA':
            phi[:] = 0.5*self.rho*self.uint + self.fcore(self.rhobar)
        else:
            phi[:] = 0.0
        self.Flj = sum(phi)*self.dV

        self.Fexc =  self.Fhs + self.Flj
        self.F = self.Fid + self.Fexc

    def Calculate_Omega(self):
        self.Calculate_Free_energy()
        self.Omega = self.F + sum((self.Vext-self.mu)*self.rho)*self.dV

    def Calculate_c1(self):
        if self.fmtfunctional == 'symmetrized':
            dPhidn0 = fftn(-log(self.oneminusn3 ))
            dPhidn1 = fftn(self.n2*self.phi2/self.oneminusn3 )
            dPhidn2 = fftn(self.n1*self.phi2/self.oneminusn3  + (3*self.n2*self.n2-3*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))*self.phi3/(24*pi*self.oneminusn3**2) )

            dPhidn3 = fftn(self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2]))*(self.dphi2dn3 + self.phi2/self.oneminusn3)/self.oneminusn3 + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/(24*pi*self.oneminusn3**2) ) 

            dPhidn1vec0 = fftn( -self.n2vec[0]*self.phi2/self.oneminusn3 )
            dPhidn1vec1 = fftn( -self.n2vec[1]*self.phi2/self.oneminusn3 )
            dPhidn1vec2 = fftn( -self.n2vec[2]*self.phi2/self.oneminusn3 )
            dPhidn2vec0 = fftn( -self.n1vec[0]*self.phi2/self.oneminusn3 + (- 6*self.n2*self.n2vec[0])*self.phi3/(24*pi*self.oneminusn3**2))
            dPhidn2vec1 = fftn(-self.n1vec[1]*self.phi2/self.oneminusn3 + (- 6*self.n2*self.n2vec[1])*self.phi3/(24*pi*self.oneminusn3**2))
            dPhidn2vec2 = fftn(-self.n1vec[2]*self.phi2/self.oneminusn3 +(-6*self.n2*self.n2vec[2])*self.phi3/(24*pi*self.oneminusn3**2))

        elif self.fmtfunctional == 'antisymmetrized':
            dPhidn0 = fftn(-log(self.oneminusn3 ))
            dPhidn1 = fftn(self.n2*self.phi2/self.oneminusn3 )
            dPhidn2 = fftn(self.n1*self.phi2/self.oneminusn3  + 3*(self.n2**2)*(1+self.xi)*((1-self.xi)**2)*self.phi3/(24*pi*self.oneminusn3**2))

            dPhidn3 = fftn(self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2]))*(self.dphi2dn3 + self.phi2/self.oneminusn3)/self.oneminusn3 + (self.n2*self.n2*self.n2*(1-self.xi)**3)*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/(24*pi*self.oneminusn3**2) ) 

            dPhidn1vec0 = fftn( -self.n2vec[0]*self.phi2/self.oneminusn3 )
            dPhidn1vec1 = fftn( -self.n2vec[1]*self.phi2/self.oneminusn3 )
            dPhidn1vec2 = fftn( -self.n2vec[2]*self.phi2/self.oneminusn3 )
            dPhidn2vec0 = fftn( -self.n1vec[0]*self.phi2/self.oneminusn3 + (- 6*self.n2*self.n2vec[0]*(1-self.xi)**2)*self.phi3/(24*pi*self.oneminusn3**2))
            dPhidn2vec1 = fftn(-self.n1vec[1]*self.phi2/self.oneminusn3 + (- 6*self.n2*self.n2vec[1]*(1-self.xi)**2)*self.phi3/(24*pi*self.oneminusn3**2))
            dPhidn2vec2 = fftn(-self.n1vec[2]*self.phi2/self.oneminusn3 +(- 6*self.n2*self.n2vec[2]*(1-self.xi)**2)*self.phi3/(24*pi*self.oneminusn3**2))

        c1_hat = -(dPhidn2 + dPhidn1/(2*pi*self.d) + dPhidn0/(pi*self.d**2))*self.w2_hat
        c1_hat[:] -= dPhidn3*self.w3_hat
        c1_hat[:] += (dPhidn2vec0+dPhidn1vec0/(2*pi*self.d))*self.w2vec_hat[0] +(dPhidn2vec1+dPhidn1vec1/(2*pi*self.d))*self.w2vec_hat[1] + (dPhidn2vec2+dPhidn1vec2/(2*pi*self.d))*self.w2vec_hat[2]

        del dPhidn0,dPhidn1,dPhidn2,dPhidn3,dPhidn1vec0,dPhidn1vec1,dPhidn1vec2,dPhidn2vec0,dPhidn2vec1,dPhidn2vec2

        self.c1hs[:] = ifftn(c1_hat).real

        if self.ljmethod == 'BFD':
            self.c1att[:] = -self.beta*self.mulj+ ifftn(self.rhodiff_hat*self.c2_hat).real
        elif self.ljmethod == 'WDA':
            self.c1att[:] = -self.beta*ifftn(self.mu_hat*self.w_hat).real
        elif self.ljmethod == 'MMFA':
            self.c1att[:] = -self.beta*self.uint -self.beta*ifftn(self.mucore_hat*self.w3_hat/(pi*self.d**3/6)).real
        else:
            self.c1att[:] = 0.0

        self.c1[:] = self.c1hs + self.c1att

    def Calculate_mu(self):
        self.muid = self.kT*log(self.rhob)

        n3 = self.rhob*pi*self.d**3/6
        n2 = self.rhob*pi*self.d**2
        n1 = self.rhob*self.d/2
        n0 = self.rhob

        if self.fmtmethod == 'RF':
            phi2 = 1.0
            dphi2dn3 = 0.0
            phi3 = 1.0
            dphi3dn3 = 0.0
        elif self.fmtmethod == 'WBI': 
            phi2 = 1.0
            dphi2dn3 = 0.0
            phi3 = phi3funcWBI(n3)
            dphi3dn3 = dphi3dnfuncWBI(n3)
        elif self.fmtmethod == 'WBII': 
            phi2 = phi2func(n3)
            dphi2dn3 = dphi2dnfunc(n3)
            phi3 = phi3funcWBII(n3)
            dphi3dn3 = dphi3dnfuncWBII(n3)

        dPhidn0 = -log(1-n3)
        dPhidn1 = n2*phi2/(1-n3)
        dPhidn2 = n1*phi2/(1-n3) + (3*n2**2)*phi3/(24*pi*(1-n3)**2)
        dPhidn3 = n0/(1-n3) +(n1*n2)*(dphi2dn3 + phi2/(1-n3))/(1-n3) + (n2**3)*(dphi3dn3+2*phi3/(1-n3))/(24*pi*(1-n3)**2)

        self.muhs = self.kT*(dPhidn0+dPhidn1*self.d/2+dPhidn2*pi*self.d**2+dPhidn3*pi*self.d**3/6)

        if self.ljmethod == 'BFD' or self.ljmethod == 'WDA' or self.ljmethod == 'MMFA':
            self.muatt = self.mulj
        else:
            self.muatt = 0.0

        self.muexc = self.muhs + self.muatt
        self.mu = self.muid + self.muexc

    def Calculate_Equilibrium(self,alpha0=0.6,dt=1.0,atol=1e-5,rtol=1e-3,max_iter=9999,method='fire',logoutput=False):

        starttime = timeit.default_timer()

        lnrho = log(self.rho) 

        if method == 'picard':
            # Picard algorithm
            alpha = alpha0

            self.Update_System()
            F = -self.rho*(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)*self.dV

            for i in range(max_iter):
                lnrhonew = self.c1 + self.beta*self.mu - self.beta*self.Vext
                lnrhonew[self.mask] = log(1.0e-16)
                
                lnrho[:] = (1-alpha)*lnrho + alpha*lnrhonew
                self.rho[:] = exp(lnrho)
                self.rho[self.mask] = 1.0e-16
                self.Update_System()

                F[:] = -self.rho*(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)*self.dV
                self.Niter = i+1

                sk=atol+rtol*np.abs(lnrho)
                error = np.linalg.norm(F/sk)
                if error < 1.0: break

                if logoutput: print(i,self.Omega,error)
                

        elif method == 'fire':
            # Fire algorithm
            Ndelay = 5
            Nnegmax = 2000
            dtmax = 10*dt
            dtmin = 0.02*dt
            alpha = alpha0
            Npos = 0
            Nneg = 0
            finc = 1.1
            fdec = 0.5
            fa = 0.99

            self.Update_System()
            V = zeros_like(self.rho)
            F = -self.rho*(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)*self.dV

            for i in range(max_iter):

                P = (F*V).sum() # dissipated power
                if (P>0):
                    Npos = Npos + 1
                    if Npos>Ndelay:
                        dt = min(dt*finc,dtmax)
                        alpha = alpha*fa
                else:
                    Npos = 0
                    Nneg = Nneg + 1
                    if Nneg > Nnegmax: break
                    if i> Ndelay:
                        dt = max(dt*fdec,dtmin)
                        alpha = alpha0
                    lnrho[:] += - 0.5*dt*V
                    V[:] = 0.0
                    self.rho[:] = exp(lnrho)
                    self.rho[self.mask] = 1.0e-16
                    self.Update_System()

                V[:] += 0.5*dt*F
                V[:] = (1-alpha)*V + alpha*F*np.linalg.norm(V)/np.linalg.norm(F)
                lnrho[:] += dt*V
                self.rho[:] = exp(lnrho)
                self.rho[self.mask] = 1.0e-16
                self.Update_System()
                F[:] = -self.rho*(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)*self.dV
                V[:] += 0.5*dt*F

                sk=atol+rtol*np.abs(lnrho)
                error = np.linalg.norm(F/sk)
                if error < 1.0: break

                # error = max(F.max(),abs(F.min()))
                # if allclose(F, 0.0, rtol=0.0, atol=atol): break
                if logoutput: print(i,self.Omega,error)
            self.Niter = i+1

            del V, F  

        self.Nadstot = self.rho.sum()*self.dV
        

        if logoutput:
            print("Time to achieve equilibrium:", timeit.default_timer() - starttime, 'sec')
            print('Number of iterations:', self.Niter)
            print('error:', error)
            print('---- Equilibrium quantities ----')
            print('Fid =',self.Fid)
            print('Fexc =',self.Fexc)
            print('Omega =',self.Omega)
            print('Nbulk =',self.rhob*self.Vol)
            print('Nadstot =',self.Nadstot)
            print('================================')

    def Set_Diffusivity(self,D=1.0):
        self.D = D*1e11
        
        print('Fluid transport properties:')
        print('D =', D, ' m2/s')
        print('D =', self.D, ' A2/ns')
        
    def Evolve_Dynamics(self,dt=0.001,Nsteps=10,logoutput=False):

        print('---- Evolving the system dynamically ----')

        # dt is ns units
        self.rho_hat[:] = fftn(self.rho) # FT initial condition

        k2 = sum(self.K*self.K,axis=0, dtype=float32)

        # The linear terms of PDE
        L_operator = -self.D*k2
        # The non-linear terms of PDE (with dealising)
        def Noperator_func():
            self.Calculate_weighted_densities()
            self.Calculate_c1()
            gradc1 = ifftn(1.0j*self.K*fftn(self.c1)).real
            gradVext = ifftn(-1.0j*self.K*fftn(self.beta*self.Vext)).real
            rhograd = fftn(self.rho*(gradc1+gradVext))
            return (-1.0j*self.D*self.K*rhograd)
        # auxiliary variables
        explinear_hat = exp(dt*L_operator) # can be calculated once
        Noperator_hat = self.rho_hat.copy()
        # time evolution loop
        for i in range(Nsteps):
            Noperator_hat[:] = Noperator_func() # calculate the non-linear term
            self.rho_hat[:] = (self.rho_hat+dt*Noperator_hat)*explinear_hat # updating in time
            self.rho[:] = ifftn(self.rho_hat).real # IFT to next step

def GrandPotential(lnrho,dft):
    dft.rho[:] = np.exp(lnrho)
    dft.Update_System()
    return dft.Omega

def Gradient(lnrho,dft):
    dft.rho[:] = np.exp(lnrho)
    dft.Update_System()
    return dft.rho*(np.log(dft.rho) - dft.c1 - dft.beta*dft.mu + dft.beta*dft.Vext)

from scipy.optimize import minimize, root

def Equilibrium(dft):
    # res = minimize(GrandPotential, np.log(dft.rho), args=(dft),method='BFGS', jac=Gradient,options={'gtol': 1e-6, 'disp': True})

    sol = root(Gradient, np.log(dft.rho), args=(dft) , method='linearmixing')

    dft.rho[:] = np.exp(sol.x)
    dft.Update_System()