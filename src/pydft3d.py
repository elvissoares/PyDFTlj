import numpy as np
from scipy.special import spherical_jn
import timeit
from eos import LJEOS, BHdiameter
from dcf import DCF3dFT, ljBH3dFT
from scipy.fftpack import fftn, ifftn
try:
    import pyfftw
    from pyfftw.interfaces.scipy_fftpack import fftn, ifftn
    # import multiprocessing
    # pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    pyfftw.config.NUM_THREADS = 4
    # pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)
except ImportError: 
    pass # Rely on scipy.fftpack routines

# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-09-23
# Updated: 2022-10-06

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

" The DFT model for Lennard-Jones fluid on 3d geometries"

" The hard-sphere FMT functional implemented are the following: "
" fmtmethod = RF (Rosenfeld functional) "
"           = WBI (White Bear version I) "
"           = WBII (White Bear version II) "

class DFT3D():
    def __init__(self,gridsize='fine',fmtmethod='WBI',ljmethod='BFD'):
        self.fmtmethod = fmtmethod 
        self.ljmethod = ljmethod 
        self.gridsize = gridsize 
        print('============== The DFT 3D for LJ fluids ==============')
        print('Methods:')
        print('FMT :', self.fmtmethod)
        print('Attractive :', self.ljmethod)
        print('The grid is',self.gridsize)

    def Set_Geometry(self,L):
        if np.isscalar(L): 
            self.L = np.array([L,L,L])
        else: 
            self.L = L
        print('Geometry properties:')
        print('Lx =', self.L[0], ' A')
        print('Ly =', self.L[1], ' A')
        print('Lz =', self.L[2], ' A')
        self.Vol = self.L[0]*self.L[1]*self.L[2]
        print('Vol =',self.Vol, ' A³')

    def Set_FluidProperties(self,sigma=1.0,epsilon=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
        
        print('Fluid properties:')
        print('epsilon/kB =', epsilon, ' K')
        print('sigma =', sigma, ' A')
    
    def Set_Temperature(self,kT):

        print('Temperature =', kT, ' K')
        self.kT = kT
        self.beta = 1/self.kT
        if self.ljmethod == 'MFA' or self.ljmethod == 'BFD' or self.ljmethod == 'CWDA':
            # Baker-Henderson effective diameter
            self.d = np.round(BHdiameter(self.kT,sigma=self.sigma,epsilon=self.epsilon),3)
            print('Baker-Henderson diameter =', self.d, ' A')
        else:
            self.d = self.sigma

        if self.gridsize == 'finest':
            self.delta = 0.01*self.d
        elif self.gridsize == 'fine':
            self.delta = 0.05*self.d
        elif self.gridsize == 'medium':
            self.delta = 0.1*self.d
        elif self.gridsize == 'grained':
            self.delta = 0.2*self.d
        self.x = np.arange(-0.5*self.L[0],0.5*self.L[0],self.delta) + 0.5*self.delta
        self.y = np.arange(-0.5*self.L[1],0.5*self.L[1],self.delta) + 0.5*self.delta
        self.z = np.arange(-0.5*self.L[2],0.5*self.L[2],self.delta) + 0.5*self.delta

        self.N = np.array([self.x.size,self.y.size,self.z.size])

        self.rho = np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.float32)
        self.rho_hat = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)
        if self.ljmethod == 'BFD':
            self.rhodiff_hat = np.empty_like(self.rho_hat)
        elif self.ljmethod == 'CWDA':
            self.rhobar = np.empty_like(self.rho)
            self.uint = np.empty_like(self.rho)
        self.Vext = np.zeros_like(self.rho)

        self.c1 = np.empty_like(self.rho)
        self.c1hs = np.empty_like(self.rho)
        self.c1att = np.empty_like(self.rho)

        self.w3_hat = np.empty_like(self.rho_hat)
        self.w2_hat = np.empty_like(self.rho_hat)
        self.w2vec_hat = np.empty((3,self.N[0],self.N[1],self.N[2]),dtype=np.complex64)

        self.n0 = np.empty_like(self.rho)
        self.n1 = np.empty_like(self.rho)
        self.n3 = np.empty_like(self.rho)
        self.n2 = np.empty_like(self.rho)
        self.n2vec = np.empty((3,self.N[0],self.N[1],self.N[2]),dtype=np.float32)
        self.n1vec = np.empty_like(self.n2vec)
        
        kx = np.fft.fftfreq(self.N[0], d=self.delta)*twopi
        ky = np.fft.fftfreq(self.N[1], d=self.delta)*twopi
        kz = np.fft.fftfreq(self.N[2], d=self.delta)*twopi
        self.kcut = np.array([kx.max(),ky.max(),kz.max()])
        self.Kx,self.Ky,self.Kz = np.meshgrid(kx,ky,kz,indexing ='ij')
        self.K = np.sqrt(self.Kx**2 + self.Ky**2 + self.Kz**2)
        del kx, ky, kz

        # Defining the weight functions
        self.w3_hat[:] = w3FT(self.K,sigma=self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)*translationFT(self.Kx,self.Ky,self.Kz,0.5*self.L)
        self.w2_hat[:] = w2FT(self.K,sigma=self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)*translationFT(self.Kx,self.Ky,self.Kz,0.5*self.L)
        self.w2vec_hat[0] = -1.0j*self.Kx*self.w3_hat
        self.w2vec_hat[1] = -1.0j*self.Ky*self.w3_hat
        self.w2vec_hat[2] = -1.0j*self.Kz*self.w3_hat

    def Set_BulkDensity(self,rhob):

        self.rhob = rhob 

        if self.ljmethod == 'BFD':
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)
            self.flj = ljeos.fatt(self.rhob,self.kT)
            self.mulj = ljeos.muatt(self.rhob,self.kT)
            self.c2_hat = DCF3dFT(self.K,self.rhob,self.kT,sigma=self.sigma,epsilon=self.epsilon)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)*translationFT(self.Kx,self.Ky,self.Kz,0.5*self.L) # to avoid Gibbs phenomenum 
        elif self.ljmethod == 'CWDA':
            self.ulj_hat = ljBH3dFT(self.K,self.epsilon,self.sigma)
            self.amft = -32*np.pi*self.epsilon*self.sigma**3/9
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)
            self.mulj = ljeos.muatt(self.rhob,self.kT)
            self.fcore = lambda rhob: ljeos.fatt(rhob,self.kT) - 0.5*self.amft*rhob**2
            self.mucore = lambda rhob: ljeos.muatt(rhob,self.kT) - self.amft*rhob
            
        self.Calculate_mu()
        print('Bulk Density:',self.rhob)
        print('muid:',self.muid.round(3))
        print('muhs:',self.muhs.round(3))
        print('muatt:',self.muatt.round(3))

    def Set_External_Potential(self,Vext):
        self.Vext[:] = Vext

    def Set_InitialCondition(self):
        self.rho[:] = self.rhob
        mask = self.Vext>=16128*self.epsilon
        self.rho[mask] = 1.e-16
        self.Vext[mask] = 0.0
        self.Update_System()

    def Update_System(self):
        self.Calculate_weighted_densities()
        self.Calculate_c1()
        self.Calculate_Omega()

    def Calculate_weighted_densities(self):
        self.rho_hat[:] = fftn(self.rho)

        self.n3[:] = ifftn(self.rho_hat*self.w3_hat).real
        self.n2[:] = ifftn(self.rho_hat*self.w2_hat).real
        self.n2vec[0] = ifftn(self.rho_hat*self.w2vec_hat[0]).real
        self.n2vec[1] = ifftn(self.rho_hat*self.w2vec_hat[1]).real
        self.n2vec[2] = ifftn(self.rho_hat*self.w2vec_hat[2]).real
        self.n1vec[0] = self.n2vec[0]/(twopi*self.d)
        self.n1vec[1] = self.n2vec[1]/(twopi*self.d)
        self.n1vec[2] = self.n2vec[2]/(twopi*self.d)

        self.n0[:] = self.n2/(np.pi*self.d**2)
        self.n1[:] = self.n2/(twopi*self.d)
        self.oneminusn3 = 1-self.n3

        if self.fmtmethod == 'RF': 
            self.phi2 = 1.0
            self.dphi2dn3 = 0.0
            self.phi3 = 1.0
            self.dphi3dn3 = 0.0
        if self.fmtmethod == 'WBI': 
            self.phi2 = 1.0
            self.dphi2dn3 = 0.0
            self.phi3 = phi1func(self.n3)
            self.dphi3dn3 = dphi1dnfunc(self.n3)
        elif self.fmtmethod == 'WBII': 
            self.phi2 = phi2func(self.n3)
            self.dphi2dn3 = dphi2dnfunc(self.n3)
            self.phi3 = phi3func(self.n3)
            self.dphi3dn3 = dphi3dnfunc(self.n3)
        
        if self.ljmethod == 'BFD':
            self.rhodiff_hat[:] = fftn(self.rho-self.rhob)
        elif self.ljmethod == 'CWDA':
            self.rhobar[:] = ifftn(self.rho_hat*self.w3_hat/(np.pi*self.d**3/6)).real
            self.uint[:] = ifftn(self.rho_hat*self.ulj_hat).real

    def Calculate_Free_energy(self):
        self.Fid = self.kT*np.sum(self.rho*(np.log(self.rho)-1.0))*self.delta**3

        phi = -self.n0*np.log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2])) + (self.phi3/(24*np.pi*self.oneminusn3**2))*(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))
        
        self.Fhs = self.kT*np.sum(phi)*self.delta**3

        if self.ljmethod == 'BFD':
            phi[:] = self.flj + self.mulj*(self.rho-self.rhob) -self.kT*0.5*(self.rho-self.rhob)*ifftn(self.rhodiff_hat*self.c2_hat).real
            self.Flj = np.sum(phi)*self.delta**3
        elif self.ljmethod == 'CWDA':
            phi[:] = 0.5*self.rho*self.uint + self.fcore(self.rhobar)
            self.Flj = np.sum(phi)*self.delta**3
        else:
            self.Flj = 0.0

        self.Fexc =  self.Fhs + self.Flj
        self.F = self.Fid + self.Fexc

    def Calculate_Omega(self):
        self.Calculate_Free_energy()
        self.Omega = self.F + np.sum((self.Vext-self.mu)*self.rho)*self.delta**3

    def Calculate_c1(self):
        dPhidn0 = fftn(-np.log(self.oneminusn3 ))
        dPhidn1 = fftn(self.n2*self.phi2/self.oneminusn3 )
        dPhidn2 = fftn(self.n1*self.phi2/self.oneminusn3  + (3*self.n2*self.n2-3*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))*self.phi3/(24*np.pi*self.oneminusn3**2) )

        dPhidn3 = fftn(self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2]))*(self.dphi2dn3 + self.phi2/self.oneminusn3)/self.oneminusn3 + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/(24*np.pi*self.oneminusn3**2) ) 

        dPhidn1vec0 = fftn( -self.n2vec[0]*self.phi2/self.oneminusn3 )
        dPhidn1vec1 = fftn( -self.n2vec[1]*self.phi2/self.oneminusn3 )
        dPhidn1vec2 = fftn( -self.n2vec[2]*self.phi2/self.oneminusn3 )
        dPhidn2vec0 = fftn( -self.n1vec[0]*self.phi2/self.oneminusn3 + (- 6*self.n2*self.n2vec[0])*self.phi3/(24*np.pi*self.oneminusn3**2))
        dPhidn2vec1 = fftn(-self.n1vec[1]*self.phi2/self.oneminusn3 + (- 6*self.n2*self.n2vec[1])*self.phi3/(24*np.pi*self.oneminusn3**2))
        dPhidn2vec2 = fftn(-self.n1vec[2]*self.phi2/self.oneminusn3 +(-6*self.n2*self.n2vec[2])*self.phi3/(24*np.pi*self.oneminusn3**2))

        c1_hat = -(dPhidn2 + dPhidn1/(twopi*self.d) + dPhidn0/(np.pi*self.d**2))*self.w2_hat
        c1_hat[:] -= dPhidn3*self.w3_hat
        c1_hat[:] += (dPhidn2vec0+dPhidn1vec0/(twopi*self.d))*self.w2vec_hat[0] +(dPhidn2vec1+dPhidn1vec1/(twopi*self.d))*self.w2vec_hat[1] + (dPhidn2vec2+dPhidn1vec2/(twopi*self.d))*self.w2vec_hat[2]

        del dPhidn0,dPhidn1,dPhidn2,dPhidn3,dPhidn1vec0,dPhidn1vec1,dPhidn1vec2,dPhidn2vec0,dPhidn2vec1,dPhidn2vec2

        self.c1hs[:] = ifftn(c1_hat).real

        if self.ljmethod == 'BFD':
            self.c1att[:] = -self.beta*self.mulj
            self.c1att[:] += ifftn(self.rhodiff_hat*self.c2_hat).real
        elif self.ljmethod == 'CWDA':
            self.c1att[:] = -self.beta*self.uint -self.beta*ifftn(self.mucore(self.rhobar)**self.w3_hat/(np.pi*self.d**3/6)).real
        else:
            self.c1att[:] = 0.0

        self.c1[:] = self.c1hs + self.c1att

    def Calculate_mu(self):
        self.muid = self.kT*np.log(self.rhob)

        n3 = self.rhob*np.pi*self.d**3/6
        n2 = self.rhob*np.pi*self.d**2
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
            phi3 = phi1func(n3)
            dphi3dn3 = dphi1dnfunc(n3)
        elif self.fmtmethod == 'WBII': 
            phi2 = phi2func(n3)
            dphi2dn3 = dphi2dnfunc(n3)
            phi3 = phi3func(n3)
            dphi3dn3 = dphi3dnfunc(n3)

        dPhidn0 = -np.log(1-n3)
        dPhidn1 = n2*phi2/(1-n3)
        dPhidn2 = n1*phi2/(1-n3) + (3*n2**2)*phi3/(24*np.pi*(1-n3)**2)
        dPhidn3 = n0/(1-n3) +(n1*n2)*(dphi2dn3 + phi2/(1-n3))/(1-n3) + (n2**3)*(dphi3dn3+2*phi3/(1-n3))/(24*np.pi*(1-n3)**2)

        self.muhs = self.kT*(dPhidn0+dPhidn1*self.d/2+dPhidn2*np.pi*self.d**2+dPhidn3*np.pi*self.d**3/6)

        if self.ljmethod == 'BFD' or self.ljmethod == 'CWDA':
            self.muatt = self.mulj
        else:
            self.muatt = 0.0

        self.mu = self.muid + self.muhs + self.muatt

    def Calculate_Equilibrium(self,alpha0=0.19,dt=0.02,rtol=1e-3,atol=1e-6,logoutput=False):

        print('---- Obtaining the thermodynamic equilibrium ----')

        # Fire algorithm
        Ndelay = 20
        Nmax = 10000
        finc = 1.1
        fdec = 0.5
        fa = 0.99
        Nnegmax = 2000
        dtmax = 10*dt
        dtmin = 0.02*dt
        alpha = alpha0
        Npos = 0
        Nneg = 0

        starttime = timeit.default_timer()

        lnrho = np.log(self.rho)
        V = np.zeros_like(self.rho)
        F = -self.rho*(self.kT*lnrho - self.kT*self.c1 - self.mu + self.Vext)*self.delta**3
        error0 = max(np.abs(F.min()),F.max())

        for i in range(Nmax):

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
                self.rho[:] = np.exp(lnrho)
                self.Update_System()

            V[:] += 0.5*dt*F
            V[:] = (1-alpha)*V + alpha*F*np.linalg.norm(V)/np.linalg.norm(F)
            lnrho[:] += dt*V
            self.rho[:] = np.exp(lnrho)
            self.Update_System()
            F = -self.rho*(self.kT*lnrho - self.kT*self.c1 - self.mu + self.Vext)*self.delta**3
            V[:] += 0.5*dt*F

            error = max(np.abs(F.min()),F.max())
            if error/error0 < rtol and error < atol: break
            if logoutput: print(i,self.Omega,error)
        self.Niter = i

        del V, F  

        print("Time to achieve equilibrium:", timeit.default_timer() - starttime, 'sec')
        print('Number of iterations:', self.Niter)
        print('error:', error)
        print('---- Equilibrium quantities ----')
        print('Fid =',self.Fid)
        print('Fexc =',self.Fexc)
        print('Omega =',self.Omega)
        print('Nbulk =',self.rhob*self.Vol)
        print('Ntot =',self.rho.sum()*self.delta**3)
        print('================================')