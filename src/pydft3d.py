import numpy as np
from scipy.special import spherical_jn
import timeit
from eos import LJEOS, YKEOS, Sfunc, Lfunc, Qfunc
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
# Updated: 2022-09-26

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

" The DFT model for Lennard-Jones fluid on 3d geometries"

" The hard-sphere FMT functional implemented are the following: "
" fmtmethod = RF (Rosenfeld functional) "
"           = WBI (White Bear version I) "
"           = WBII (White Bear version II) "

class DFT3D():
    def __init__(self,gridsize='fine',fmtmethod='WBI',ljmethod='FMSA-MBWR'):
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
        if self.ljmethod == 'FMSA' or self.ljmethod == 'FMSA-MBWR':
            # Baker-Henderson effective diameter
            kTstar = kT/self.epsilon
            self.d = np.round(self.sigma*(1+0.2977*kTstar)/(1+0.33163*kTstar+1.0477e-3*kTstar**2),3)
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
        self.rhodiff_hat = np.empty_like(self.rho_hat)
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
        # Two Yukawa parameters of LJ direct correlation function
        l = np.array([2.64279,14.9677])*self.d/self.sigma
        eps = 1.94728*self.epsilon*(self.sigma/self.d)*np.array([1,-1])*np.exp(l*(self.sigma/self.d-1))
        # l = np.array([2.9637,14.0167])*self.d/self.sigma
        # eps = 2.1714*self.epsilon*(self.sigma/self.d)*np.array([1,-1])*np.exp(l*(self.sigma/self.d-1))

        if self.ljmethod == 'FMSA':
            ljeos = YKEOS(sigma=self.d,epsilon=eps,l=l)
            self.flj = ljeos.f(self.rhob,self.kT)
            self.mulj = ljeos.mu(self.rhob,self.kT)

        elif self.ljmethod == 'FMSA-MBWR':
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon,method='MBWR')
            self.flj = ljeos.fatt(self.rhob,self.kT)
            self.mulj = ljeos.muatt(self.rhob,self.kT)
        
        if self.ljmethod == 'FMSA' or self.ljmethod == 'FMSA-MBWR':
            self.rc = 5.0*self.d # cutoff radius        

            eta = np.pi*self.rhob*self.d**3/6
            denom = ((1-eta)**4*l**6*Qfunc(l,eta)**2)
            A0 = -24*eta*Sfunc(l,eta)*Lfunc(l,eta)/denom
            A1 = 24*eta*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
            A2 = -12*eta*(Sfunc(l,eta)*Lfunc(l,eta)*l**2-(1-eta)**2*(1+0.5*eta)*l**6)/denom
            A4 = 0.5*eta*A1
            C1 = -Sfunc(l,eta)**2/denom
            C2 = -144*eta**2*Lfunc(l,eta)**2/denom

            self.c2_hat = np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)

            for i in range(eps.size):
                self.c2_hat[:] += -self.beta*eps[i]*YKcutoffFT(self.K,l[i],rc=self.rc/self.d,sigma=self.d) + self.beta*eps[i]*(A0[i]*A0funcFT(self.K,sigma=self.d)+A1[i]*A1funcFT(self.K,sigma=self.d)+A2[i]*A2funcFT(self.K,sigma=self.d)+A4[i]*A4funcFT(self.K,sigma=self.d)+C1[i]*YKcoreFT(self.K,l[i],sigma=self.d)+C2[i]*YKcoreFT(self.K,-l[i],sigma=self.d))

            self.c2_hat[:] *= sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)*translationFT(self.Kx,self.Ky,self.Kz,0.5*self.L) # to avoid Gibbs phenomenum 
            
            del A0, A1, A2, A4, C1, C2, eta, denom

        self.Calculate_mu()
        print('Bulk Density:',self.rhob)
        print('muid:',self.muid.round(3))
        print('muhs:',self.muhs.round(3))
        print('muatt:',self.muatt.round(3))

    def Set_External_Potential(self,Vext):
        self.Vext[:] = Vext

    def Set_InitialCondition(self):
        self.rho[:] = self.rhob
        mask = self.Vext>16128
        self.rho[mask] = 1.e-16
        self.Vext[mask] = 0.0
        self.Update_System()

    def Update_System(self):
        self.Calculate_weighted_densities()
        self.Calculate_c1()
        self.Calculate_Omega()

    def Calculate_weighted_densities(self):
        self.rho_hat[:] = fftn(self.rho)
        self.rhodiff_hat[:] = fftn(self.rho-self.rhob)

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

        if self.fmtmethod == 'RF' or self.fmtmethod == 'WBI': 
            self.phi2 = 1.0
            self.dphi2dn3 = 0.0
        elif self.fmtmethod == 'WBII': 
            self.phi2 = phi2func(self.n3)
            self.dphi2dn3 = dphi2dnfunc(self.n3)

        if self.fmtmethod == 'WBI': 
            self.phi3 = phi1func(self.n3)
            self.dphi3dn3 = dphi1dnfunc(self.n3)
        elif self.fmtmethod == 'WBII': 
            self.phi3 = phi3func(self.n3)
            self.dphi3dn3 = dphi3dnfunc(self.n3)
        else: 
            self.phi3 = 1.0
            self.dphi3dn3 = 0.0

    def Calculate_Free_energy(self):
        self.Fid = self.kT*np.sum(self.rho*(np.log(self.rho)-1.0))*self.delta**3

        phi = -self.n0*np.log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2])) + (self.phi3/(24*np.pi*self.oneminusn3**2))*(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))
        
        self.Fhs = self.kT*np.sum(phi)*self.delta**3

        if self.ljmethod == 'FMSA' or self.ljmethod == 'FMSA-MBWR':
            phi[:] = self.flj
            phi[:] += self.mulj*(self.rho-self.rhob)
            phi[:] += -self.kT*0.5*(self.rho-self.rhob)*ifftn(self.rhodiff_hat*self.c2_hat).real
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
        c1_hat -= dPhidn3*self.w3_hat
        c1_hat += (dPhidn2vec0+dPhidn1vec0/(twopi*self.d))*self.w2vec_hat[0] +(dPhidn2vec1+dPhidn1vec1/(twopi*self.d))*self.w2vec_hat[1] + (dPhidn2vec2+dPhidn1vec2/(twopi*self.d))*self.w2vec_hat[2]

        del dPhidn0,dPhidn1,dPhidn2,dPhidn3,dPhidn1vec0,dPhidn1vec1,dPhidn1vec2,dPhidn2vec0,dPhidn2vec1,dPhidn2vec2

        self.c1hs[:] = ifftn(c1_hat).real

        if self.ljmethod == 'FMSA' or self.ljmethod == 'FMSA-MBWR':
            self.c1att[:] = -self.beta*self.mulj
            self.c1att[:] += ifftn(self.rhodiff_hat*self.c2_hat).real
        else:
            self.c1att[:] = 0.0

        self.c1[:] = self.c1hs + self.c1att

    def Calculate_mu(self):
        self.muid = self.kT*np.log(self.rhob)

        n3 = self.rhob*np.pi*self.d**3/6
        n2 = self.rhob*np.pi*self.d**2
        n1 = self.rhob*self.d/2
        n0 = self.rhob

        if self.fmtmethod == 'RF' or self.fmtmethod == 'WBI': 
            phi2 = 1.0
            dphi2dn3 = 0.0
        elif self.fmtmethod == 'WBII': 
            phi2 = phi2func(n3)
            dphi2dn3 = dphi2dnfunc(n3)

        if self.fmtmethod == 'WBI': 
            phi3 = phi1func(n3)
            dphi3dn3 = dphi1dnfunc(n3)
        elif self.fmtmethod == 'WBII': 
            phi3 = phi3func(n3)
            dphi3dn3 = dphi3dnfunc(n3)
        else: 
            phi3 = 1.0
            dphi3dn3 = 0.0

        dPhidn0 = -np.log(1-n3)
        dPhidn1 = n2*phi2/(1-n3)
        dPhidn2 = n1*phi2/(1-n3) + (3*n2**2)*phi3/(24*np.pi*(1-n3)**2)
        dPhidn3 = n0/(1-n3) +(n1*n2)*(dphi2dn3 + phi2/(1-n3))/(1-n3) + (n2**3)*(dphi3dn3+2*phi3/(1-n3))/(24*np.pi*(1-n3)**2)

        self.muhs = self.kT*(dPhidn0+dPhidn1*self.d/2+dPhidn2*np.pi*self.d**2+dPhidn3*np.pi*self.d**3/6)

        if self.ljmethod == 'FMSA' or self.ljmethod == 'FMSA-MBWR':
            self.muatt = self.mulj
        else:
            self.muatt = 0.0

        self.mu = self.muid + self.muhs + self.muatt

    def Calculate_Equilibrium(self,alpha0=0.62,dt=20.0,rtol=1e-7,atol=1e-8,logoutput=False):

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
        F = -self.rho*(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)*self.delta**3/self.Vol
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
            F[:] = -self.rho*(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)*self.delta**3/self.Vol
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