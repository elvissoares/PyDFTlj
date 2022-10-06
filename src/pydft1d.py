import numpy as np
import timeit
from eos import LJEOS, BHdiameter
from dcf import DCF1d, ljWCA1d, ljBH1d
from scipy.ndimage import convolve1d
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-06-09
# Updated: 2022-10-06

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

def convolve1dplanar(rho,w,z):
    return convolve1d(rho,w, mode='nearest')

def convolve1dspherical(rho,w,r):
    return convolve1d(rho*r,w, mode='nearest')/r

def integrate1dplanar(f,z,dz):
    return np.sum(f)*dz

def integrate1dspherical(f,r,dr):
    return np.sum(f*4*np.pi*r**2)*dr

def Vsteele(z,sigmaw,epsw,Delta):
    return epsw*(0.4*(sigmaw/z)**10-(sigmaw/z)**4-sigmaw**4/(3*Delta*(z+0.61*Delta)**3))

" The DFT model for Lennard-jones fluid on 1d geometries"

" The hard-sphere FMT functional implemented are the following: "
" fmtmethod = RF (Rosenfeld functional) "
"           = WBI (White Bear version I) "
"           = WBII (White Bear version II) "

class DFT1D():
    def __init__(self,fmtmethod='WBI',ljmethod='BFD',geometry='Planar'):
        self.geometry = geometry
        self.fmtmethod = fmtmethod 
        self.ljmethod = ljmethod 
        print('============== The DFT 1D for LJ fluids ==============')
        print('FMT method:', self.fmtmethod)
        print('LJ method:', self.ljmethod)
        print('Geometry:', self.geometry)

    def Set_Geometry(self,L):
        self.L = L
        print('Geometry properties:')
        print('L =', self.L, ' A')

    def Set_FluidProperties(self,sigma=1.0,epsilon=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
        print('Fluid properties:')
        print('epsilon/kB =', self.epsilon, ' K')
        print('sigma =', self.sigma, ' A')

    def Set_Temperature(self,kT):

        print('Temperature =', kT, ' K')
        self.kT = kT
        self.beta = 1/self.kT
        if self.ljmethod == 'MFA' or self.ljmethod == 'BFD' or self.ljmethod == 'CWDA':
            self.d = np.round(BHdiameter(self.kT,sigma=self.sigma,epsilon=self.epsilon),4)
            print('Baker-Henderson diameter =', self.d, ' A')
        else:
            self.d = self.sigma

        self.delta = 0.01*self.d
        self.z = np.arange(0.0,self.L,self.delta)+0.5*self.delta
        if self.geometry == 'Planar':
            self.convolve = convolve1dplanar
            self.integrate = integrate1dplanar
        elif self.geometry == 'Spherical':
            self.convolve = convolve1dspherical
            self.integrate = integrate1dspherical
        self.N = self.z.size

        self.rho = np.empty(self.N,dtype=np.float32)
        self.Vext = np.zeros_like(self.rho)

        self.n0 = np.empty_like(self.rho)
        self.n1 = np.empty_like(self.rho)
        self.n3 = np.empty_like(self.rho)
        self.n2 = np.empty_like(self.rho)
        self.n2vec = np.empty_like(self.rho)
        self.n1vec = np.empty_like(self.rho)

        self.c1hs = np.empty_like(self.rho)
        self.c1lj = np.empty_like(self.rho)
        self.c1 = np.empty_like(self.rho)

        x = np.arange(-0.5*self.d,0.5*self.d,self.delta)+0.5*self.delta
        self.w3 = np.pi*((0.5*self.d)**2-x**2)
        self.w2 = self.d*np.pi*np.ones_like(x)
        self.w2vec = twopi*x
        if self.ljmethod == 'CWDA':
            # x = np.arange(-self.d,self.d,self.delta)+0.5*self.delta
            # self.w = np.pi*((self.d)**2-x**2)/(4*np.pi*self.d**3/3)
            self.w = self.w3/(np.pi*self.d**3/6)

    def Set_BulkDensity(self,rhob):
        print('---- Setting bulk quantities ----')
        self.rhob = rhob

        if self.ljmethod == 'BFD':
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)
            self.flj = ljeos.fatt(self.rhob,self.kT)
            self.mulj = ljeos.muatt(self.rhob,self.kT)
            self.rc = 5.0*self.d # cutoff radius
            x = np.arange(-self.rc,self.rc,self.delta)+0.5*self.delta
            self.c2lj = DCF1d(x,self.rhob,self.kT,sigma=self.sigma,epsilon=self.epsilon)
        elif self.ljmethod == 'MFA':
            self.rc = 5.0*self.d # cutoff radius
            x = np.arange(-self.rc,self.rc,self.delta)+0.5*self.delta
            self.ulj = ljWCA1d(x,self.epsilon,self.sigma)
            self.amft =-32*np.sqrt(2)*np.pi*self.epsilon*self.sigma**3/9
        elif self.ljmethod == 'CWDA':
            self.rc = 5.0*self.d # cutoff radius
            x = np.arange(-self.rc,self.rc,self.delta)+0.5*self.delta
            self.ulj = ljBH1d(x,self.epsilon,self.sigma)
            self.amft = -32*np.pi*self.epsilon*self.sigma**3/9
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)
            self.mulj = ljeos.muatt(self.rhob,self.kT)
            self.fcore = lambda rhob: ljeos.fatt(rhob,self.kT) - 0.5*self.amft*rhob**2
            self.mucore = lambda rhob: ljeos.muatt(rhob,self.kT) - self.amft*rhob

        self.Calculate_mu()
        print('Bulk Density:',self.rhob)
        print('muid:',round(self.muid,3))
        print('muhs:',round(self.muhs,3))
        print('muatt:',round(self.muatt,3))
    
    def Set_External_Potential(self,extpotmodel='hardwall',params='None'):
        print('---- Setting external potential ----')
        self.extpotmodel = extpotmodel
        self.params = params
        if self.extpotmodel  == 'hardwall' or self.extpotmodel  == 'hardpore' or self.extpotmodel  == 'hardsphere':
            self.Vext[:] = 0.0
        elif self.extpotmodel  == 'steele' or self.extpotmodel  == 'Steele':
            sigmaw, epsw, Delta = params
            self.Vext[:] = Vsteele(self.z,sigmaw,epsw,Delta)+Vsteele(self.L-self.z,sigmaw,epsw,Delta)
        print('External Potential model is:',self.extpotmodel)

    def Set_InitialCondition(self):
        nsig = int(0.5*self.d/self.delta)
        n2sig = int(self.d/self.delta)
        if self.extpotmodel  == 'hardwall':
            self.rho[:] = self.rhob
            self.rho[:nsig] = 1.0e-16
        elif self.extpotmodel  == 'hardpore':
            self.rho[:] = self.rhob
            self.rho[:nsig] = 1.0e-16
            self.rho[-nsig:] = 1.0e-16
        elif self.extpotmodel  == 'hardsphere':
            self.rho[:] = self.rhob
            self.rho[:n2sig] = 1.0e-16
        elif self.extpotmodel  == 'steele' or self.extpotmodel == 'Steele':
            self.rho[:] = self.rhob
            self.rho[self.Vext>16128*self.epsilon] = 1.0e-16
            self.Vext[self.Vext>16128*self.epsilon] = 0.0
        self.Update_System()

    def Update_System(self):
        self.Calculate_weighted_densities()
        self.Calculate_c1()
        self.Calculate_Omega()

    def Calculate_weighted_densities(self):
        self.n3[:] = self.convolve(self.rho,self.w3,self.z)*self.delta
        self.n2[:] = self.convolve(self.rho,self.w2,self.z)*self.delta
        self.n2vec[:] = self.convolve(self.rho,self.w2vec,self.z)*self.delta
        if self.geometry == 'Spherical':
            self.n2vec[:] += self.n3/self.z
        self.n1vec[:] = self.n2vec/(twopi*self.d)
        self.n0[:] = self.n2/(np.pi*self.d**2)
        self.n1[:] = self.n2/(twopi*self.d)
            
        self.oneminusn3 = 1-self.n3

        if self.fmtmethod == 'RF':
            self.phi2 = 1.0
            self.dphi2dn3 = 0.0
            self.phi3 = 1.0
            self.dphi3dn3 = 0.0
        elif self.fmtmethod == 'WBI': 
            self.phi2 = 1.0
            self.dphi2dn3 = 0.0
            self.phi3 = phi1func(self.n3)
            self.dphi3dn3 = dphi1dnfunc(self.n3)
        elif self.fmtmethod == 'WBII': 
            self.phi2 = phi2func(self.n3)
            self.dphi2dn3 = dphi2dnfunc(self.n3)
            self.phi3 = phi3func(self.n3)
            self.dphi3dn3 = dphi3dnfunc(self.n3)

        if self.ljmethod == 'MFA':
            self.uint = self.convolve(self.rho,self.ulj,self.z)*self.delta
        elif self.ljmethod == 'CWDA':
            self.uint = self.convolve(self.rho,self.ulj,self.z)*self.delta
            self.rhobar = self.convolve(self.rho,self.w,self.z)*self.delta

    def Calculate_Free_energy(self):
        self.Fid = self.kT*self.integrate(self.rho*(np.log(self.rho)-1.0),self.z,self.delta)

        phi = -self.n0*np.log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec*self.n2vec)) + (self.phi3/(24*np.pi*self.oneminusn3**2))*(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec*self.n2vec))

        self.Fhs = self.kT*self.integrate(phi,self.z,self.delta)

        if self.ljmethod == 'BFD':
            phi[:] = self.flj
            phi[:] += self.mulj*(self.rho-self.rhob)
            phi[:] += -self.kT*0.5*(self.rho-self.rhob)*self.convolve(self.rho-self.rhob,self.c2lj,self.z)*self.delta
            self.Flj = self.integrate(phi,self.z,self.delta)
        elif self.ljmethod == 'MFA':
            self.Flj = 0.5*self.integrate(self.rho*self.uint,self.z,self.delta)
        elif self.ljmethod == 'CWDA':
            self.Flj = 0.5*self.integrate(self.rho*self.uint,self.z,self.delta) + self.integrate(self.fcore(self.rhobar),self.z,self.delta)
        else:
            self.Flj = 0.0

        self.Fexc =  self.Fhs + self.Flj
        self.F = self.Fid+self.Fexc

    def Calculate_Omega(self):
        self.Calculate_Free_energy()
        self.Omega = (self.F + self.integrate((self.Vext-self.mu)*self.rho,self.z,self.delta))
        self.Ncalc = self.integrate(self.rho,self.z,self.delta)

    def Calculate_c1(self):
        dPhidn0 = -np.log(self.oneminusn3 )
        dPhidn1 = self.n2*self.phi2/self.oneminusn3
        dPhidn2 = self.n1*self.phi2/self.oneminusn3  + (3*self.n2*self.n2-3*(self.n2vec*self.n2vec))*self.phi3/(24*np.pi*self.oneminusn3**2)

        dPhidn3 = self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec*self.n2vec))*(self.dphi2dn3 + self.phi2/self.oneminusn3)/self.oneminusn3 + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec*self.n2vec))*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/(24*np.pi*self.oneminusn3**2)

        dPhidn1vec0 = -self.n2vec*self.phi2/self.oneminusn3 
        dPhidn2vec0 = -self.n1vec*self.phi2/self.oneminusn3  - self.n2*self.n2vec*self.phi3/(4*np.pi*self.oneminusn3**2)

        self.c1hs[:] = -self.convolve(dPhidn2 + dPhidn1/(twopi*self.d) + dPhidn0/(np.pi*self.d**2),self.w2,self.z)*self.delta - self.convolve(dPhidn3,self.w3,self.z)*self.delta + self.convolve(dPhidn2vec0+dPhidn1vec0/(twopi*self.d),self.w2vec,self.z)*self.delta
        if self.geometry == 'Spherical':
            self.c1hs[:] += -self.convolve(dPhidn2vec0+dPhidn1vec0/(twopi*self.d),self.w3,self.z)*self.delta/self.z

        del dPhidn0,dPhidn1,dPhidn2,dPhidn3,dPhidn1vec0,dPhidn2vec0

        if self.ljmethod == 'BFD':
            self.c1lj[:] = -self.beta*self.mulj + self.convolve((self.rho-self.rhob),self.c2lj,self.z)*self.delta
        elif self.ljmethod == 'MFA':
            self.c1lj[:] = -self.beta*self.uint 
        elif self.ljmethod == 'CWDA':
            self.c1lj[:] = -self.beta*self.uint -self.beta*self.convolve(self.mucore(self.rhobar),self.w,self.z)*self.delta 
        else: 
            self.c1lj[:] = 0.0
        
        self.c1[:] = self.c1hs + self.c1lj

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

        if self.ljmethod == 'BFD' or self.ljmethod == 'CWDA':
            self.muatt = self.mulj
        elif self.ljmethod == 'MFA':
            self.muatt = self.amft*self.rhob
        else:
            self.muatt = 0.0

        self.muexc = self.muhs + self.muatt

        self.mu = self.muid + self.muexc

    def Calculate_Equilibrium(self,alpha0=0.19,dt=0.05,rtol=1e-3,atol=1e-6,logoutput=False):

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
        F = -self.rho*(self.kT*lnrho -self.kT*self.c1 - self.mu+self.Vext)*self.delta

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
            F[:] = -self.rho*(self.kT*lnrho -self.kT*self.c1 - self.mu+self.Vext)*self.delta
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
        print('Fid =',self.Fid.round(4))
        print('Fexc =',self.Fexc.round(4))
        print('Ncalc =',self.Ncalc.round(4))
        print('Nbulk =',self.rhob*self.L)
        print('Omega =',self.Omega.round(4))
        print('================================')