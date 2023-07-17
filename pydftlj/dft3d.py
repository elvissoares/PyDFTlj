import numpy as np
from numpy import pi, exp, log, sqrt, round, linspace, isscalar, array, meshgrid
import timeit
from tqdm import tqdm
from .eos import LJEOS, BHdiameter
from .fmtaux import sigmaLancsozFT,translationFT, w3FT, w2FT,phi1func,dphi1dnfunc,phi2func,dphi2dnfunc, phi3func, dphi3dnfunc, phi3funcWBI, dphi3dnfuncWBI, phi2funcWBII, dphi2dnfuncWBII, phi3funcWBII, dphi3dnfuncWBII
from .dcf import  ljBH3dFT
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2023-04-27
# Updated: 2023-04-27

" The DFT model for Lennard-Jones fluid on 3d geometries"

" The hard-sphere FMT functional implemented are the following: "
" fmtmethod = RF (Rosenfeld functional) "
"           = WBI (White Bear version I) "
"           = WBII (White Bear version II) "

class dft3d():
    def __init__(self,gridsize,fmtmethod='WBI',ljmethod='MMFA',padding=True,padding_value='zero'):
        self.fmtmethod = fmtmethod
        self.ljmethod = ljmethod 
        self.Ncell = gridsize 
        self.Ncelltot = self.Ncell[0]*self.Ncell[1]*self.Ncell[2]
        self.padding = padding
        self.padding_value = padding_value
        self.psi = 1.3862

    def Set_FluidProperties(self,sigma=1.0,epsilon=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
    
    def Set_Geometry(self,Lcell):
        self.Lcell = Lcell
        if self.Ncell[0] == 0 and self.Ncell[1] == 0:
            if self.padding: 
                if self.ljmethod == 'WDA':
                    self.Lgrid = Lcell + 4*self.psi*self.sigma
                if self.ljmethod == 'MFA' or self.ljmethod == 'MMFA':
                    self.Lgrid = Lcell + 8*self.sigma
                else:
                    self.Lgrid = Lcell + self.sigma
            else: self.Lgrid = Lcell
            self.delta = Lcell/self.Ncell[2]
            self.x = np.array([0.0])
            self.y = np.array([0.0])
            self.z = np.arange(-0.5*self.Lgrid,0.5*self.Lgrid,self.delta)
            self.X,self.Y,self.Z = meshgrid(self.x,self.y,self.z,indexing ='ij')
            self.dV = self.delta
            self.Vol = self.Lgrid

            self.Ngrid = np.array([self.x.size,self.y.size,self.z.size])
            self.Ngridtot = self.Ngrid[0]*self.Ngrid[1]*self.Ngrid[2]
            if self.padding: 
                self.Npad = (self.Ngrid - self.Ncell)//2
                self.Npad[0] = self.Npad[1] = 0

            kx = np.array([0.0])
            ky = np.array([0.0])
            kz = np.fft.fftfreq(self.Ngrid[2], d=self.delta)*2*pi
            self.kcut = array([1.0,1.0,kz.max()])
            self.Kx,self.Ky,self.Kz = meshgrid(kx,ky,kz,indexing ='ij')
            self.K = sqrt(self.Kx**2 + self.Ky**2 + self.Kz**2)
            del kx, ky, kz

        else:
            if self.padding: 
                self.Lgrid = Lcell + 4*self.psi*self.sigma
            else: self.Lgrid = Lcell
            self.delta = Lcell/self.Ncell
            self.x = np.arange(-0.5*self.Lgrid[0],0.5*self.Lgrid[0],self.delta[0])
            self.y = np.arange(-0.5*self.Lgrid[1],0.5*self.Lgrid[1],self.delta[1])
            self.z = np.arange(-0.5*self.Lgrid[2],0.5*self.Lgrid[2],self.delta[2])
            self.X,self.Y,self.Z = meshgrid(self.x,self.y,self.z,indexing ='ij')
            self.dV = self.delta[0]*self.delta[1]*self.delta[2]
            self.Vol = self.Lgrid[0]*self.Lgrid[1]*self.Lgrid[2]

            self.Ngrid = np.array([self.x.size,self.y.size,self.z.size])
            self.Ngridtot = self.Ngrid[0]*self.Ngrid[1]*self.Ngrid[2]
            if self.padding: self.Npad = (self.Ngrid - self.Ncell)//2

            kx = np.fft.fftfreq(self.Ngrid[0], d=self.delta[0])*2*pi
            ky = np.fft.fftfreq(self.Ngrid[1], d=self.delta[1])*2*pi
            kz = np.fft.fftfreq(self.Ngrid[2], d=self.delta[2])*2*pi
            self.kcut = array([kx.max(),ky.max(),kz.max()])
            if self.kcut[0] == 0.0: self.kcut[0] = 1.0
            if self.kcut[1] == 0.0: self.kcut[1] = 1.0
            if self.kcut[2] == 0.0: self.kcut[2] = 1.0
            self.Kx,self.Ky,self.Kz = meshgrid(kx,ky,kz,indexing ='ij')
            self.K = sqrt(self.Kx**2 + self.Ky**2 + self.Kz**2)
            del kx, ky, kz

        # creating arrays
        self.rho = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32)

        self.rho_hat = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.complex64, device=device)

        self.rhobar = torch.empty_like(self.rho)
        self.mu_hat = torch.empty_like(self.rho_hat)
        
        self.c1 = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        self.c1hs = torch.zeros_like(self.c1)
        self.c1att = torch.zeros_like(self.c1)

        self.n0 = torch.empty_like(self.rho)
        self.n1 = torch.empty_like(self.rho)
        self.n3 = torch.empty_like(self.rho)
        self.n2 = torch.empty_like(self.rho)
        self.n2vec = torch.empty((3,self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32)
        self.n1vec = torch.empty_like(self.n2vec)
        

    def Set_Temperature(self,kT):
        self.kT = kT
        self.beta = 1/self.kT
        if self.ljmethod == 'WDA' or self.ljmethod == 'MMFA' or self.ljmethod == 'MFA':
            self.d = round(BHdiameter(self.kT,sigma=self.sigma,epsilon=self.epsilon),3)
        else: self.d = self.sigma

        # Defining the weight functions
        self.w3_hat = torch.tensor(w3FT(self.K,sigma=self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut),dtype=torch.complex64, device=device)
        self.w2_hat = torch.tensor(w2FT(self.K,sigma=self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut),dtype=torch.complex64, device=device)
        self.w2vec_hat = torch.zeros((3,self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.complex64, device=device)
        self.w2vec_hat[0] = self.w3_hat*torch.tensor(-1.0j*self.Kx,dtype=torch.complex64, device=device)
        self.w2vec_hat[1] = self.w3_hat*torch.tensor(-1.0j*self.Ky,dtype=torch.complex64, device=device)
        self.w2vec_hat[2] = self.w3_hat*torch.tensor(-1.0j*self.Kz,dtype=torch.complex64, device=device)

        if self.ljmethod == 'WDA':
            self.w_hat = torch.tensor(w3FT(self.K,sigma=2*self.psi*self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)/(pi*(2*self.psi*self.d)**3/6),dtype=torch.complex64, device=device)
        elif self.ljmethod == 'MMFA':
            self.amft = -32*pi*self.epsilon*self.sigma**3/9
            self.fcore = lambda rr: self.fdisp(rr) - 0.5*self.amft*rr**2
            self.mucore = lambda rr: self.mudisp(rr) - self.amft*rr
            self.uint = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32)
            self.w_hat = torch.tensor(w3FT(self.K,sigma=2*self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)/(4*pi*(self.d)**3/3),dtype=torch.complex64, device=device)
            self.ulj_hat = torch.tensor(ljBH3dFT(self.K,self.kT,self.sigma,self.epsilon)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut),dtype=torch.complex64, device=device) # to avoid Gibbs phenomenum
        elif self.ljmethod == 'MFA':
            self.amft = -32*pi*self.epsilon*self.sigma**3/9
            self.uint = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32)
            self.ulj_hat = torch.tensor(ljBH3dFT(self.K,self.kT,self.sigma,self.epsilon)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut),dtype=torch.complex64, device=device) # to avoid Gibbs phenomenum

    def Set_BulkDensity(self,rhob):
        self.rhob = rhob 

        if self.ljmethod == 'WDA' or self.ljmethod == 'MMFA':
            ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)
            self.fdisp = lambda rr: ljeos.fatt(rr,self.kT)
            self.mudisp = lambda rr: ljeos.muatt(rr,self.kT) 

        if self.padding_value == 'zero': self.pad_constant = 0.0
        elif self.padding_value == 'bulk': self.pad_constant = self.rhob
            
        self.Calculate_mu()
        
    def Set_External_Potential(self,Vext):

        self.Vext = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)

        self.Vext[:] = torch.tensor(Vext)
        self.mask = (self.Vext<16128)
        self.Vext[self.Vext>=16128] = 16128

        if self.padding: 
            if self.Ncell[0] == 0 and self.Ncell[1] == 0:
                if self.padding_value == 'zero':
                    self.Vext[:,:,:self.Npad[2]] = 16128
                    self.Vext[:,:,-self.Npad[2]:] = 16128
                elif self.padding_value == 'bulk':
                    self.Vext[:,:,:self.Npad[2]] = 0.
                    self.Vext[:,:,-self.Npad[2]:] = 0.
            else:
                if self.padding_value == 'zero':
                    self.Vext[:self.Npad[0],:,:] = 16128
                    self.Vext[:,:self.Npad[1],:] = 16128
                    self.Vext[:,:,:self.Npad[2]] = 16128
                    self.Vext[-self.Npad[0]:,:,:] = 16128
                    self.Vext[:,-self.Npad[1]:,:] = 16128
                    self.Vext[:,:,-self.Npad[2]:] = 16128
                elif self.padding_value == 'bulk':
                    self.Vext[:self.Npad[0],:,:] = 0.
                    self.Vext[:,:self.Npad[1],:] = 0.
                    self.Vext[:,:,:self.Npad[2]] = 0.
                    self.Vext[-self.Npad[0]:,:,:] = 0.
                    self.Vext[:,-self.Npad[1]:,:] = 0.
                    self.Vext[:,:,-self.Npad[2]:] = 0.

    def Padding_Density(self):
        if self.padding: 
            if self.Ncell[0] == 0 and self.Ncell[1] == 0:
                self.rho[:,:,:self.Npad[2]] = self.pad_constant
                self.rho[:,:,-self.Npad[2]:] = self.pad_constant
            else:
                self.rho[:self.Npad[0],:,:] = self.pad_constant 
                self.rho[:,:self.Npad[1],:] = self.pad_constant 
                self.rho[:,:,:self.Npad[2]] = self.pad_constant
                self.rho[-self.Npad[0]:,:,:] = self.pad_constant
                self.rho[:,-self.Npad[1]:,:] = self.pad_constant
                self.rho[:,:,-self.Npad[2]:] = self.pad_constant

        self.Update_System()

    def Set_InitialCondition(self):

        self.rho[:] = 0.0
        # self.rho[self.mask] = self.rhob
        self.rho[self.mask] = self.rhob*np.exp(-0.01*self.beta*self.Vext[self.mask].cpu())

        if self.padding: 
            self.Padding_Density()

        self.Update_System()

    def GetSystemInformation(self):
        print('============== The DFT 3D for LJ fluids ==============')
        print('Methods:')
        print('HS method = ',self.fmtmethod)
        print('LJ method = ',self.ljmethod)
        print('The grid is',self.Ngrid)
        print('--- Geometry properties ---')
        print('Lx =', self.Lgrid[0], ' A')
        print('Ly =', self.Lgrid[1], ' A')
        print('Lz =', self.Lgrid[2], ' A')
        print('delta = ', self.delta, ' A')
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
        self.rho_hat[:] = torch.fft.fftn(self.rho.to(device))

        # Unpack the results and assign to self.n 
        self.n3[:] = torch.fft.ifftn(self.rho_hat*self.w3_hat).real.cpu()
        self.n2[:] = torch.fft.ifftn(self.rho_hat*self.w2_hat).real.cpu()
        self.n2vec[0] = torch.fft.ifftn(self.rho_hat*self.w2vec_hat[0]).real.cpu()
        self.n2vec[1] = torch.fft.ifftn(self.rho_hat*self.w2vec_hat[1]).real.cpu()
        self.n2vec[2] = torch.fft.ifftn(self.rho_hat*self.w2vec_hat[2]).real.cpu()

        self.n3[self.n3>=1.0] = 1.0-1e-30 # to avoid Nan on some calculations
        self.xi = (self.n2vec*self.n2vec).sum(dim=0)/((self.n2+1e-30)**2)
        self.xi[self.xi>=1.0] = 1.0-1e-30

        self.n1vec[:] = self.n2vec/(2*pi*self.d)

        self.n0[:] = self.n2/(pi*self.d**2)
        self.n1[:] = self.n2/(2*pi*self.d)
        
        self.phi1 = torch.tensor(phi1func(self.n3.numpy()),dtype=torch.float32)
        self.dphi1dn3 = torch.tensor(dphi1dnfunc(self.n3.numpy()),dtype=torch.float32)
        if self.fmtmethod == 'RF':
            self.phi2 = torch.tensor(phi2func(self.n3.numpy()),dtype=torch.float32)
            self.dphi2dn3 = torch.tensor(dphi2dnfunc(self.n3.numpy()),dtype=torch.float32)
            self.phi3 = torch.tensor(phi3func(self.n3.numpy()),dtype=torch.float32)
            self.dphi3dn3 = torch.tensor(dphi3dnfunc(self.n3.numpy()),dtype=torch.float32)
        elif self.fmtmethod == 'WBI':
            self.phi2 = torch.tensor(phi2func(self.n3.numpy()),dtype=torch.float32)
            self.dphi2dn3 = torch.tensor(dphi2dnfunc(self.n3.numpy()),dtype=torch.float32)
            self.phi3 = torch.tensor(phi3funcWBI(self.n3.numpy()),dtype=torch.float32)
            self.dphi3dn3 = torch.tensor(dphi3dnfuncWBI(self.n3.numpy()),dtype=torch.float32)
        elif self.fmtmethod == 'WBII':
            self.phi2 = torch.tensor(phi2funcWBII(self.n3.numpy()),dtype=torch.float32)
            self.dphi2dn3 = torch.tensor(dphi2dnfuncWBII(self.n3.numpy()),dtype=torch.float32)
            self.phi3 = torch.tensor(phi3funcWBII(self.n3.numpy()),dtype=torch.float32)
            self.dphi3dn3 = torch.tensor(dphi3dnfuncWBII(self.n3.numpy()),dtype=torch.float32)
        
        if self.ljmethod == 'WDA':
            self.rhobar[:] = torch.fft.ifftn(self.rho_hat*self.w_hat).real.cpu()
            self.mu_hat[:] =  torch.fft.fftn(torch.tensor(self.mudisp(self.rhobar.numpy()),dtype=torch.float32, device=device))
        elif self.ljmethod == 'MMFA':
            self.rhobar[:] = torch.fft.ifftn(self.rho_hat*self.w_hat).real.cpu()
            self.uint[:] = torch.fft.ifftn(self.rho_hat*self.ulj_hat).real.cpu()
            self.mu_hat[:] =  torch.fft.fftn(torch.tensor(self.mucore(self.rhobar.numpy()),dtype=torch.float32, device=device))
        elif self.ljmethod == 'MFA':
            self.uint[:] = torch.fft.ifftn(self.rho_hat*self.ulj_hat).real.cpu()

    def Calculate_Free_energy(self):
        self.Fid = self.kT*torch.sum(self.rho*(torch.log(self.rho+1.0e-30)-1.0))*self.dV

        phi = self.n0*self.phi1+self.phi2*(self.n1*self.n2-(self.n1vec*self.n2vec).sum(dim=0)) + self.phi3*self.n2**3*(1-self.xi)**3
        
        self.Fhs = self.kT*torch.sum(phi)*self.dV

        if self.ljmethod == 'WDA':
            phi[:] = torch.tensor(self.fdisp(self.rhobar.numpy()),dtype=torch.float32)
            self.Flj = torch.sum(phi)*self.dV
        elif self.ljmethod == 'MMFA':
            phi[:] = 0.5*self.rho*self.uint + torch.tensor(self.fcore(self.rhobar.numpy()),dtype=torch.float32)
            self.Flj = torch.sum(phi)*self.dV
        elif self.ljmethod == 'MFA':
            phi[:] = 0.5*self.rho*self.uint
            self.Flj = torch.sum(phi)*self.dV
        else:
            self.Flj = 0.0

        del phi

        self.Fexc =  self.Fhs + self.Flj
        self.F = self.Fid + self.Fexc

    def Calculate_Omega(self):
        self.Calculate_Free_energy()
        self.Omega = self.F + torch.sum((self.Vext.cpu()-self.mu)*self.rho)*self.dV

    def Calculate_c1(self):

        c1_hat = -torch.fft.fftn(self.phi1.to(device))/(pi*self.d**2)*self.w2_hat #dPhidn0
        c1_hat[:] += -torch.fft.fftn((self.n2*self.phi2).to(device) )/(2*pi*self.d)*self.w2_hat #dPhidn1
        c1_hat[:] += -torch.fft.fftn((self.n1*self.phi2 + 3*(self.n2**2)*(1+self.xi)*((1-self.xi)**2)*self.phi3).to(device))*self.w2_hat #dPhidn2

        c1_hat[:] += -torch.fft.fftn((self.n0*self.dphi1dn3 +(self.n1*self.n2-(self.n1vec*self.n2vec).sum(dim=0))*self.dphi2dn3 + (self.n2**3*(1-self.xi)**3)*self.dphi3dn3).to(device) )*self.w3_hat #dPhidn3

        c1_hat[:] += torch.fft.fftn( (-self.n2vec[0]*self.phi2).to(device) )/(2*pi*self.d)*self.w2vec_hat[0] #dPhidn1vec0
        c1_hat[:] += torch.fft.fftn( (-self.n2vec[1]*self.phi2).to(device) )/(2*pi*self.d)*self.w2vec_hat[1] #dPhidn1vec1
        c1_hat[:] += torch.fft.fftn( (-self.n2vec[2]*self.phi2).to(device) )/(2*pi*self.d)*self.w2vec_hat[2] #dPhidn1vec2
        c1_hat[:] += torch.fft.fftn( (-self.n1vec[0]*self.phi2 + (- 6*self.n2*self.n2vec[0]*(1-self.xi)**2)*self.phi3).to(device))*self.w2vec_hat[0]
        c1_hat[:] += torch.fft.fftn((-self.n1vec[1]*self.phi2 + (- 6*self.n2*self.n2vec[1]*(1-self.xi)**2)*self.phi3).to(device))*self.w2vec_hat[1]
        c1_hat[:] += torch.fft.fftn((-self.n1vec[2]*self.phi2 +(- 6*self.n2*self.n2vec[2]*(1-self.xi)**2)*self.phi3).to(device))*self.w2vec_hat[2]

        self.c1hs[:] = torch.fft.ifftn(c1_hat).real

        del c1_hat
        torch.cuda.empty_cache()

        if self.ljmethod == 'WDA':
            self.c1att[:] = -self.beta*torch.fft.ifftn(self.mu_hat*self.w_hat).real
        elif self.ljmethod == 'MMFA':
            self.c1att[:] = -self.beta*self.uint.to(device) -self.beta*torch.fft.ifftn(self.mu_hat*self.w_hat).real
        elif self.ljmethod == 'MFA':
            self.c1att[:] = -self.beta*self.uint.to(device)

        self.c1[:] = self.c1hs + self.c1att

    def Calculate_mu(self):
        self.muid = self.kT*log(self.rhob)

        n3 = self.rhob*pi*self.d**3/6
        n2 = self.rhob*pi*self.d**2
        n1 = self.rhob*self.d/2
        n0 = self.rhob

        phi1 = phi1func(n3)
        dphi1dn3 = dphi1dnfunc(n3)
        if self.fmtmethod == 'RF':
            phi2 = phi2func(n3)
            dphi2dn3 = dphi2dnfunc(n3)
            phi3 = phi3func(n3)
            dphi3dn3 = dphi3dnfunc(n3)
        elif self.fmtmethod == 'WBI':
            phi2 = phi2func(n3)
            dphi2dn3 = dphi2dnfunc(n3)
            phi3 = phi3funcWBI(n3)
            dphi3dn3 = dphi3dnfuncWBI(n3)
        elif self.fmtmethod == 'WBII':
            phi2 = phi2funcWBII(n3)
            dphi2dn3 = dphi2dnfuncWBII(n3)
            phi3 = phi3funcWBII(n3)
            dphi3dn3 = dphi3dnfuncWBII(n3)

        dPhidn0 = phi1
        dPhidn1 = n2*phi2
        dPhidn2 = n1*phi2 + (3*n2**2)*phi3
        dPhidn3 = n0*dphi1dn3 +(n1*n2)*dphi2dn3 + (n2**3)*dphi3dn3
        self.muhs = self.kT*(dPhidn0+dPhidn1*self.d/2+dPhidn2*pi*self.d**2+dPhidn3*pi*self.d**3/6)

        if self.ljmethod == 'WDA' or self.ljmethod == 'MMFA':
            self.muatt = self.mudisp(self.rhob)
        elif self.ljmethod == 'MFA':
            self.muatt = self.rhob*self.amft
        else:
            self.muatt = 0.0

        self.muexc = self.muhs + self.muatt
        self.mu = self.muid + self.muexc

    def Calculate_Equilibrium(self,alpha0=0.25,dt=0.1,atol=1e-6,rtol=1e-4,max_iter=9999,method='fire',logoutput=False):

        starttime = timeit.default_timer()

        alpha = alpha0
        self.dt = torch.tensor(dt, device=device)

        lnrho = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        lnrho[:] = torch.log(self.rho+1.0e-30) # to avoid log(0)
        self.Update_System()
        rho0 = self.rho.clone()

        F = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        sk = torch.empty_like(F)
        F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
        sk[:] = atol+rtol*torch.abs(self.rho)
        error = torch.norm(self.rho.to(device)*F/sk)/np.sqrt(self.Ngridtot)

        if logoutput: print(0,self.Omega.numpy(),error.cpu().numpy(),self.dt.cpu().numpy())

        if method == 'picard':
            # Picard algorithm
            
            for i in range(max_iter):                
                lnrho[:] += alpha*F
                self.rho[:] = torch.exp(lnrho).cpu()
                self.Update_System()
                F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
                
                self.Niter = i+1
                sk[:]=atol+rtol*torch.abs(self.rho)
                error = torch.norm(self.rho.to(device)*F/sk)/np.sqrt(self.Ngridtot)
                if logoutput: print(self.Niter,self.Omega.numpy(),error.cpu().numpy())
                if error < 1.0: break

        elif method == 'fire':
            # Fire algorithm
            Ndelay = 20
            Nnegmax = 2000
            dtmax = 10*dt
            dtmin = 0.02*dt
            Npos = 0
            Nneg = 0
            finc = 1.1
            fdec = 0.5
            fa = 0.99
            V = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)

            for i in range(max_iter):

                P = torch.sum(F*V) # dissipated power
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
                    lnrho[:] -= V*0.5*dt
                    V[:] = 0.0
                    self.rho[:] = torch.exp(lnrho).cpu()
                    if self.padding: 
                        self.Padding_Density()
                    self.Update_System()

                V[:] += F*0.5*dt
                V[:] = (1-alpha)*V + alpha*F*torch.norm(V)/torch.norm(F)
                lnrho[:] += dt*V
                self.rho[:] = torch.exp(lnrho).cpu()
                if self.padding: 
                    self.Padding_Density()
                self.Update_System()
                F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
                V[:] += F*0.5*dt

                self.Niter = i+1
                sk[:]=atol+rtol*torch.abs(self.rho)
                error = torch.norm(self.rho.to(device)*F/sk)/np.sqrt(self.Ngridtot)
                if logoutput: print(self.Niter,self.Omega.numpy(),error.cpu().numpy())
                if error < 1.0 and self.Niter> Ndelay: break
                
            del V

        elif method == 'abc-fire':
            # ABC-Fire algorithm https://doi.org/10.1016/j.commatsci.2022.111978
            Ndelay = 20
            Nnegmax = 2000
            dtmax = 10*dt
            dtmin = 0.02*dt
            Npos = 1
            Nneg = 0
            finc = 1.1
            fdec = 0.5
            fa = 0.99
            V = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
            i = 0 

            while i < max_iter:

                P = torch.sum(F*V) # dissipated power
                if (P>0):
                    Npos = Npos + 1
                    if Npos>Ndelay:
                        dt = min(dt*finc,dtmax)
                        alpha = max(1.0e-10,alpha*fa)
                else:
                    Npos = 1
                    Nneg = Nneg + 1
                    if Nneg > Nnegmax: break
                    if i> Ndelay:
                        dt = max(dt*fdec,dtmin)
                        alpha = alpha0
                    lnrho[:] -= V*0.5*dt
                    V[:] = 0.0
                    self.rho[:] = torch.exp(lnrho).cpu()
                    if self.padding: 
                        self.Padding_Density()
                    self.Update_System()

                V[:] += F*0.5*dt
                V[:] = (1/(1-(1-alpha)**Npos))*((1-alpha)*V + alpha*F*torch.norm(V)/torch.norm(F))
                lnrho[:] += dt*V
                self.rho[:] = torch.exp(lnrho).cpu()
                if self.padding: 
                    self.Padding_Density()
                self.Update_System()
                F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
                V[:] += F*0.5*dt

                self.Niter = i+1
                i += 1
                sk[:]=atol+rtol*torch.abs(self.rho)
                error = torch.norm(self.rho.to(device)*F/sk)/np.sqrt(self.Ngridtot)

                if logoutput: print(self.Niter,self.Omega.numpy(),error.cpu().numpy(),'|',alpha,dt)
                if error < 1.0 and self.Niter> Ndelay: break
                if torch.isnan(error):
                    print('DFT::ABC-FIRE: The system is out fo equilibrium!')
                    break
                #     i = 0 
                #     lnrho[:] = torch.log(rho0+1.0e-30) # to avoid log(0)
                #     self.rho[:] = rho0
                #     self.Update_System()
                #     F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
                #     V[:] = 0.0
                #     sk[:] = atol+rtol*torch.abs(self.rho)
                #     error = torch.norm(self.rho.to(device)*F/sk)/np.sqrt(self.Ngridtot)
                #     alpha = alpha0
                #     dt = dt/2.0
                #     dtmax = 10*dt
                #     dtmin = 0.02*dt
                #     Npos = 1
                #     Nneg = 0
                
            del V

        elif method == 'metropolis':

            self.sigma_lnrho = 0.001
            burning_steps = int(max_iter/10)
            self.rho_ensemble = torch.zeros((int(max_iter),self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
            self.Omega_ensemble = torch.zeros(int(max_iter),dtype=torch.float32, device=device)
            lnrho_prev = torch.tensor(lnrho,dtype=torch.float32, device=device)
            Omega_prev = self.Omega
            self.lnrho_hi = 4.0
            self.lnrho_lo = -70.0

            for i in tqdm(range(int(max_iter))):

                lnrho_prop = torch.normal(mean=lnrho_prev,std=self.sigma_lnrho).to(device)
                # reflect proposals outside of bounds
                too_hi = lnrho_prop > self.lnrho_hi
                too_lo = lnrho_prop < self.lnrho_lo

                lnrho_prop[too_hi] = 2*self.lnrho_hi - lnrho_prop[too_hi]
                lnrho_prop[too_lo] = 2*self.lnrho_lo - lnrho_prop[too_lo]

                self.rho[:] = torch.exp(lnrho_prop).cpu()
                if self.padding: 
                    self.Padding_Density()
                self.Update_System()

                Omega_prop = self.Omega

                U = torch.rand(1,device=device)
                ratio = torch.minimum(torch.tensor([1.0],device=device), torch.exp(-self.beta*(Omega_prop-Omega_prev)))
                # print(U,ratio)
                if (U <= ratio):
                    self.rho_ensemble[i,:] = self.rho
                    self.Omega_ensemble[i] = Omega_prop
                    lnrho_prev = lnrho_prop
                    Omega_prev = Omega_prop
                else:
                    self.rho_ensemble[i,:] = torch.exp(lnrho_prev)
                    self.Omega_ensemble[i] = Omega_prev

            self.rho = self.rho_ensemble[burning_steps:,:].mean(axis=0).cpu()
            self.Update_System()

        elif method == 'rkf2(3)':
            # Fehlberg method 1(2)
            atolrk = 1e-3
            # K1 = F
            K2 = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
            K3 = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)

            self.Omega0 = self.Omega

            for i in range(max_iter):
                lnrho[:] += 0.5*F*self.dt
                self.rho[:] = torch.exp(lnrho).cpu()
                self.Update_System()
                K2[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)

                lnrho[:] += (1.0/256)*self.dt*(-127*F+255*K2)
                self.rho[:] = torch.exp(lnrho).cpu()
                self.Update_System()
                K3[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)

                self.Niter = i+1
                
                errrk = torch.norm(self.dt*torch.abs(F-K3)/512)/np.sqrt(self.Ngridtot)
                
                if errrk >= atolrk:
                    dtopt = 0.9*self.dt*torch.pow(atolrk/errrk,1/2.0)
                    lnrho[:] -= (1.0/256)*self.dt*(F+255*K2)
                    self.dt = torch.tensor([dtopt,0.1*self.dt], device=device).max()
                else:
                    if self.Omega <= self.Omega0:
                        dtopt = 0.9*self.dt*torch.pow(atolrk/errrk,1/3.0)
                        self.dt = torch.tensor([dtopt,5*self.dt], device=device).min()
                        F[:] = K3
                        # sk[:]= atol+rtol*torch.abs(self.rho)
                        # error = torch.norm(self.rho.to(device)*F/sk)/np.sqrt(self.Ngridtot)
                        error = (self.Omega0-self.Omega)/atol
                        self.Omega0 = self.Omega
                        if logoutput: print(self.Niter,self.Omega.numpy(),error.cpu().numpy(),self.dt.cpu().numpy())
                        if error < 1.0: break
                    else:
                        lnrho[:] -= (1.0/256)*self.dt*(F+255*K2)
                        self.dt = 0.5*self.dt
                
            del K2,K3

        del F  
        torch.cuda.empty_cache()

        self.Nabs = self.rho.sum()*self.dV
        
        if logoutput:
            print("Time to achieve equilibrium:", timeit.default_timer() - starttime, 'sec')
            print('Number of iterations:', self.Niter)
            print('error:', error.cpu().numpy())
            print('---- Equilibrium quantities ----')
            print('Fid =',self.Fid.numpy())
            print('Fexc =',self.Fexc.numpy())
            print('Omega =',self.Omega.numpy())
            print('Nbulk =',self.rhob*self.Vol)
            print('Nabs =',self.Nabs.numpy())
            print('================================')
