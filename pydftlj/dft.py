import numpy as np
from numpy import pi, log, round, meshgrid
import timeit
from .equilibrium import Equilibrium
from .eos import HSEOS, LJEOS, BHdiameter
from .aux import w3FT, w2FT,phi1func,dphi1dnfunc,phi2func,dphi2dnfunc, phi3func, dphi3dnfunc, lj3dFT
import torch

# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2023-04-27
# Updated: 2024-09-24

" The DFT model for Lennard-Jones fluid on 3d geometries"

" The hard-sphere FMT functionals are implemented and denominated as: "
" functional = RF (Rosenfeld functional) "
"            = WBI (White Bear version I) "
"            = WBII (White Bear version II) "
"            = aRF (Assymetrical Rosenfeld functional) "
"            = aWBI (Assymetrical White Bear version I) "
"            = aWBII (Assymetrical White Bear version II) "

" The LJ functionals are implemented and denominated as: "
" functional = WBI+MFA (WBI + Mean-Field Theory with WCA attractive potential) "
"            = WBI+WDA (WBI + Weighted Density Approximation using MBWR EoS) "
"            = WBI+MMFA (WBI + Modified Mean Field Approximation ) "
"            = WBII+MFT (WBI + Mean-Field approximation with WCA attractive potential) "
"            = WBII+WDA (WBI + Weighted Density Approximation using MBWR EoS) "
"            = WBII+MMFA (WBI + Modified Mean Field Approximation ) "

class DFT():
    def __init__(self,ndim=3,functional='WBI+MMFA',padding=False,device='cuda'):
        self.device = torch.device(device)
        self.Set_Dimension(ndim)
        self.functional = functional
        self.padding = padding
        if '+' in self.functional:
            self.fmtfunctional, self.ljfunctional = self.functional.split('+')
            if self.ljfunctional == 'WDA':
                self.psi = 1.3862 # for WDA
        else:
            self.fmtfunctional = self.functional
            self.ljfunctional = None

        # Some housekeeping parameters  
        self.setfluid = False # Set the fluid properties
        self.setrhobulk = False # Set the bulk density 
        self.settemperature = False # Set the bulk temperature 
        self.setextpot = False
        self.setequilibrium = False # Set Equilibrium Parameter
        self.setic = False # Set Initial Condition

    # Method to Set the Number of dimensions
    def Set_Dimension(self,ndim=None):
        self.ndim = ndim
        # Some housekeeping parameters 
        self.setgridpoints = False # Set the Number of Gridpoints
        self.setdelta = False # Set the grid 
        self.setbox = False # Set the box dimensions
        self.setrhobulk = False # Set the bulk density 
        self.settemperature = False # Set the bulk temperature 

    # Method to Set the Number of GridPoints
    def Set_Number_of_Gridpoints(self,number_gridpoints=None):
        self.number_gridpoints = number_gridpoints
        self.Ngrid = np.zeros(self.ndim,dtype=np.int16)
        self.Ngrid[:] = self.number_gridpoints
        self.setgridpoints = True
    
    # Method to Set the Gridsize
    def Set_Gridsize(self,gridsize=None):
        self.gridsize = gridsize
        self.delta = np.zeros(self.ndim,dtype=np.float32)
        self.delta[:] = self.gridsize
        self.setdelta = True

    # Method to Set the Fluid Properties by user
    def Set_FluidProperties(self,sigma=1.0,epsilon=1.0,cut_off=None, D = 1.0):
        self.sigma = sigma
        self.epsilon = epsilon
        self.rcut = cut_off
        self.D = D
        self.setfluid = True
    
    # Method to Set the Box of Geometry by user
    def Set_Geometry(self,box_dimensions):
        self.box_dimensions = box_dimensions
        self.Lgrid = np.zeros(self.ndim,dtype=np.float32)
        self.Lgrid[:] = box_dimensions
        self.setbox = True

        # Test if gridpoints or gridsize were setted
        if self.setgridpoints:
            self.Set_Gridsize(gridsize=tuple(self.Lgrid/self.Ngrid))
        elif self.setdelta:
            self.Set_Number_of_Gridpoints(number_gridpoints=tuple(self.Lgrid//self.delta))
        else:
            print('User must set the Gridpoints or the Gridsize')

        # If padding is necessary
        if self.padding: 
            if self.ljfunctional == 'MMFA': self.padding_length = 2*self.rcut
            elif self.ljfunctional == 'WDA': self.padding_length = 2*self.psi*self.sigma
            else: self.padding_length = self.sigma
            self.Lgrid[:] = self.box_dimensions + self.padding_length
            
        else:
            self.padding_length = 0.0

        # Define the Geometry 
        if self.ndim == 1:
            self.x = np.arange(0.0,self.Lgrid[0],self.delta[0]) - 0.5*self.padding_length
            self.X = self.x.copy()
            self.dV = self.delta[0]
            self.Vol = self.Lgrid[0]
            self.Ngrid[:] = self.x.size # re-define the number of gridpoints
            self.Ngridtot = float(self.Ngrid[0])
        elif self.ndim == 2:
            self.x = np.arange(0.0,self.Lgrid[0],self.delta[0]) - 0.5*self.padding_length
            self.y = np.arange(0.0,self.Lgrid[1],self.delta[1]) - 0.5*self.padding_length
            self.X,self.Y = meshgrid(self.x,self.y,indexing ='ij')
            self.dV = self.delta[0]*self.delta[1]
            self.Vol = self.Lgrid[0]*self.Lgrid[1]
            self.Ngrid[:] = (self.x.size,self.y.size) # re-define the number of gridpoints
            self.Ngridtot = float(self.Ngrid[0])*float(self.Ngrid[1])
        else:
            self.x = np.arange(0.0,self.Lgrid[0],self.delta[0]) - 0.5*self.padding_length
            self.y = np.arange(0.0,self.Lgrid[1],self.delta[1]) - 0.5*self.padding_length
            self.z = np.arange(0.0,self.Lgrid[2],self.delta[2]) - 0.5*self.padding_length
            self.X,self.Y,self.Z = meshgrid(self.x,self.y,self.z,indexing ='ij')
            self.dV = self.delta[0]*self.delta[1]*self.delta[2]
            self.Vol = self.Lgrid[0]*self.Lgrid[1]*self.Lgrid[2]
            self.Ngrid[:] = (self.x.size,self.y.size,self.z.size) # re-define the number of gridpoints
            self.Ngridtot = float(self.Ngrid[0])*float(self.Ngrid[1])*float(self.Ngrid[2])

        if self.settemperature:
            self.Create_System()
    
    # def Set_Geometry_from_XYZ(self,X,Y,Z):

    #     self.box_dimensions = np.array([X.max()-X.min(),Y.max()-Y.min(),Z.max()-Z.min()])
    #     self.Lgrid = self.box_dimensions.copy()
            
    #     self.X,self.Y,self.Z = X, Y, Z
    #     self.delta = self.box_dimensions.copy()
    #     self.delta[0] = X[1,0,0]-X[0,0,0]
    #     self.delta[1] = Y[0,1,0]-Y[0,0,0]
    #     self.delta[2] = Z[0,0,1]-Z[0,0,0]
    #     self.dV = self.delta[0]*self.delta[1]*self.delta[2]
    #     self.Vol = self.Lgrid[0]*self.Lgrid[1]*self.Lgrid[2]

    #     self.Ngridtot = self.Ngrid[0]*self.Ngrid[1]*self.Ngrid[2]
        
    #     self.Create_System()
    
    def Create_System(self):
        if self.ndim == 1:
            kx = torch.fft.fftfreq(self.Ngrid[0], d=self.delta[0])*2*pi
            self.kcut = 2*pi*torch.tensor(1.0/self.delta,dtype=torch.float32, device=self.device)

            self.Kx = kx.clone()
            self.K = torch.stack((self.Kx,)).to(self.device)
            self.Knorm = torch.sqrt(self.Kx**2).to(self.device)
            del kx

            self.fft = lambda x: torch.fft.fft(x)
            self.ifft = lambda x: torch.fft.ifft(x)
            self.fftv = lambda x: torch.fft.fft(x,dim=(1))
            self.ifftv = lambda x: torch.fft.ifft(x,dim=(1))

            self.sigmaLancsoz = torch.sinc(self.K[0]/self.kcut)

        elif self.ndim == 2:
            kx = torch.fft.fftfreq(self.Ngrid[0], d=self.delta[0])*2*pi
            ky = torch.fft.fftfreq(self.Ngrid[1], d=self.delta[1])*2*pi
            self.kcut = 2*pi*torch.tensor(1.0/self.delta,dtype=torch.float32, device=self.device)

            self.Kx,self.Ky = torch.meshgrid(kx,ky,indexing ='ij')
            self.K = torch.stack((self.Kx,self.Ky)).to(self.device)
            self.Knorm = torch.sqrt(self.Kx**2 + self.Ky**2).to(self.device)
            del kx, ky

            self.fft = lambda x: torch.fft.fft2(x)
            self.ifft = lambda x: torch.fft.ifft2(x)
            self.fftv = lambda x: torch.fft.fft2(x,dim=(1,2))
            self.ifftv = lambda x: torch.fft.ifft2(x,dim=(1,2))

            self.sigmaLancsoz = torch.sinc(self.K[0]/self.kcut[0])*torch.sinc(self.K[1]/self.kcut[1])

        else:
            kx = torch.fft.fftfreq(self.Ngrid[0], d=self.delta[0])*2*pi
            ky = torch.fft.fftfreq(self.Ngrid[1], d=self.delta[1])*2*pi
            kz = torch.fft.fftfreq(self.Ngrid[2], d=self.delta[2])*2*pi
            self.kcut = 2*pi*torch.tensor(1.0/self.delta,dtype=torch.float32, device=self.device)

            self.Kx,self.Ky,self.Kz = torch.meshgrid(kx,ky,kz,indexing ='ij')
            self.K = torch.stack((self.Kx,self.Ky,self.Kz)).to(self.device)
            self.Knorm = torch.sqrt(self.Kx**2 + self.Ky**2 + self.Kz**2).to(self.device)
            del kx, ky, kz

            self.fft = lambda x: torch.fft.fftn(x)
            self.ifft = lambda x: torch.fft.ifftn(x)
            self.fftv = lambda x: torch.fft.fftn(x,dim=(1,2,3))
            self.ifftv = lambda x: torch.fft.ifftn(x,dim=(1,2,3))

            self.sigmaLancsoz = torch.sinc(self.K[0]/self.kcut[0])*torch.sinc(self.K[1]/self.kcut[1])*torch.sinc(self.K[2]/self.kcut[2])

        #  Defining the Equation of States
        self.hseos = HSEOS(sigma=self.d,model=self.fmtfunctional)
        self.ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)

        # creating arrays
        self.rho = torch.zeros(tuple(self.Ngrid),dtype=torch.float32, device=self.device)
        self.Vext = torch.zeros_like(self.rho, device=self.device)

        self.rho_hat = torch.zeros(tuple(self.Ngrid),dtype=torch.complex64, device=self.device)

        self.rhobar = torch.zeros_like(self.rho, device=self.device)
        self.mu_hat = torch.empty_like(self.rho_hat, device=self.device)
        
        self.c1 = torch.zeros_like(self.rho, device=self.device)
        self.c1_hat = torch.empty_like(self.rho_hat, device=self.device)
        self.c1hs = torch.zeros_like(self.rho, device=self.device)
        self.c1att = torch.zeros_like(self.rho, device=self.device)

        self.n0 = torch.empty_like(self.rho, device=self.device)
        self.n1 = torch.empty_like(self.rho, device=self.device)
        self.n3 = torch.empty_like(self.rho, device=self.device)
        self.n2 = torch.empty_like(self.rho, device=self.device)
        self.n2vec = torch.empty((self.ndim,)+tuple(self.Ngrid),dtype=torch.float32, device=self.device)
        self.n1vec = torch.empty_like(self.n2vec, device=self.device)

        # Defining the weight functions
        self.w3_hat = w3FT(self.Knorm,sigma=self.d)*self.sigmaLancsoz
        self.w2_hat = w2FT(self.Knorm,sigma=self.d)*self.sigmaLancsoz
        self.w2vec_hat = torch.zeros((self.ndim,)+tuple(self.Ngrid),dtype=torch.complex64, device=self.device)
        self.w2vec_hat = self.K*(-1.0j*self.w3_hat)

        if self.ljfunctional == 'MFA':
            self.ulj_hat = lj3dFT(self.Knorm,self.sigma,self.epsilon,cutoff=self.rcut,model='WCA')*self.sigmaLancsoz # to avoid Gibbs phenomenum
            if self.ndim == 1:
                self.amft = self.ulj_hat[0]
            elif self.ndim == 2:
                self.amft = self.ulj_hat[0,0]
            elif self.ndim == 3:
                self.amft = self.ulj_hat[0,0,0]
            self.uint = torch.zeros_like(self.rho)
        elif self.ljfunctional == 'WDA':
            self.w_hat = w3FT(self.Knorm,sigma=2*self.psi*self.d)*self.sigmaLancsoz/(pi*(2*self.psi*self.d)**3/6)
        elif self.ljfunctional == 'MMFA':
            self.ulj_hat = lj3dFT(self.Knorm,self.sigma,self.epsilon,cutoff=self.rcut,model='BH')*self.sigmaLancsoz # to avoid Gibbs phenomenum
            if self.ndim == 1:
                self.amft = self.ulj_hat[0]
            elif self.ndim == 2:
                self.amft = self.ulj_hat[0,0]
            elif self.ndim == 3:
                self.amft = self.ulj_hat[0,0,0]
            self.uint = torch.zeros_like(self.rho)
            self.w_hat = w3FT(self.Knorm,sigma=2*self.d)*self.sigmaLancsoz/(pi*(2*self.d)**3/6)
             
    def Set_Temperature(self,kT=1.0):
        self.kT = kT
        self.beta = 1/self.kT
        self.settemperature = True

        if self.ljfunctional != None:
            self.d = round(BHdiameter(self.kT,sigma=self.sigma,epsilon=self.epsilon),3)
        else: self.d = self.sigma
        
        if self.setbox:
            self.Create_System()
        
        
    def Set_BulkDensity(self,rhob):
        self.rhob = rhob 
        self.setrhobulk = True # bulk density is specified
        self.Calculate_BulkProperties()

    
    def Calculate_BulkProperties(self):
        # Ideal Gas
        self.fid = self.kT*self.rhob*(np.log(self.rhob)-1.0)
        self.muid = self.kT*log(self.rhob)

        # Hard-Spheres
        self.fhs  = self.kT*self.hseos.betaf(torch.tensor(self.rhob)).item()
        self.muhs = self.kT*self.hseos.betamu(torch.tensor(self.rhob)).item()

        # Lennard-Jones 
        if self.ljfunctional == 'WDA' or self.ljfunctional == 'MMFA':
            self.fatt = self.ljeos.fexc(torch.tensor(self.rhob),self.kT).item() - self.kT*self.hseos.betaf(torch.tensor(self.rhob)).item()
            self.muatt = self.ljeos.muexc(torch.tensor(self.rhob),self.kT).item() - self.kT*self.hseos.betamu(torch.tensor(self.rhob)).item()
        elif self.ljfunctional == 'MFA':
            self.fatt = 0.5*self.rhob**2*self.amft
            self.muatt = self.rhob*self.amft
        else:
            self.fatt = 0.0
            self.muatt = 0.0

        self.fexc = self.fhs + self.fatt
        self.muexc = self.muhs + self.muatt

        self.f = self.fid + self.fexc
        self.mu = self.muid + self.muexc
        
    def Set_External_Potential(self,Vext):

        if self.settemperature == False and self.ljfunctional == None:
            self.Set_Temperature()

        self.Vext[:] = torch.from_numpy(Vext).to(self.device)
        self.mask = (self.beta*self.Vext<50.0)
        self.Vext[self.beta*self.Vext>=50.0] = 50.0*self.kT

        self.setextpot = True # The external potential was specified

    def Set_InitialCondition(self,model='bulk'):

        self.rho[:] = 1.e-16
        if model == 'bulk': self.rho[self.mask] = self.rhob
        elif model == 'idealgas': self.rho[self.mask] = self.rhob*torch.exp(-self.beta*self.Vext[self.mask])

        self.Ngrideff = self.rho[self.mask].size()[0]
        self.Update_System()

        self.setic = True

    def GetInformation(self):
        print('============== The DFT 3D for LJ fluids ==============')
        print('Functionals:')
        print('HS functional = ',self.fmtfunctional)
        if self.ljfunctional != None: print('LJ Functional = ',self.ljfunctional)
        print('The number of gridpoints is ',self.Ngrid)
        print('--- Geometry properties ---')
        print('Lx =', self.Lgrid[0], ' A')
        print('Ly =', self.Lgrid[1], ' A')
        print('Lz =', self.Lgrid[2], ' A')
        print('delta = ', self.delta, ' A')
        print('Vol =',self.Vol, ' A³')
        if self.setfluid:
            print('--- Fluid properties ---')
            print('epsilon/kB =', self.epsilon, ' K')
            print('sigma =', self.sigma, ' A')
        if self.settemperature:
            print('Temperature =', self.kT, ' K')
            print('Baker-Henderson diameter =', self.d, ' A')

    def GetFluidDensityInformation(self):
        print('Bulk Density:',self.rhob, ' particles/A³')
        print('muid:',self.muid.round(3))
        print('muhs:',self.muhs.round(3))
        print('muatt:',self.muatt.round(3))

    def Update_System(self):
        self.Calculate_FT()
        self.Calculate_weighted_densities()
        self.Calculate_c1()
        self.Calculate_Omega()

    def Calculate_FT(self):
        self.rho_hat[:] = torch.fft.fftn(self.rho)

    def Calculate_weighted_densities(self):
        # Unpack the results and assign to self.n 
        self.n3[:] = self.ifft(self.rho_hat*self.w3_hat).real
        self.n2[:] = self.ifft(self.rho_hat*self.w2_hat).real
        self.n1[:] = self.n2/(2*pi*self.d)
        self.n0[:] = self.n2/(pi*self.d**2)
        self.n2vec[:] = self.ifftv(self.rho_hat*self.w2vec_hat).real
        self.n1vec[:] = self.n2vec/(2*pi*self.d)

        self.n3[self.n3>=1.0] = 1.0-1e-16 # to avoid Nan on some calculations
        if 'a' in self.fmtfunctional:
            self.xi = (self.n2vec*self.n2vec).sum(dim=0)/((self.n2+1e-16)**2)
            self.xi[self.xi>=1.0] = 1.0-1e-16      
        
        self.phi1 = phi1func(self.n3)
        self.dphi1dn3 = dphi1dnfunc(self.n3)
        self.phi2 = phi2func(self.n3,model=self.fmtfunctional)
        self.dphi2dn3 = dphi2dnfunc(self.n3,model=self.fmtfunctional)
        self.phi3 = phi3func(self.n3,model=self.fmtfunctional)
        self.dphi3dn3 = dphi3dnfunc(self.n3,model=self.fmtfunctional)
        
        if self.ljfunctional == 'WDA':
            self.rhobar[:] = self.ifft(self.rho_hat*self.w_hat).real
        elif self.ljfunctional == 'MMFA':
            self.rhobar[:] = self.ifft(self.rho_hat*self.w_hat).real
            self.uint[:] = self.ifft(self.rho_hat*self.ulj_hat).real
        elif self.ljfunctional == 'MFA':
            self.uint[:] = self.ifft(self.rho_hat*self.ulj_hat).real

    def Calculate_Free_energy(self):
        # Ideal gas contribution
        self.Fid = self.kT*torch.sum(self.rho*(torch.log(self.rho+1.0e-30)-1.0))*self.dV

        # Hard-Spheres contribution
        if 'a' in self.fmtfunctional:
            phi = self.n0*self.phi1+self.phi2*(self.n1*self.n2-(self.n1vec*self.n2vec).sum(dim=0)) + self.phi3*self.n2**3*(1-self.xi)**3
        else:
            phi = self.n0*self.phi1+self.phi2*(self.n1*self.n2-(self.n1vec*self.n2vec).sum(dim=0)) + self.phi3*(self.n2**3-3*self.n2*(self.n2vec*self.n2vec).sum(dim=0))
            
        self.Fhs = self.kT*torch.sum(phi)*self.dV

        # Lennard-Jones contribution
        if self.ljfunctional == 'WDA':
            phi[:] = self.ljeos.fexc(self.rhobar,self.kT) - self.kT*self.hseos.betaf(self.rhobar)
        elif self.ljfunctional == 'MMFA':
            phi[:] = 0.5*self.rho*self.uint + self.ljeos.fexc(self.rhobar,self.kT) - self.kT*self.hseos.betaf(self.rhobar) - 0.5*self.amft*self.rhobar**2
        elif self.ljfunctional == 'MFA':
            phi[:] = 0.5*self.rho*self.uint
        else:
            phi[:] = 0.0
        
        self.Flj = torch.sum(phi)*self.dV

        del phi

        self.Fexc =  self.Fhs + self.Flj
        self.F = self.Fid + self.Fexc

    def Calculate_Omega(self):
        self.Calculate_Free_energy()
        self.Omega = self.F + torch.sum((self.Vext-self.mu)*self.rho)*self.dV

    def Calculate_c1(self):

        self.c1_hat[:] = -self.fft(self.phi1)/(pi*self.d**2)*self.w2_hat #dPhidn0
        self.c1_hat[:] += -self.fft(self.n2*self.phi2)/(2*pi*self.d)*self.w2_hat #dPhidn1

        if 'a' in self.fmtfunctional:
            self.c1_hat[:] += -self.fft(self.n1*self.phi2 + 3*(self.n2**2)*(1+self.xi)*((1-self.xi)**2)*self.phi3)*self.w2_hat #dPhidn2

            self.c1_hat[:] += -self.fft((self.n0*self.dphi1dn3 +(self.n1*self.n2-(self.n1vec*self.n2vec).sum(dim=0))*self.dphi2dn3 + (self.n2**3*(1-self.xi)**3)*self.dphi3dn3) )*self.w3_hat #dPhidn3

            self.c1_hat[:] += (self.fftv( (-self.n2vec*self.phi2))/(2*pi*self.sigma)*self.w2vec_hat).sum(dim=0) #dPhidn1vec
            self.c1_hat[:] += (self.fftv( (-self.n1vec*self.phi2 + (- 6*self.n2*self.n2vec*(1-self.xi)**2)*self.phi3))*self.w2vec_hat).sum(dim=0) #dPhidn2vec

        else:
            self.c1_hat[:] += -self.fft(self.n1*self.phi2 + 3*(self.n2**2-(self.n2vec*self.n2vec).sum(dim=0))*self.phi3)*self.w2_hat #dPhidn2

            self.c1_hat[:] += -self.fft(self.n0*self.dphi1dn3 +(self.n1*self.n2-(self.n1vec*self.n2vec).sum(dim=0))*self.dphi2dn3 + (self.n2**3-3*self.n2*(self.n2vec*self.n2vec).sum(dim=0))*self.dphi3dn3)*self.w3_hat #dPhidn3

            self.c1_hat[:] += (self.fftv( (-self.n2vec*self.phi2))/(2*pi*self.sigma)*self.w2vec_hat).sum(dim=0) #dPhidn1vec
            self.c1_hat[:] += (self.fftv( (-self.n1vec*self.phi2 - 6*self.n2*self.n2vec*self.phi3))*self.w2vec_hat).sum(dim=0) #dPhidn2vec

        self.c1hs[:] = self.ifft(self.c1_hat).real

        if self.ljfunctional == 'WDA':
            self.mu_hat[:] =  self.fft(self.ljeos.muexc(self.rhobar,self.kT) - self.kT*self.hseos.betamu(self.rhobar))
            self.c1att[:] = -self.beta*self.ifft(self.mu_hat*self.w_hat).real
        elif self.ljfunctional == 'MMFA':
            self.mu_hat[:] =  self.fft(self.ljeos.muexc(self.rhobar,self.kT) - self.kT*self.hseos.betamu(self.rhobar) - self.amft*self.rhobar)
            self.c1att[:] = -self.beta*self.uint -self.beta*self.ifft(self.mu_hat*self.w_hat).real
        elif self.ljfunctional == 'MFA':
            self.c1att[:] = -self.beta*self.uint
            
        self.c1_hat[:] += self.fft(self.c1att)

        self.c1[:] = self.c1hs + self.c1att

    def Set_Solver_Equilibrium(self,solver='abc-fire',alpha=0.15,dt=0.002,atol=1e-6,rtol=1e-4,max_iter=9999):
        self.optimizer = Equilibrium(self,solver)
        self.optimizer.Set_Solver_Parameters(alpha,dt,atol,rtol,max_iter)

    # Method to obtain the equilibrium
    def Calculate_Equilibrium(self,logoutput=False):
        self.optimizer.Calculate_Equilibrium(logoutput)