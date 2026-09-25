import numpy as np
from numpy import pi, log, round, meshgrid
import timeit
from .eos import HSEOS, LJEOS, BHdiameter
from .aux import w3FT, w2FT,phi1func,dphi1dnfunc,phi2func,dphi2dnfunc, phi3func, dphi3dnfunc, lj3dFT, lj_potential
from .optimizer import Optimizer
import torch
import pandas as pd

# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2023-04-27
# Updated: 2026-09-25

# from CODATA: https://www.codata.org/values
kB = 1.380649e-23 # J/K
hp = 6.62607015e-34 # Planck constant in J.s
NA = 6.02214076e23 # Avogadro number
amu=1.66053906892e-27 # atomic mass unit (u)

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
    def __init__(self,functional='WBI+MMFA',padding=False,device='cuda'):
        # if device is not available, fallback to CPU
        if not torch.cuda.is_available() and device == 'cuda':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.functional = functional
        self.padding = padding
        if '+' in self.functional:
            self.fmtfunctional, self.ljfunctional = self.functional.split('+')
            if self.ljfunctional == 'WDA':
                self.psi = 1.3862 # for WDA
        else:
            self.fmtfunctional = self.functional
            self.ljfunctional = None

    # Method to Set the Number of GridPoints
    def set_number_of_gridpoints(self,number_gridpoints=None):
        self.number_gridpoints = number_gridpoints
        self.Ngrid = np.zeros(3,dtype=np.int16)
        self.Ngrid[:] = self.number_gridpoints
    
    # Method to Set the Gridsize
    def set_gridsize(self,gridsize=None):
        self.gridsize = gridsize
        self.delta = np.zeros(3,dtype=np.float32)
        self.delta[:] = self.gridsize

    # Method to Set the Fluid Properties by user
    def set_fluid_properties(self,sigma=1.0,epsilon=1.0,mass=1.0,cut_off=None,shift_cutoff=False,D = 1.0):
        self.sigma = sigma
        self.epsilon = epsilon
        self.mff = mass
        self.rcut = cut_off
        self.shift_cutoff = shift_cutoff
        self.D = D
    
    # Method to Set the Box of Geometry by user
    def set_geometry(self,lengths=1.0,angles=(90,90,90),xmin=None, xmax = None, ymin = None, ymax=None,zmin=None,zmax=None):

        self.angles = np.array(angles,dtype=np.float32)

        # test if box_dimensions is a tuple or list or float
        if isinstance(lengths, (tuple, list)):
            if len(lengths) == 2:
                self.Lbox = np.array([lengths[0],lengths[1],self.delta[2]],dtype=np.float32)
            else:
                self.Lbox = np.array(lengths,dtype=np.float32)
        else:
            self.Lbox = np.array([lengths,self.delta[1],self.delta[2]],dtype=np.float32)

        if xmin is None: self.xmin = 0.0
        else: self.xmin = xmin
        if xmax is None: self.xmax = self.Lbox[0]
        else: self.xmax = xmax
        if ymin is None: self.ymin = 0.0
        else: self.ymin = ymin
        if ymax is None: self.ymax = self.Lbox[1]
        else: self.ymax = ymax
        if zmin is None: self.zmin = 0.0
        else: self.zmin = zmin
        if zmax is None: self.zmax = self.Lbox[2]
        else: self.zmax = zmax
        self._from_ase = True
        self.Volbox = self.Lbox[0]*self.Lbox[1]*self.Lbox[2]

        self._define_geometry()

    def set_geometry_from_ase(self, atoms,xmin=None, xmax = None, ymin = None, ymax=None,zmin=None,zmax=None):
        self.solid_structure = atoms
        # Method to Set the Box of Geometry from ASE atoms object
        self.Lbox = atoms.cell.lengths()
        self.angles  = atoms.cell.angles()
        # test if cell angles are all 90 degrees (orthogonal)
        if not all(np.isclose(self.angles, [90, 90, 90])):
            raise ValueError("Non-orthogonal cell angles are not supported.")

        self.Volbox = atoms.get_volume()
        self.mss = atoms.get_masses().sum() # in atomic mass units (u)
        self.rhos = self.mss/self.Volbox # in u/A^3
        if xmin is None: self.xmin = 0.0
        else: self.xmin = xmin
        if xmax is None: self.xmax = self.Lbox[0]
        else: self.xmax = xmax
        if ymin is None: self.ymin = 0.0
        else: self.ymin = ymin
        if ymax is None: self.ymax = self.Lbox[1]
        else: self.ymax = ymax
        if zmin is None: self.zmin = 0.0
        else: self.zmin = zmin
        if zmax is None: self.zmax = self.Lbox[2]
        else: self.zmax = zmax
        self._from_ase = True
        
        print(f"Geometry defined from ASE atoms object:") 
        print(f"Lx={self.Lbox[0]:.2f} A, Ly={self.Lbox[1]:.2f} A, Lz={self.Lbox[2]:.2f} A")
        print(f"Unit cell Volume = {self.Volbox:.2f} A³")
        print(f"Solid Mass = {self.mss:.2f} u")
        print(f"Solid Mass Density = {self.rhos:.4f} u/A³")

        self._define_geometry()

    def _define_geometry(self):
        self.padding_length = np.zeros(3,dtype=np.float32)
        # If padding is necessary
        if self.padding: 
            if self.ljfunctional == 'MMFA': self.padding_length[:] = 2*self.rcut
            elif self.ljfunctional == 'WDA': self.padding_length[:] = 2*self.psi*self.sigma
            else: self.padding_length[:] = self.sigma

        self.Lgrid = (self.xmax-self.xmin + self.padding_length[0],self.ymax-self.ymin+ self.padding_length[1],self.zmax-self.zmin+ self.padding_length[2])

        self.x = np.arange(self.xmin,self.xmax,self.delta[0]) - 0.5*self.padding_length[0]
        self.y = np.arange(self.ymin,self.ymax,self.delta[1]) - 0.5*self.padding_length[1]
        self.z = np.arange(self.zmin,self.zmax,self.delta[2]) - 0.5*self.padding_length[2]

        self.X,self.Y,self.Z = meshgrid(self.x,self.y,self.z,indexing ='ij')
        self.dV = self.delta[0]*self.delta[1]*self.delta[2]

        self.Vol = self.Lgrid[0]*self.Lgrid[1]*self.Lgrid[2]
        self.Ngrid = (self.x.size,self.y.size,self.z.size) # re-define the number of gridpoints
        
        # Define the Geometry 
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

        print('Grid defined!')
        if self.padding:
            print('Padding = ', self.padding)
            print('Padding length =', self.padding_length, ' A')
        print('Lgrid =', self.Lgrid, ' A')
        print('X-axis = [', self.xmin, ',', self.xmax, '] A')
        print('Y-axis = [', self.ymin, ',', self.ymax, '] A')
        print('Z-axis = [', self.zmin, ',', self.zmax, '] A')
        print('Number of gridpoints =', self.Ngrid)
        print('delta = ', self.delta, ' A')
        print('Vol =',self.Vol, ' A³')

    def _create_system(self):

        # external potential
        if self._from_ase:
            # calculationg the helium fraction in the pore volume
            self.Vpore = np.exp(-self.beta*self.Vext_helium).sum()*self.dV
            self.helium_fraction = self.Vpore/self.Volbox

        # defining the allowed regions
        self.allowed = (self.beta*self.Vext<36.8)
        self.Vext[~self.allowed] = 36.8*self.kT

        #  Defining the Equation of States
        self.hseos = HSEOS(sigma=self.d,model=self.fmtfunctional)
        self.ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)

        # creating arrays
        self.rho = torch.zeros(tuple(self.Ngrid),dtype=torch.float32, device=self.device)
        self.lnrho = torch.zeros_like(self.rho, device=self.device)

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
        self.n2vec = torch.empty((3,)+tuple(self.Ngrid),dtype=torch.float32, device=self.device)
        self.n1vec = torch.empty_like(self.n2vec, device=self.device)

        # Defining the weight functions
        self.w3_hat = w3FT(self.Knorm,sigma=self.d)*self.sigmaLancsoz
        self.w2_hat = w2FT(self.Knorm,sigma=self.d)*self.sigmaLancsoz
        self.w2vec_hat = torch.zeros((3,)+tuple(self.Ngrid),dtype=torch.complex64, device=self.device)
        self.w2vec_hat = self.K*(-1.0j*self.w3_hat)

        if self.ljfunctional == 'MFA':
            self.ulj_hat = lj3dFT(self.Knorm,self.sigma,self.epsilon,cutoff=self.rcut,shift_cutoff=self.shift_cutoff,model='WCA')*self.sigmaLancsoz # to avoid Gibbs phenomenum
            self.amft = self.ulj_hat[0,0,0]
            self.uint = torch.zeros_like(self.rho)
        elif self.ljfunctional == 'WDA':
            self.w_hat = w3FT(self.Knorm,sigma=2*self.psi*self.d)*self.sigmaLancsoz/(pi*(2*self.psi*self.d)**3/6)
        elif self.ljfunctional == 'MMFA':
            self.ulj_hat = lj3dFT(self.Knorm,self.sigma,self.epsilon,cutoff=self.rcut,shift_cutoff=self.shift_cutoff,model='BH')*self.sigmaLancsoz # to avoid Gibbs phenomenum
            self.amft = self.ulj_hat[0,0,0]
            self.uint = torch.zeros_like(self.rho)
            self.w_hat = w3FT(self.Knorm,sigma=2*self.d)*self.sigmaLancsoz/(pi*(2*self.d)**3/6)
             
    def set_temperature(self,kT=1.0):
        self.kT = kT
        self.beta = 1/self.kT

        if self.ljfunctional != None:
            self.d = round(BHdiameter(self.kT,sigma=self.sigma,epsilon=self.epsilon),3)
        else: self.d = self.sigma
        
        self._create_system()
        
        
    def set_bulk_density(self,rhob):
        self.rhob = rhob 
        self._calculate_bulk_properties()

    
    def _calculate_bulk_properties(self):
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

    def calculate_external_potential_lj(self,forcefield_path=None):

        Vext = np.zeros(tuple(self.Ngrid),dtype=np.float32)
        self.Vext_helium = np.zeros_like(Vext)
        #Helium parameters for pycnometer
        epsilonHe = 10.22 # kelvin
        sigmaHe = 2.58  # angstrom

        if forcefield_path:
            forcefield = pd.read_csv(forcefield_path,sep=r'\s+')
        else:
            forcefield = pd.read_csv('parameters/DREIDING-forcefield.dat',sep=r'\s+')

        # Lorentz-Berthelot combination rule
        for atom in self.solid_structure:
            sigmas = forcefield.loc[forcefield["atom"] == atom.symbol, "sigma/AA"].values[0]
            epsilons = forcefield.loc[forcefield["atom"] == atom.symbol, "epsilon/kB"].values[0]
            
            sigmasf = 0.5*(self.sigma+sigmas)
            epsilonsf = np.sqrt(self.epsilon*epsilons)

            rx = self.X - atom.position[0]
            ry = self.Y - atom.position[1]
            rz = self.Z - atom.position[2]
            rx -= self.Lbox[0]*(rx/self.Lbox[0]).round() #periodic BC
            ry -= self.Lbox[1]*(ry/self.Lbox[1]).round() #periodic BC
            rz -= self.Lbox[2]*(rz/self.Lbox[2]).round() #periodic BC
    
            R = np.sqrt(rx**2 + ry**2 + rz**2+1e-16) # to avoid zero
            Vext[:] += lj_potential(R,sigmasf,epsilonsf,self.rcut,shift_cutoff=self.shift_cutoff)

            sigmasHe = 0.5*(sigmaHe+sigmas)
            epsilonsHe = np.sqrt(epsilonHe*epsilons)

            self.Vext_helium[:] += lj_potential(R,sigmasHe,epsilonsHe,self.rcut,shift_cutoff=self.shift_cutoff)

        del R
        self.set_external_potential(Vext)
        
    def set_external_potential(self,Vext):
        self.Vext = torch.zeros(tuple(self.Ngrid),dtype=torch.float32, device=self.device)
        self.Vext[:] = torch.from_numpy(Vext).to(self.device)

    def set_initial_condition(self,model='bulk'):

        self.rho[:] = 1e-16
        if model == 'bulk': self.rho[self.allowed] = self.rhob
        elif model == 'idealgas': self.rho[self.allowed] = self.rhob*torch.exp(-self.beta*self.Vext[self.allowed])

        self.rho[:] = torch.clamp(self.rho, min=1e-16)  # Ensure rho is non-negative
        self.lnrho[:] = torch.log(self.rho)
        self.state = self.lnrho.clone()
        self.best_state = self.state.clone()

        self.Ngrideff = self.rho[self.allowed].shape[0]
        self.update_system()

    def get_information(self):
        print('============== The DFT 3D for LJ fluids ==============')
        print('Functionals:')
        print('HS functional = ',self.fmtfunctional)
        if self.ljfunctional != None: print('LJ Functional = ',self.ljfunctional)
            
    def get_fluid_density_information(self):
        print('--- Fluid properties ---')
        print('epsilon/kB =', self.epsilon, ' K')
        print('sigma =', self.sigma, ' A')
        print('Temperature =', self.kT, ' K')
        print('Baker-Henderson diameter =', self.d, ' A')
        print('Bulk Density:',self.rhob, ' particles/A³')
        print('muid:',self.muid.round(3))
        print('muhs:',self.muhs.round(3))
        print('muatt:',self.muatt.round(3))

    def update_system(self):
        self._calculate_ft()
        self._calculate_weighted_densities()
        self._calculate_c1()
        self._calculate_free_energy()
        self._calculate_omega()
        self.Nabs = torch.sum(self.rho)*self.dV

    def _calculate_ft(self):
        self.rho_hat[:] = torch.fft.fftn(self.rho)

    def _calculate_weighted_densities(self):
        # Unpack the results and assign to self.n 
        self.n3[:] = self.ifft(self.rho_hat*self.w3_hat).real
        self.n2[:] = self.ifft(self.rho_hat*self.w2_hat).real
        self.n1[:] = self.n2/(2*pi*self.d)
        self.n0[:] = self.n2/(pi*self.d**2)
        self.n2vec[:] = self.ifftv(self.rho_hat*self.w2vec_hat).real
        self.n1vec[:] = self.n2vec/(2*pi*self.d)

        self.n3[:] = torch.clamp(self.n3, max=1.0-1e-16)  # Ensure n3 is non-negative
        self.n2[:] = torch.clamp(self.n2, min=1e-16)  # Ensure n2 is non-negative
        
        if self.fmtfunctional.startswith('a'):
            self.xi = (self.n2vec*self.n2vec).sum(dim=0)/((self.n2)**2)
            self.xi[:] = torch.clamp(self.xi, max=1.0-1e-16)     
        
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

    def _calculate_free_energy(self):
        # Ideal gas contribution
        self.Fid = self.kT*torch.sum(self.rho*(torch.log(self.rho)-1.0))*self.dV

        # Hard-Spheres contribution
        if self.fmtfunctional.startswith('a'):
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

    def _calculate_omega(self):
        self.Omega = self.F + torch.sum((self.Vext-self.mu)*self.rho)*self.dV

    def _calculate_c1(self):

        self.c1_hat[:] = -self.fft(self.phi1)/(pi*self.d**2)*self.w2_hat #dPhidn0
        self.c1_hat[:] += -self.fft(self.n2*self.phi2)/(2*pi*self.d)*self.w2_hat #dPhidn1

        if self.fmtfunctional.startswith('a'):
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

    # To use with optimizer
    def compute_force(self):
        F = -(self.lnrho - self.c1 - self.beta * self.mu + self.beta * self.Vext)
        F[~self.allowed] = 0.0  # Set force to zero in forbidden regions
        return F

    def apply_update(self):
        self.lnrho[:] = self.state
        self.rho[:] = torch.clamp(torch.exp(self.lnrho), min=1e-16)  # Ensure rho is non-negative
        self.update_system()
    
    def compute_energy(self):
        return self.Omega  # or total Helmholtz free energy functional


    def calculate_rho_from_pbar(self, pbar):
        rhob = torch.pow(10.0,torch.arange(-8,-1.8,0.05))
        pb = self.ljeos.p(rhob,self.kT)*(1e-5*kB)/1e-30 # pressure in bar
        rhob_value = np.interp(pbar, pb.numpy(), rhob.numpy())
        return rhob_value

    "The function to evaluate the adsorption isotherm for a given range of bulk pressure"
    def calculate_isotherm(self, pbar_values,filename='isotherm_results.dat', export_density_profile_from_pbar=None):

        if export_density_profile_from_pbar is None:
            export_density_profile_from_pbar = []

        # find the indices of pbar_values that require exporting the density profile
        export_indices = [np.argmin(np.abs(pbar_values - p)) for p in export_density_profile_from_pbar]

        from pathlib import Path
        Path("optimizer_logs").mkdir(exist_ok=True)

        rhob_values = self.calculate_rho_from_pbar(pbar_values)

        Nabsarray = np.zeros_like(rhob_values)

        self.set_bulk_density(rhob_values[0])

        self.set_initial_condition(model="bulk")

        optimizer = Optimizer.create(
            "anderson",
            self,
            alpha=0.5,
            history_size=6,
            regularization=1.0e-8,
            start_iteration=2,
            atol=1e-8,rtol=1e-4,
            max_iter=10000,
            output_log=True,
        )

        print(
            "P (bar)\t"
            "rhob (1/u.c.)\t"
            "Nabs(1/u.c.)\t"
            "Iterations\t"
            "Error"
        )

        parameter_attempts = [
            {
                "alpha": 0.6,
                "history_size": 6,
                "regularization": 1.0e-8,
            },
            {
                "alpha": 0.1,
                "history_size": 6,
                "regularization": 1.0e-7,
            },
            {
                "alpha": 0.01,
                "history_size": 6,
                "regularization": 1.0e-6,
            },
        ]

        for i, rhob in enumerate(rhob_values):
            self.set_bulk_density(float(rhob))

            # State inherited from the previous converged density.
            initial_state = self.state.detach().clone()

            converged = False
            total_iterations = 0
            final_error = float("inf")

            for attempt, parameters in enumerate(parameter_attempts,start=1):
                # Restore the same initial condition before each attempt.
                with torch.no_grad():
                    self.state.copy_(initial_state)

                self.apply_update()

                optimizer.alpha = parameters["alpha"]
                optimizer.history_size = parameters["history_size"]
                optimizer.regularization = parameters["regularization"]

                optimizer.log_filename = Path(
                    "optimizer_logs"
                ) / (f"optimizer_P={pbar_values[i]:.2f}bar_attempt_{attempt}.log")             

                optimizer.reset()

                error, iterations = optimizer.run(logoutput=False,log_interval=10)

                total_iterations += iterations
                final_error = error

                if optimizer.converged:
                    converged = True
                    break

            if not converged:
                print(f"Warning: convergence failed for P ={pbar_values[i]:.2f} bar after {total_iterations} iterations. Final error: {final_error:.2e}")

            if i in export_indices:
                # export to .cube using ase
                cube_filename = f"density_profile_{filename}_P={pbar_values[i]:.2f}bar.cube"
                from ase.io.cube import write_cube
                with open(cube_filename, 'w') as f:
                    write_cube(f, self.solid_structure,data=self.rho.cpu().numpy())

            Nabsarray[i] = self.Nabs # absolute adsorption in molecules/unit cell

            print(
                f"{pbar_values[i]:.2f}\t"
                f"{rhob * self.Vpore:.3f}\t"
                f"{Nabsarray[i]:.3f}\t"
                f"{total_iterations}\t"
                f"{final_error:.1e}"
            )

        Nexcarray = Nabsarray - rhob_values * self.Vpore # excess adsorption in molecules/unit cell

        Mabsarray = self.mff*Nabsarray
        Mexcarray = self.mff*Nexcarray

        df = pd.DataFrame()
        df['Pressure (bar)'] = pbar_values
        df['Bulk Density (molecules/AA3)'] = rhob_values
        df['Bulk Density (mol/m3)'] = rhob_values*1e30/NA
        df['Bulk Mass Density (kg/m3)'] = rhob_values*self.mff*1e-3*1e30/NA

        lam_A = hp/np.sqrt(2*np.pi*self.mff*amu*kB*self.kT) * 1e10 # in angstroms
        muarray = self.ljeos.mu(torch.from_numpy(rhob_values), self.kT, l_de_Broglie=lam_A) # in kT units
        murange = muarray * kB * NA / 4184.0 # convert to kcal/mol
        df['Chemical Potential (kcal/mol)'] = murange
        
        df['Absolute adsorption (molecules/uc)'] = Nabsarray
        df['Absolute adsorption (molecules/nm^3)'] = Nabsarray/(self.Vpore*1e-3)
        df['Absolute adsorption (mg/g)'] = 1e3*Mabsarray/self.mss
        df['Absolute adsorption (%w.t.)'] = 100*Mabsarray/(self.mss+Mabsarray)
        df['Absolute adsorption (mol/kg)'] = 1e3*Nabsarray/self.mss
        df['Absolute adsorption (cm^3 STP/cm^3)'] = 1e30*(kB*273.15/1e5)*Nabsarray/self.Vpore # STP 0ºC and 1e5 Pa (IUPAC since 1982)
        df['Excess adsorption (molecules/uc)'] = Nexcarray
        df['Excess adsorption (molecules/nm^3)'] = Nexcarray/(self.Vpore*1e-3)
        df['Excess adsorption (mg/g)'] = 1e3*Mexcarray/self.mss
        df['Excess adsorption (%w.t.)'] = 100*Mexcarray/(self.mss+Mexcarray)
        df['Excess adsorption (mol/kg)'] = 1e3*Nexcarray/self.mss
        df['Excess adsorption (cm^3 STP/cm^3)'] = 1e30*(kB*273.15/1e5)*Nexcarray/self.Vpore

        df.to_csv(f"{filename}.dat", index=False)