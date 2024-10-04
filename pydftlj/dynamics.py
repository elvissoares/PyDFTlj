import numpy as np
import torch

# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2024-10-01
# Updated: 2024-10-01

" The solver for dynamics of the density profile of DDFT"

class Dynamics():
    def __init__(self,dftclass):
        self.dftclass = dftclass
        
    def Set_Solver_Parameters(self,dt=1e-4):
        self.h = dt*self.dftclass.sigma**2/self.dftclass.D # in scales of diffusion
        # The linear terms of PDE
        Loperator_k = -self.dftclass.D*self.dftclass.Knorm**2
        self.Tlinear_k = torch.exp(self.h*Loperator_k) 
        # Dealising matrix
        dealias = (torch.abs(self.dftclass.K[0]) < self.dftclass.kcut[0]*2.0/3.0 )*(torch.abs(self.dftclass.K[1]) < self.dftclass.kcut[1]*2.0/3.0 )*(torch.abs(self.dftclass.K[2]) < self.dftclass.kcut[2]*2.0/3.0 ).to(device)
        # Defining the time marching operators arrays
        self.Tnon_k = dealias*self.h*torch.where(self.Tlinear_k == 1.0,1.0,(self.Tlinear_k -1.0)/torch.log(self.Tlinear_k ))

        self.Noperator_k = torch.zeros_like(self.dftclass.rho_hat,device=self.dftclass.device)

        self.Vext_hat = torch.fft.fftn(self.dftclass.Vext)

        self.dftclass.Calculate_FT()

    # Compute a new time step using the exponential integrators in pseudo spectral methods
    def Calculate_TimeStep(self,ti,tf):

        t = np.arange(ti,tf,self.h)
        
        self.dftclass.Calculate_FT()

        for i in range(t.size):
            self.dftclass.Calculate_weighted_densities()
            self.dftclass.Calculate_c1()
            
            # calculate the nonlinear operator (with dealising)
            self.Noperator_k[:] = -1.0j*self.dftclass.D*torch.sum(self.dftclass.K*torch.fft.fftn(self.dftclass.rho*torch.fft.ifftn(1.0j*self.dftclass.K*(self.dftclass.c1_hat-self.dftclass.beta*self.Vext_hat),dim=(1,2,3)).real,dim=(1,2,3)),dim=0)

            # updating in time
            self.dftclass.rho_hat[:] = self.dftclass.rho_hat*self.Tlinear_k + self.Noperator_k*self.Tnon_k 

            # IFT to next step
            self.dftclass.rho[:] = torch.fft.ifftn(self.dftclass.rho_hat).real 

        # Update Free-energy and so on
        self.dftclass.Update_System()

