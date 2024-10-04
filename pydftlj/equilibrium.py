import numpy as np
import torch

# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2024-10-01
# Updated: 2024-10-01

" The Optimizer for equilibrium density profile of DFT"

class Equilibrium():
    def __init__(self,dftclass,solver='abc-fire'):
        self.solver = solver
        self.dftclass = dftclass

        if self.solver == 'picard':
            self.solverpicard = Picard(self.dftclass)
        elif self.solver == 'fire' or self.solver == 'abc-fire':
            self.solverfire = Fire(self.dftclass,self.solver)
        
    def Set_Solver_Parameters(self,alpha=0.15,dt=0.002,atol=1e-6,rtol=1e-4,max_iter=9999):
        self.alpha = alpha
        self.dt0 = dt
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter

        if self.solver == 'picard':
            self.solverpicard.Set_Solver_Parameters(alpha,atol,rtol,max_iter)
        elif self.solver == 'fire' or self.solver == 'abc-fire':
            self.solverfire.Set_Solver_Parameters(alpha,dt,atol,rtol,max_iter)

        self.setparametereq = False

    def Calculate_Equilibrium(self,logoutput=False):

        if self.setparametereq: self.Set_Solver_Parameters()
        if self.solver == 'picard':
            self.error, self.Niter = self.solverpicard.Calculate_Equilibrium(logoutput)
        elif self.solver == 'fire' or self.solver == 'abc-fire':
            self.error, self.Niter = self.solverfire.Calculate_Equilibrium(logoutput)

    def Get_Equilibrium_Properties(self):
        print('Number of iterations:', self.Niter)
        print('error:', self.error.cpu().numpy())
        print('---- Equilibrium quantities ----')
        print('Fid =',self.dftclass.Fid.cpu().numpy())
        print('Fexc =',self.dftclass.Fexc.cpu().numpy())
        print('Omega =',self.dftclass.Omega.cpu().numpy())
        print('Nbulk =',self.dftclass.Nbulk)
        print('Nabs =',self.dftclass.Nabs.cpu().numpy())

# The class of Picard Algorithm
class Picard():
    def __init__(self,dftclass):
        self.dftclass = dftclass 

    def Set_Solver_Parameters(self,alpha=0.15,atol=1e-6,rtol=1e-4,max_iter=9999):
        self.alpha = alpha
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter

    def Calculate_Equilibrium(self,logoutput=False):

        lnrho = torch.empty_like(self.dftclass.rho)
        lnrho[:] = torch.log(self.dftclass.rho) # to avoid log(0)
        self.dftclass.Update_System()

        F = torch.zeros_like(self.dftclass.rho)
        sk = torch.zeros_like(F)
        F[self.dftclass.mask] = -(lnrho[self.dftclass.mask] - self.dftclass.c1[self.dftclass.mask] - self.dftclass.beta*self.dftclass.mu + self.dftclass.beta*self.dftclass.Vext[self.dftclass.mask])
        sk[self.dftclass.mask] = self.atol+self.rtol*torch.abs(self.dftclass.rho[self.dftclass.mask])
        self.error = torch.norm(self.dftclass.rho[self.dftclass.mask]*F[self.dftclass.mask]/sk[self.dftclass.mask])/np.sqrt(self.dftclass.Ngrideff)

        if logoutput: 
            print('Iter.','Omega','error','|','alpha','dt')
            print(0,self.dftclass.Omega.cpu().numpy(),self.error.cpu().numpy(),'|',self.alpha,self.dt)
        
        # Picard algorithm    
        for i in range(self.max_iter):                
            lnrho[self.dftclass.mask] += self.alpha*F[self.dftclass.mask]
            self.dftclass.rho[self.dftclass.mask] = torch.exp(lnrho[self.dftclass.mask])
            self.dftclass.Update_System()
            F[self.dftclass.mask] = -(lnrho[self.dftclass.mask] - self.dftclass.c1[self.dftclass.mask] - self.dftclass.beta*self.dftclass.mu + self.dftclass.beta*self.dftclass.Vext[self.dftclass.mask])
            
            self.Niter = i+1
            sk[self.dftclass.mask] = self.atol+self.rtol*torch.abs(self.dftclass.rho[self.dftclass.mask])
            self.error = torch.norm(self.dftclass.rho[self.dftclass.mask]*F[self.dftclass.mask]/sk[self.dftclass.mask])/np.sqrt(self.dftclass.Ngrideff)
            if logoutput: print(self.Niter,self.dftclass.Omega.cpu().numpy(),self.error.cpu().numpy(),'|',self.alpha,self.dt)
            if self.error < 1.0: break
            if torch.isnan(self.error):
                print('Equilibrium: The system cannot achieve equilibrium!')
                break

        del F  
        torch.cuda.empty_cache()

        self.dftclass.Nabs = self.dftclass.rho.sum()*self.dftclass.dV

        return self.error.cpu().numpy(), self.Niter


# The class of FIRE Algorithm
class Fire():
    def __init__(self,dftclass,solver):
        self.dftclass = dftclass 
        self.solver = solver

    def Set_Solver_Parameters(self,alpha=0.15,dt=0.002,atol=1e-6,rtol=1e-4,max_iter=9999):
        self.alpha0 = self.alpha = alpha
        self.dt0 = self.dt = dt 
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter

    def Calculate_Equilibrium(self,logoutput=False):

        self.alpha = self.alpha0
        self.dt = self.dt0

        lnrho = torch.empty_like(self.dftclass.rho)
        lnrho[:] = torch.log(self.dftclass.rho) # to avoid log(0)
        self.dftclass.Update_System()

        F = torch.zeros_like(self.dftclass.rho)
        sk = torch.zeros_like(F)
        F[self.dftclass.mask] = -(lnrho[self.dftclass.mask] - self.dftclass.c1[self.dftclass.mask] - self.dftclass.beta*self.dftclass.mu + self.dftclass.beta*self.dftclass.Vext[self.dftclass.mask])
        sk[self.dftclass.mask] = self.atol+self.rtol*torch.abs(self.dftclass.rho[self.dftclass.mask])
        self.error = torch.norm(self.dftclass.rho[self.dftclass.mask]*F[self.dftclass.mask]/sk[self.dftclass.mask])/np.sqrt(self.dftclass.Ngrideff)

        if logoutput: 
            print('Iter.','Omega','error','|','alpha','dt')
            print(0,self.dftclass.Omega.cpu().numpy(),self.error.cpu().numpy(),'|',self.alpha,self.dt)

        # ABC-Fire algorithm https://doi.org/10.1016/j.commatsci.2022.111978
        Ndelay = 20
        Nnegmax = 2000
        dtmax = 10*self.dt0
        dtmin = 0.02*self.dt0
        Npos = 1
        Nneg = 0
        finc = 1.1
        fdec = 0.5
        fa = 0.99
        V = torch.zeros_like(self.dftclass.rho)

        for i in range(self.max_iter): 

            P = torch.sum(F[self.dftclass.mask]*V[self.dftclass.mask]) # dissipated power
            if (P>0):
                Npos = Npos + 1
                if Npos>Ndelay:
                    self.dt = min(self.dt*finc,dtmax)
                    self.alpha = max(1.0e-10,self.alpha*fa)
            else:
                Npos = 1
                Nneg = Nneg + 1
                if Nneg > Nnegmax: break
                if i> Ndelay:
                    self.dt = max(self.dt*fdec,dtmin)
                    self.alpha = self.alpha0
                lnrho[self.dftclass.mask] -= V[self.dftclass.mask]*0.5*self.dt
                V[self.dftclass.mask] = 0.0
                self.dftclass.rho[self.dftclass.mask] = torch.exp(lnrho[self.dftclass.mask])
                self.dftclass.Update_System()

            V[self.dftclass.mask] += F[self.dftclass.mask]*0.5*self.dt
            V[self.dftclass.mask] = (1-self.alpha)*V[self.dftclass.mask] + self.alpha*F[self.dftclass.mask]*torch.norm(V[self.dftclass.mask])/torch.norm(F[self.dftclass.mask])
            if self.solver == 'abc-fire': V[self.dftclass.mask] *= (1/(1-(1-self.alpha)**Npos))

            lnrho[self.dftclass.mask] += self.dt*V[self.dftclass.mask]
            self.dftclass.rho[self.dftclass.mask] = torch.exp(lnrho[self.dftclass.mask])
            self.dftclass.Update_System()
            F[self.dftclass.mask] = -(lnrho[self.dftclass.mask] - self.dftclass.c1[self.dftclass.mask] - self.dftclass.beta*self.dftclass.mu + self.dftclass.beta*self.dftclass.Vext[self.dftclass.mask])
            V[self.dftclass.mask] += F[self.dftclass.mask]*0.5*self.dt

            self.Niter = i+1
            sk[self.dftclass.mask] = self.atol+self.rtol*torch.abs(self.dftclass.rho[self.dftclass.mask])
            self.error = torch.norm(self.dftclass.rho[self.dftclass.mask]*F[self.dftclass.mask]/sk[self.dftclass.mask])/np.sqrt(self.dftclass.Ngrideff)

            if logoutput: print(self.Niter,self.dftclass.Omega.cpu().numpy(),self.error.cpu().numpy(),'|',self.alpha,self.dt)
            if self.error < 1.0 and self.Niter> Ndelay: break
            if torch.isnan(self.error):
                print('Equilibrium: The system cannot achieve equilibrium!')
                break
                
        del V, F
        torch.cuda.empty_cache()

        self.dftclass.Nabs = self.dftclass.rho.sum()*self.dftclass.dV

        return self.error.cpu().numpy(), self.Niter