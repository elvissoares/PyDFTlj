import numpy as np
from numpy import pi, exp, log, sqrt, round, linspace, isscalar, array, meshgrid
import timeit
from eos import LJEOS, BHdiameter
from fmtaux import sigmaLancsozFT,translationFT, w3FT, w2FT, phi3funcWBI, dphi3dnfuncWBI
from dcf import  ljBH3dFT
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
    def __init__(self,gridsize,ljmethod='MMFA'):
        self.ljmethod = ljmethod 
        self.Ngrid = gridsize 
        self.Ngridtot = self.Ngrid[0]*self.Ngrid[1]*self.Ngrid[2]

    def Set_Geometry(self,L):
        if isscalar(L): 
            self.L = array([L,L,L])
        else: 
            self.L = L
        self.Vol = self.L[0]*self.L[1]*self.L[2]

        self.delta = self.L/self.Ngrid

        self.x = np.arange(-0.5*self.L[0],0.5*self.L[0],self.delta[0])
        self.y = np.arange(-0.5*self.L[1],0.5*self.L[1],self.delta[1])
        self.z = np.arange(-0.5*self.L[2],0.5*self.L[2],self.delta[2])

        self.X,self.Y,self.Z = meshgrid(self.x,self.y,self.z,indexing ='ij')

    def Set_FluidProperties(self,sigma=1.0,epsilon=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
    
    def Set_Temperature(self,kT):

        self.kT = kT
        self.beta = 1/self.kT
        self.d = round(BHdiameter(self.kT,sigma=self.sigma,epsilon=self.epsilon),3)

    def Set_BulkDensity(self,rhob):
        self.rhob = rhob 

        ljeos = LJEOS(sigma=self.sigma,epsilon=self.epsilon)
        self.fdisp = lambda rr: ljeos.fatt(rr,self.kT)
        self.mudisp = lambda rr: ljeos.muatt(rr,self.kT) 
            
        self.Calculate_mu()
        
    def Set_External_Potential(self,Vext):

        self.Vext = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)

        self.Vext[:] = torch.tensor(Vext)
        self.mask = (self.Vext<16128)
        self.Vext[self.Vext>=16128] = 16128

    def Set_InitialCondition(self):

        self.rho = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32)

        self.rho[self.mask] = self.rhob

        # self.rho[self.mask] = self.rhob*np.exp(-0.01*self.beta*self.Vext[self.mask].cpu())

        self.rho_hat = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.complex64, device=device)

        self.rhobar = torch.empty_like(self.rho)
        self.mu_hat = torch.empty_like(self.rho_hat)
        
        self.c1 = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        self.c1hs = torch.empty_like(self.c1)
        self.c1att = torch.empty_like(self.c1)

        self.n0 = torch.empty_like(self.rho)
        self.n1 = torch.empty_like(self.rho)
        self.n3 = torch.empty_like(self.rho)
        self.n2 = torch.empty_like(self.rho)
        self.n2vec = torch.empty((3,self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32)
        self.n1vec = torch.empty_like(self.n2vec)
        
        kx = np.fft.fftfreq(self.Ngrid[0], d=self.delta[0])*2*pi
        ky = np.fft.fftfreq(self.Ngrid[1], d=self.delta[1])*2*pi
        kz = np.fft.fftfreq(self.Ngrid[2], d=self.delta[2])*2*pi
        self.kcut = array([kx.max(),ky.max(),kz.max()])
        self.Kx,self.Ky,self.Kz = meshgrid(kx,ky,kz,indexing ='ij')
        self.K = sqrt(self.Kx**2 + self.Ky**2 + self.Kz**2)
        del kx, ky, kz

        self.dV = self.delta[0]*self.delta[1]*self.delta[2]

        # Defining the weight functions
        self.w3_hat = torch.tensor(w3FT(self.K,sigma=self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut),dtype=torch.complex64, device=device)
        self.w2_hat = torch.tensor(w2FT(self.K,sigma=self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut),dtype=torch.complex64, device=device)
        self.w2vec_hat = torch.zeros((3,self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.complex64, device=device)
        self.w2vec_hat[0] = self.w3_hat*torch.tensor(-1.0j*self.Kx,dtype=torch.complex64, device=device)
        self.w2vec_hat[1] = self.w3_hat*torch.tensor(-1.0j*self.Ky,dtype=torch.complex64, device=device)
        self.w2vec_hat[2] = self.w3_hat*torch.tensor(-1.0j*self.Kz,dtype=torch.complex64, device=device)

        if self.ljmethod == 'WDA':
            psi = 1.3862
            self.w_hat = torch.tensor(w3FT(self.K,sigma=2*psi*self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)/(pi*(2*psi*self.d)**3/6),dtype=torch.complex64, device=device)
        elif self.ljmethod == 'MMFA':
            self.amft = -32*pi*self.epsilon*self.sigma**3/9
            self.fcore = lambda rr: self.fdisp(rr) - 0.5*self.amft*rr**2
            self.mucore = lambda rr: self.mudisp(rr) - self.amft*rr
            self.uint = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32)
            self.w_hat = torch.tensor(w3FT(self.K,sigma=2*self.d)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut)/(4*pi*(self.d)**3/3),dtype=torch.complex64, device=device)
            self.ulj_hat = torch.tensor(ljBH3dFT(self.K,self.sigma,self.epsilon)*sigmaLancsozFT(self.Kx,self.Ky,self.Kz,self.kcut),dtype=torch.complex64, device=device) # to avoid Gibbs phenomenum

        del self.Kx, self.Ky, self.Kz

        self.Update_System()

    def GetSystemInformation(self):
        print('============== The DFT 3D for LJ fluids ==============')
        print('Methods:')
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
        self.rho_hat[:] = torch.fft.fftn(self.rho.to(device))

        # Unpack the results and assign to self.n 
        self.n3[:] = torch.fft.ifftn(self.rho_hat*self.w3_hat).real.cpu()
        self.n2[:] = torch.fft.ifftn(self.rho_hat*self.w2_hat).real.cpu()
        self.n2vec[0] = torch.fft.ifftn(self.rho_hat*self.w2vec_hat[0]).real.cpu()
        self.n2vec[1] = torch.fft.ifftn(self.rho_hat*self.w2vec_hat[1]).real.cpu()
        self.n2vec[2] = torch.fft.ifftn(self.rho_hat*self.w2vec_hat[2]).real.cpu()

        self.n3[self.n3>=1.0] = 1.0-1e-9 # to avoid Nan on some calculations
        self.xi = (self.n2vec*self.n2vec).sum(dim=0)/(self.n2**2)
        self.xi[self.xi>=1.0] = 1.0

        self.n1vec[:] = self.n2vec/(2*pi*self.d)

        self.n0[:] = self.n2/(pi*self.d**2)
        self.n1[:] = self.n2/(2*pi*self.d)
        self.oneminusn3 = 1-self.n3

        self.phi2 = 1.0
        self.dphi2dn3 = 0.0
        self.phi3 = torch.tensor(phi3funcWBI(self.n3.numpy()),dtype=torch.float32)
        self.dphi3dn3 = torch.tensor(dphi3dnfuncWBI(self.n3.numpy()),dtype=torch.float32)
        
        if self.ljmethod == 'WDA':
            self.rhobar[:] = torch.fft.ifftn(self.rho_hat*self.w_hat).real.cpu()
            self.mu_hat[:] =  torch.fft.fftn(torch.tensor(self.mudisp(self.rhobar.numpy()),dtype=torch.float32, device=device))
        elif self.ljmethod == 'MMFA':
            self.rhobar[:] = torch.fft.ifftn(self.rho_hat*self.w_hat).real.cpu()
            self.uint[:] = torch.fft.ifftn(self.rho_hat*self.ulj_hat).real.cpu()
            self.mu_hat[:] =  torch.fft.fftn(torch.tensor(self.mucore(self.rhobar.numpy()),dtype=torch.float32, device=device))

    def Calculate_Free_energy(self):
        self.Fid = self.kT*torch.sum(self.rho*(torch.log(self.rho+1.0e-16)-1.0))*self.dV

        phi = -self.n0*torch.log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec*self.n2vec).sum(dim=0)) + (self.phi3/(24*pi*self.oneminusn3**2))*self.n2*self.n2*self.n2*(1-self.xi)**3
        
        self.Fhs = self.kT*torch.sum(phi)*self.dV

        if self.ljmethod == 'WDA':
            phi[:] = torch.tensor(self.fdisp(self.rhobar.numpy()),dtype=torch.float32)
        elif self.ljmethod == 'MMFA':
            phi[:] = 0.5*self.rho*self.uint + torch.tensor(self.fcore(self.rhobar.numpy()),dtype=torch.float32)
        self.Flj = torch.sum(phi)*self.dV

        del phi

        self.Fexc =  self.Fhs + self.Flj
        self.F = self.Fid + self.Fexc

    def Calculate_Omega(self):
        self.Calculate_Free_energy()
        self.Omega = self.F + torch.sum((self.Vext.cpu()-self.mu)*self.rho)*self.dV

    def Calculate_c1(self):

        c1_hat = -torch.fft.fftn((-torch.log(self.oneminusn3 )).to(device))/(pi*self.d**2)*self.w2_hat #dPhidn0
        c1_hat[:] += -torch.fft.fftn((self.n2*self.phi2/self.oneminusn3).to(device) )/(2*pi*self.d)*self.w2_hat #dPhidn1
        c1_hat[:] += -torch.fft.fftn((self.n1*self.phi2/self.oneminusn3  + 3*(self.n2**2)*(1+self.xi)*((1-self.xi)**2)*self.phi3/(24*pi*self.oneminusn3**2)).to(device))*self.w2_hat #dPhidn2

        c1_hat[:] += -torch.fft.fftn((self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec*self.n2vec).sum(dim=0))*(self.dphi2dn3 + self.phi2/self.oneminusn3)/self.oneminusn3 + (self.n2*self.n2*self.n2*(1-self.xi)**3)*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/ (24*pi*self.oneminusn3**2)).to(device) )*self.w3_hat #dPhidn3

        c1_hat[:] += torch.fft.fftn( (-self.n2vec[0]*self.phi2/self.oneminusn3).to(device) )/(2*pi*self.d)*self.w2vec_hat[0] #dPhidn1vec0
        c1_hat[:] += torch.fft.fftn( (-self.n2vec[1]*self.phi2/self.oneminusn3).to(device) )/(2*pi*self.d)*self.w2vec_hat[1] #dPhidn1vec1
        c1_hat[:] += torch.fft.fftn( (-self.n2vec[2]*self.phi2/self.oneminusn3).to(device) )/(2*pi*self.d)*self.w2vec_hat[2] #dPhidn1vec2
        c1_hat[:] += torch.fft.fftn( (-self.n1vec[0]*self.phi2/self.oneminusn3 + (- 6*self.n2*self.n2vec[0]*(1-self.xi)**2)*self.phi3/(24*pi*self.oneminusn3**2)).to(device))*self.w2vec_hat[0]
        c1_hat[:] += torch.fft.fftn((-self.n1vec[1]*self.phi2/self.oneminusn3 + (- 6*self.n2*self.n2vec[1]*(1-self.xi)**2)*self.phi3/(24*pi*self.oneminusn3**2)).to(device))*self.w2vec_hat[1]
        c1_hat[:] += torch.fft.fftn((-self.n1vec[2]*self.phi2/self.oneminusn3 +(- 6*self.n2*self.n2vec[2]*(1-self.xi)**2)*self.phi3/(24*pi*self.oneminusn3**2)).to(device))*self.w2vec_hat[2]

        self.c1hs[:] = torch.fft.ifftn(c1_hat).real

        del c1_hat
        torch.cuda.empty_cache()

        if self.ljmethod == 'WDA':
            self.c1att[:] = -self.beta*torch.fft.ifftn(self.mu_hat*self.w_hat).real
        elif self.ljmethod == 'MMFA':
            self.c1att[:] = -self.beta*self.uint.to(device) -self.beta*torch.fft.ifftn(self.mu_hat*self.w_hat).real

        self.c1[:] = self.c1hs + self.c1att

    def Calculate_mu(self):
        self.muid = self.kT*log(self.rhob)

        n3 = self.rhob*pi*self.d**3/6
        n2 = self.rhob*pi*self.d**2
        n1 = self.rhob*self.d/2
        n0 = self.rhob

        phi2 = 1.0
        dphi2dn3 = 0.0
        phi3 = phi3funcWBI(n3)
        dphi3dn3 = dphi3dnfuncWBI(n3)

        dPhidn0 = -log(1-n3)
        dPhidn1 = n2*phi2/(1-n3)
        dPhidn2 = n1*phi2/(1-n3) + (3*n2**2)*phi3/(24*pi*(1-n3)**2)
        dPhidn3 = n0/(1-n3) +(n1*n2)*(dphi2dn3 + phi2/(1-n3))/(1-n3) + (n2**3)*(dphi3dn3+2*phi3/(1-n3))/(24*pi*(1-n3)**2)

        self.muhs = self.kT*(dPhidn0+dPhidn1*self.d/2+dPhidn2*pi*self.d**2+dPhidn3*pi*self.d**3/6)

        self.muatt = self.mudisp(self.rhob)

        self.muexc = self.muhs + self.muatt
        self.mu = self.muid + self.muexc

    def Calculate_Equilibrium(self,alpha0=0.25,dt=0.3,atol=1e-4,rtol=1e-3,max_iter=9999,method='fire',logoutput=False):

        starttime = timeit.default_timer()

        alpha = alpha0

        lnrho = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        lnrho[:] = torch.log(self.rho+1.0e-16) # to avoid log(0)
        self.Update_System()

        F = torch.zeros((self.Ngrid[0],self.Ngrid[1],self.Ngrid[2]),dtype=torch.float32, device=device)
        sk = torch.empty_like(F)
        F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
        sk[:] = atol+rtol*torch.abs(self.rho)
        error = torch.norm(self.rho.to(device)*F/sk)/np.sqrt(self.Ngridtot)

        if logoutput: print(0,self.Omega.numpy(),error.cpu().numpy())

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
                    self.Update_System()

                V[:] += 0.5*dt*F
                V[:] = (1-alpha)*V + alpha*F*torch.norm(V)/torch.norm(F)
                lnrho[:] += dt*V
                self.rho[:] = torch.exp(lnrho).cpu()
                self.Update_System()
                F[:] = -(lnrho - self.c1 - self.beta*self.mu + self.beta*self.Vext)
                V[:] += F*0.5*dt

                self.Niter = i+1
                sk[:]=atol+rtol*torch.abs(self.rho)
                error = torch.norm(self.rho.to(device)*F/sk)/np.sqrt(self.Ngridtot)
                if logoutput: print(self.Niter,self.Omega.numpy(),error.cpu().numpy())
                if error < 1.0: break
                
            
            del V

        del F  
        torch.cuda.empty_cache()

        self.Nadstot = self.rho.sum()*self.dV
        
        if logoutput:
            print("Time to achieve equilibrium:", timeit.default_timer() - starttime, 'sec')
            print('Number of iterations:', self.Niter)
            print('error:', error.cpu().numpy())
            print('---- Equilibrium quantities ----')
            print('Fid =',self.Fid.numpy())
            print('Fexc =',self.Fexc.numpy())
            print('Omega =',self.Omega.numpy())
            print('Nbulk =',self.rhob*self.Vol)
            print('Nadstot =',self.Nadstot.numpy())
            print('================================')
