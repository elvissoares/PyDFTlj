#!/usr/bin/env python3
# Doi: 10.1103/PhysRevE.64.011602 
# This script is reproduces the Nitrogen on graphite results (Section D.1)
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-07-04
#
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../PyDFT3D/")
from fire import optimize_fire2
from fmt1d import FMTplanar, FMTspherical
from lj_fmsa1d import LJPlanar,ljpotential

################################################################################
if __name__ == "__main__":
        
    epsilonff = 94.45
    sigmaff = 3.575
    delta = 0.01*sigmaff
    L = 10*sigmaff

    rhob = 0.004959169999999999/sigmaff**3
    kT = 77.4
    beta = 1.0/kT

    sigmasf, epssf, rhov, Delta = 3.494, 53.22, 0.114, 3.35

    x = np.arange(0,L,delta)+0.5*delta
    N = x.size

    def Vsteele(z,sigmaw,epsw,rhov,Delta):
        return 2*np.pi*rhov*epsw*sigmaw**2*Delta*((sigmaw/z)**10-(sigmaw/z)**4-sigmaw**4/(3*Delta*(z+0.61*Delta)**3))

    Vext = beta*Vsteele(x,sigmasf,epssf,rhov,Delta)+beta*Vsteele(L-x,sigmasf,epssf,rhov,Delta)

    n = rhob*np.ones(N,dtype=np.float32)
    nsig = int(0.5*sigmaff/delta)
    mask = Vext>beta*1e4
    n[mask] = 1.0e-16
    Vext[mask] = beta*1e4

    # Now we will solve the DFT 
    nn = n.copy()

    lj = LJPlanar(N,delta,sigma=sigmaff,epsilon=epsilonff,rhob=rhob,kT=kT)
    fmt = FMTplanar(N,delta,species=1,d=np.array([lj.d]))

    mu = np.log(rhob) + fmt.mu(rhob) + beta*lj.mu

    # Now we will solve the DFT equations
    def Omega(var,mu):
        nn[:] = np.exp(var)
        Fid = np.sum(nn*(var-1.0))*delta
        Fhs = fmt.F(nn)
        Flj = beta*lj.F(nn)
        return (Fid+Fhs+Flj-np.sum(mu*nn*delta))/L

    def dOmegadnR(var,mu):
        nn[:] = np.exp(var)
        return nn*(var -fmt.c1(nn) -lj.c1(nn) - mu)*delta/L

    var = np.log(n)
    [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,mu,rtol=1.0e-9,dt=0.5,logoutput=True)
    n[:] = np.exp(varsol)

    # MCdata = np.loadtxt('MCdata/lj-hardwall-rhob0.82-T1.35.dat')
    # xMC,rhoMC = MCdata[:,0], MCdata[:,1]

    # plt.scatter(xMC+0.5,rhoMC)
    plt.plot(x,n)
    plt.xlim(0.0,18.0)
    plt.ylim(0,0.2)
    plt.show()

    # np.save('results/profiles-lennardjones-hardwall-rhob='+str(rhob)+'-T='+str(kT)+'.npy',[x,n])