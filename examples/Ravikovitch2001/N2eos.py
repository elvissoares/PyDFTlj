#!/usr/bin/env python3
# Doi: 10.1103/PhysRevE.64.011602 
# This script reproduces the Nitrogen on graphite results (Section D.1)
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
from lj_eos import LJEOS

################################################################################
if __name__ == "__main__":
        
    epsilonff = 94.45
    sigmaff = 3.575

    ljeos = LJEOS(sigma=sigmaff,epsilon=epsilonff,method='MBWR')

    rhostar = np.arange(0.004959,0.0049592,0.00000001)

    kT =  77.4 # kelvin
    beta = 1.0/kT

    pstar = (kT*rhostar+ljeos.p(rhostar/sigmaff**3,kT)*sigmaff**3)/epsilonff

    for i in range(rhostar.size):
        print(rhostar[i],pstar[i],1e-5*pstar[i]*(1.38e-23*epsilonff)/(sigmaff*1e-9)**3)

    plt.plot(rhostar,pstar,label='77 K')
    plt.show()

    rhomin = 0.001/sigmaff**3
    rhomax = 0.9/sigmaff**3

    output = False

    lnn1 = np.log(0.001/sigmaff**3)
    lnn2 = np.log(0.9/sigmaff**3)

    print('#######################################')
    print("kT\tmu\trho\trho2\tOmega1\tOmega2")

    mumin = kT*np.log(rhomin) + ljeos.mu(rhomin,kT) 
    mumax = kT*np.log(rhomax) + ljeos.mu(rhomax,kT) 

    ## The Grand Canonical Potential
    def Omega(lnn,mu):
        n = np.exp(lnn)
        return (kT*n*(lnn-1) + ljeos.f(n,kT) - mu*n)

    def dOmegadnR(lnn,mu):
        n = np.exp(lnn)
        return n*(kT*lnn + ljeos.mu(n,kT) - mu)

    error = 1.0

    while error> 1.e-6:

        muarray = np.linspace(mumin,mumax,5)

        for i in range(muarray.size):

            mu = muarray[i]

            [lnnsol,Omegasol,Niter] = optimize_fire2(lnn1,Omega,dOmegadnR,mu,alpha0=0.2,rtol=1.0e-6,dt=0.1,logoutput=output)
            [lnnsol2,Omegasol2,Niter] = optimize_fire2(lnn2,Omega,dOmegadnR,mu,alpha0=0.2,rtol=1.0e-6,dt=0.1,logoutput=output)

            rhomean = np.exp(lnnsol)
            rhomean2 = np.exp(lnnsol2)

            if (abs(rhomean2-rhomean)> 0.001):
                if Omegasol>Omegasol2: 
                    error = min(error,abs(Omegasol2-Omegasol))
                    mumax = min(mumax,mu)
                else: 
                    error = min(error,abs(Omegasol2-Omegasol))
                    mumin = max(mumin,mu)
        
    mu = (mumax+mumin)*0.5           
    [lnnsol,Omegasol,Niter] = optimize_fire2(lnn1,Omega,dOmegadnR,mu,rtol=1.0e-6,dt=0.1,logoutput=output)
    [lnnsol2,Omegasol2,Niter] = optimize_fire2(lnn2,Omega,dOmegadnR,mu,rtol=1.0e-6,dt=0.1,logoutput=output)

    rhov = np.exp(lnnsol)
    rhol = np.exp(lnnsol2)
    omega = (Omegasol+Omegasol2)*0.5
    error = abs(Omegasol-Omegasol2)

    # print('---------------------')
    print(kT,rhov*sigmaff**3,rhol*sigmaff**3,mu,-omega*sigmaff**3/epsilonff)
    print(kT,rhov*1e4/6.02,rhol*1e4/6.02)

    rhomin = rhov
    rhomax = rhol
