#!/usr/bin/env python3

# This script is the python implementation of the Density Functional Theory
# for Lennard-Jones fluids in the presence of an external potential
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-08-22
# Updated: 2022-08-22
# Version: 1.0
#
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src/')
from pydft1d import DFT1D
plt.style.use(['science'])

# fluid properties
sigma = 1.0
epsilon = 1.0
L = 12.0*sigma
# Temperature and Density 
kT = 1.35
rhob = 0.5
# Test the HS functional 
hs = DFT1D(fmtmethod='WBI',ljmethod='None',geometry='Planar')
hs.Set_Geometry(L=L)
hs.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
hs.Set_Temperature(kT)
hs.Set_BulkDensity(rhob)
hs.Set_External_Potential(extpotmodel='hardwall')
hs.Set_InitialCondition()
hs.Calculate_Equilibrium(logoutput=False)
# Test the MFA functional 
mfa = DFT1D(fmtmethod='WBI',ljmethod='MFA',geometry='Planar')
mfa.Set_Geometry(L=L)
mfa.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
mfa.Set_Temperature(kT)
mfa.Set_BulkDensity(rhob)
mfa.Set_External_Potential(extpotmodel='hardwall')
mfa.Set_InitialCondition()
mfa.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
bfd = DFT1D(fmtmethod='WBI',ljmethod='BFD',geometry='Planar')
bfd.Set_Geometry(L=L)
bfd.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
bfd.Set_Temperature(kT)
bfd.Set_BulkDensity(rhob)
bfd.Set_External_Potential(extpotmodel='hardwall')
bfd.Set_InitialCondition()
bfd.Calculate_Equilibrium(logoutput=False)
# Test the CWDA functional 
cwda = DFT1D(fmtmethod='WBI',ljmethod='CWDA',geometry='Planar')
cwda.Set_Geometry(L=L)
cwda.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
cwda.Set_Temperature(kT)
cwda.Set_BulkDensity(rhob)
cwda.Set_External_Potential(extpotmodel='hardwall')
cwda.Set_InitialCondition()
cwda.Calculate_Equilibrium(logoutput=False)
#####################################
MCdata = np.loadtxt('../MCdata/lj-hardwall-rhob0.5-T1.35.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
plt.scatter(xMC+0.5,rhoMC,marker='o',edgecolors='C0',facecolors='none',label='MC')
plt.plot(hs.z,hs.rho,':',color='grey',label='FMT')
plt.plot(mfa.z,mfa.rho,'--',color='grey',label='MFA')
plt.plot(bfd.z,bfd.rho,'-.k',label='BFD')
plt.plot(cwda.z,cwda.rho,'-k',label='CWDA')
plt.ylim(0.0,0.7)
plt.xlim(0.0,6.0)
plt.xlabel(r'$z/\sigma$')
plt.ylabel(r'$\rho(z) \sigma^3$')
plt.text(1.5,0.57,r'$k_B T/\epsilon = 1.35$',ha='center')
plt.text(1.5,0.5,r'$\rho_b \sigma^3 = 0.5$',ha='center')
plt.legend(loc='lower right',ncol=1)
plt.savefig('../figures/lj1d-hardwall-rhob=0.5-T=1.35.png',dpi=200)
plt.show()
