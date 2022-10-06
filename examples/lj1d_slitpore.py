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
L = 5.0*sigma
# Temperature and Density 
kT = 1.35
rhob = 0.09334
# solid-fluid Steele potential parameters
sigmaw, epsw, Delta = 0.903*sigma, 12.96*epsilon, 0.8044*sigma
# Test the BFD functional 
hs = DFT1D(fmtmethod='WBI',ljmethod='None',geometry='Planar')
hs.Set_Geometry(L=L)
hs.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
hs.Set_Temperature(kT)
hs.Set_BulkDensity(rhob)
hs.Set_External_Potential(extpotmodel='steele',params=[sigmaw, epsw, Delta])
hs.Set_InitialCondition()
hs.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
mfa = DFT1D(fmtmethod='WBI',ljmethod='MFA',geometry='Planar')
mfa.Set_Geometry(L=L)
mfa.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
mfa.Set_Temperature(kT)
mfa.Set_BulkDensity(rhob)
mfa.Set_External_Potential(extpotmodel='steele',params=[sigmaw, epsw, Delta])
mfa.Set_InitialCondition()
mfa.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
bfd = DFT1D(fmtmethod='WBI',ljmethod='BFD',geometry='Planar')
bfd.Set_Geometry(L=L)
bfd.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
bfd.Set_Temperature(kT)
bfd.Set_BulkDensity(rhob)
bfd.Set_External_Potential(extpotmodel='steele',params=[sigmaw, epsw, Delta])
bfd.Set_InitialCondition()
bfd.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
cwda = DFT1D(fmtmethod='WBI',ljmethod='CWDA',geometry='Planar')
cwda.Set_Geometry(L=L)
cwda.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
cwda.Set_Temperature(kT)
cwda.Set_BulkDensity(rhob)
cwda.Set_External_Potential(extpotmodel='steele',params=[sigmaw, epsw, Delta])
cwda.Set_InitialCondition()
cwda.Calculate_Equilibrium(logoutput=False)
################################################
MCdata = np.loadtxt('../MCdata/slitpore-steele-rhob0.09334-T1.35-H5.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
plt.scatter(xMC,rhoMC,marker='o',edgecolors='C0',facecolors='none',label='MC')
plt.scatter(L-xMC[::-1],rhoMC[::-1],marker='o',edgecolors='C0',facecolors='none')

plt.plot(hs.z,hs.rho,':',color='grey',label='FMT')
plt.plot(mfa.z,mfa.rho,'--',color='grey',label='MFA')
plt.plot(bfd.z,bfd.rho,'-.k',label='BFD')
plt.plot(cwda.z,cwda.rho,'-k',label='CWDA')
plt.ylim(0.0,5.5)
plt.xlim(0.0,5.0)
plt.xlabel(r'$z/\sigma$')
plt.ylabel(r'$\rho(z) \sigma^3$')
# plt.text(2.5,2.2,r'$H/\sigma = 5$',ha='center')
plt.text(2.5,1.6,r'$\rho_b \sigma^3 = 0.09334$',ha='center')
plt.text(2.5,2.0,r'$k_B T/\epsilon = 1.35$',ha='center')
plt.legend(loc='upper center',ncol=1)
plt.savefig('../figures/lj1d-slitpore-steele-rhob0.09334-T1.35-H5.png',dpi=200)
plt.show()
