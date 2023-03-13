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
from dft1d import dft1d
plt.style.use(['science'])

# fluid properties
sigma = 1.0
epsilon = 1.0
L = 7.5*sigma
# Temperature and Density 
kT = 1.2
rhob = 0.5925
# solid-fluid Steele potential parameters
sigmaw, epsw, Delta = sigma, 2*np.pi*epsilon, sigma/np.sqrt(2)
# Test the BFD functional 
hs = dft1d(fmtmethod='WBI',ljmethod='None',geometry='Planar')
hs.Set_Geometry(L=L)
hs.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
hs.Set_Temperature(kT)
hs.Set_BulkDensity(rhob)
hs.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
hs.Set_InitialCondition()
hs.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
mfa = dft1d(fmtmethod='WBI',ljmethod='MFA',geometry='Planar')
mfa.Set_Geometry(L=L)
mfa.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
mfa.Set_Temperature(kT)
mfa.Set_BulkDensity(rhob)
mfa.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
mfa.Set_InitialCondition()
mfa.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
bfd = dft1d(fmtmethod='WBI',ljmethod='BFD',geometry='Planar')
bfd.Set_Geometry(L=L)
bfd.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
bfd.Set_Temperature(kT)
bfd.Set_BulkDensity(rhob)
bfd.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
bfd.Set_InitialCondition()
bfd.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
wda = dft1d(fmtmethod='WBI',ljmethod='WDA',geometry='Planar')
wda.Set_Geometry(L=L)
wda.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
wda.Set_Temperature(kT)
wda.Set_BulkDensity(rhob)
wda.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
wda.Set_InitialCondition()
wda.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
mmfa = dft1d(fmtmethod='WBI',ljmethod='MMFA',geometry='Planar')
mmfa.Set_Geometry(L=L)
mmfa.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
mmfa.Set_Temperature(kT)
mmfa.Set_BulkDensity(rhob)
mmfa.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
mmfa.Set_InitialCondition()
mmfa.Calculate_Equilibrium(logoutput=False)
# ################################################
MCdata = np.loadtxt('../MCdata/lj-slitpore-steele-T1.2-rhob0.5925-H7.5-GEMC.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
plt.scatter(xMC,rhoMC,marker='o',edgecolors='C0',facecolors='none',label='MC')
plt.plot(hs.z,hs.rho,':',color='grey',label='FMT')
plt.plot(mfa.z,mfa.rho,':',color='C1',label='MFA')
plt.plot(bfd.z,bfd.rho,'--',color='C2',label='BFD')
plt.plot(wda.z,wda.rho,'-.',color='C3',label='WDA')
plt.plot(mmfa.z,mmfa.rho,'-k',label='MMFA')
plt.ylim(0.0,4)
plt.xlim(0.0,L)
plt.xlabel(r'$z/\sigma$')
plt.ylabel(r'$\rho(z) \sigma^3$')
plt.text(0.5*L,1.0,r'$H/\sigma =$'+str(L),ha='center')
plt.text(0.5*L,1.3,r'$\rho_b \sigma^3 =$'+str(rhob),ha='center')
plt.text(0.5*L,1.6,r'$k_B T/\epsilon =$'+str(kT),ha='center')
plt.legend(loc='upper center',ncol=2)
plt.savefig('../figures/lj1d-slitpore-steele-T1.2-rhob0.5925-H7.5.png',dpi=200)
plt.show()

L = 3.0*sigma
# Test the BFD functional 
hs.Set_Geometry(L=L)
hs.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
hs.Set_Temperature(kT)
hs.Set_BulkDensity(rhob)
hs.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
hs.Set_InitialCondition()
hs.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
mfa.Set_Geometry(L=L)
mfa.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
mfa.Set_Temperature(kT)
mfa.Set_BulkDensity(rhob)
mfa.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
mfa.Set_InitialCondition()
mfa.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
bfd.Set_Geometry(L=L)
bfd.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
bfd.Set_Temperature(kT)
bfd.Set_BulkDensity(rhob)
bfd.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
bfd.Set_InitialCondition()
bfd.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
wda.Set_Geometry(L=L)
wda.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
wda.Set_Temperature(kT)
wda.Set_BulkDensity(rhob)
wda.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
wda.Set_InitialCondition()
wda.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
mmfa.Set_Geometry(L=L)
mmfa.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
mmfa.Set_Temperature(kT)
mmfa.Set_BulkDensity(rhob)
mmfa.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
mmfa.Set_InitialCondition()
mmfa.Calculate_Equilibrium(logoutput=False)
################################################
MCdata = np.loadtxt('../MCdata/lj-slitpore-steele-T1.2-rhob0.5925-H3-GEMC.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
plt.scatter(xMC,rhoMC,marker='^',edgecolors='C0',facecolors='none',label=r'$H/\sigma =$'+str(L))
plt.plot(hs.z,hs.rho,':',color='grey')
plt.plot(mfa.z,mfa.rho,':',color='C1')
plt.plot(bfd.z,bfd.rho,'--',color='C2')
plt.plot(wda.z,wda.rho,'-.',color='C3')
plt.plot(mmfa.z,mmfa.rho,'-k')
plt.ylim(0.0,6)
plt.xlim(0.0,L)
plt.xlabel(r'$z/\sigma$')
plt.ylabel(r'$\rho(z) \sigma^3$')

L = 2.0*sigma
# Test the BFD functional 
hs.Set_Geometry(L=L)
hs.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
hs.Set_Temperature(kT)
hs.Set_BulkDensity(rhob)
hs.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
hs.Set_InitialCondition()
hs.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
mfa.Set_Geometry(L=L)
mfa.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
mfa.Set_Temperature(kT)
mfa.Set_BulkDensity(rhob)
mfa.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
mfa.Set_InitialCondition()
mfa.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
bfd.Set_Geometry(L=L)
bfd.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
bfd.Set_Temperature(kT)
bfd.Set_BulkDensity(rhob)
bfd.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
bfd.Set_InitialCondition()
bfd.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
wda.Set_Geometry(L=L)
wda.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
wda.Set_Temperature(kT)
wda.Set_BulkDensity(rhob)
wda.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
wda.Set_InitialCondition()
wda.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
mmfa.Set_Geometry(L=L)
mmfa.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
mmfa.Set_Temperature(kT)
mmfa.Set_BulkDensity(rhob)
mmfa.Set_External_Potential_Model(extpotmodel='steele',params=[sigmaw, epsw, Delta])
mmfa.Set_InitialCondition()
mmfa.Calculate_Equilibrium(logoutput=False)
MCdata = np.loadtxt('../MCdata/lj-slitpore-steele-T1.2-rhob0.5925-H2-GEMC.dat')
xMC,rhoMC = MCdata[:,0], MCdata[:,1]
plt.scatter(xMC-0.5,rhoMC,marker='s',edgecolors='C0',facecolors='none',label=r'$H/\sigma =$'+str(L))
plt.plot(hs.z-0.5,hs.rho,':',color='grey')
plt.plot(mfa.z-0.5,mfa.rho,':',color='C1')
plt.plot(bfd.z-0.5,bfd.rho,'--',color='C2')
plt.plot(wda.z-0.5,wda.rho,'-.',color='C3')
plt.plot(mmfa.z-0.5,mmfa.rho,'-k')
plt.legend(loc='upper center',ncol=1)
plt.savefig('../figures/lj1d-slitpore-steele-T1.2-rhob0.5925-H3.0and2.0.png',dpi=200)
plt.show()