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
L = 10.0*sigma
# Temperature and Density 
kT = 0.71
rhob = 0.84
def ljpotential(r,eps=1.0,sigma=1.0):
    return 4*eps*((sigma/r)**(12)-(sigma/r)**(6))
# Test the HS functional 
hs = DFT1D(fmtmethod='WBI',ljmethod='None',geometry='Spherical')
hs.Set_Geometry(L=L)
hs.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
hs.Set_Temperature(kT)
hs.Set_BulkDensity(rhob)
Vext = ljpotential(hs.r)
hs.Set_External_Potential(Vext)
hs.Set_InitialCondition()
hs.Calculate_Equilibrium(logoutput=False)
# Test the MFA functional 
mfa = DFT1D(fmtmethod='WBI',ljmethod='MFA',geometry='Spherical')
mfa.Set_Geometry(L=L)
mfa.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
mfa.Set_Temperature(kT)
mfa.Set_BulkDensity(rhob)
Vext = ljpotential(mfa.r)
mfa.Set_External_Potential(Vext)
mfa.Set_InitialCondition()
mfa.Calculate_Equilibrium(logoutput=False)
# Test the BFD functional 
bfd = DFT1D(fmtmethod='WBI',ljmethod='BFD',geometry='Spherical')
bfd.Set_Geometry(L=L)
bfd.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
bfd.Set_Temperature(kT)
bfd.Set_BulkDensity(rhob)
bfd.Set_External_Potential(Vext)
bfd.Set_InitialCondition()
bfd.Calculate_Equilibrium(logoutput=False)
# Test the wda functional 
wda = DFT1D(fmtmethod='WBI',ljmethod='WDA',geometry='Spherical')
wda.Set_Geometry(L=L)
wda.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
wda.Set_Temperature(kT)
wda.Set_BulkDensity(rhob)
wda.Set_External_Potential(Vext)
wda.Set_InitialCondition()
wda.Calculate_Equilibrium(logoutput=False)
# Test the MMFA functional 
mmfa = DFT1D(fmtmethod='WBI',ljmethod='MMFA',geometry='Spherical')
mmfa.Set_Geometry(L=L)
mmfa.Set_FluidProperties(sigma=sigma,epsilon=epsilon)
mmfa.Set_Temperature(kT)
mmfa.Set_BulkDensity(rhob)
mmfa.Set_External_Potential(Vext)
mmfa.Set_InitialCondition()
mmfa.Calculate_Equilibrium(logoutput=False)
#####################################
import pandas as pd
df = pd.read_excel('../MCdata/MCdata-radialdistribution-lennardjones-Verlet1968.xls',sheet_name='Argon')
plt.scatter(df['r']/3.405,df['KT=0.71-rhob=0.84'],marker='o',edgecolors='C0',facecolors='none',label=r'MD')
plt.plot(hs.z,hs.rho/rhob,':',color='grey',label='FMT')
plt.plot(mfa.z,mfa.rho/rhob,':',color='C1',label='MFA')
plt.plot(bfd.z,bfd.rho/rhob,'--',color='C2',label='BFD')
plt.plot(wda.z,wda.rho/rhob,'-.',color='C3',label='WDA')
plt.plot(mmfa.z,mmfa.rho/rhob,'-k',label='MMFA')
plt.ylim(0.0,3.5)
plt.xlim(0.0,5.0)
plt.xlabel(r'$r/\sigma$')
plt.ylabel(r'$g(r)$')
# plt.text(2.5,2.1,r'${}^{36}$Ar @ 85 K',ha='center')
plt.text(2.5,1.8,r'$k_B T/\epsilon = 0.71$',ha='center')
plt.text(2.5,1.5,r'$\rho_b \sigma^3 = 0.84$',ha='center')
plt.legend(loc='upper right',ncol=2)
plt.savefig('../figures/lj1d-rdf-rhob=0.84-T=0.71.png',dpi=200)
plt.show()


# # Other example
# # Temperature and Density 
# kT = 1.35
# rhob = 0.65
# # pure FMT 
# hs.Set_Temperature(kT)
# hs.Set_BulkDensity(rhob)
# hs.Set_External_Potential(Vext)
# hs.Set_InitialCondition()
# hs.Calculate_Equilibrium(logoutput=False)
# # MFA
# mfa.Set_Temperature(kT)
# mfa.Set_BulkDensity(rhob)
# mfa.Set_External_Potential(Vext)
# mfa.Set_InitialCondition()
# mfa.Calculate_Equilibrium(logoutput=False)
# # BFD
# bfd.Set_Temperature(kT)
# bfd.Set_BulkDensity(rhob)
# bfd.Set_External_Potential(Vext)
# bfd.Set_InitialCondition()
# bfd.Calculate_Equilibrium(logoutput=False)
# # WDA
# wda.Set_Temperature(kT)
# wda.Set_BulkDensity(rhob)
# wda.Set_External_Potential(Vext)
# wda.Set_InitialCondition()
# wda.Calculate_Equilibrium(logoutput=False)
# # MMFA
# mmfa.Set_Temperature(kT)
# mmfa.Set_BulkDensity(rhob)
# mmfa.Set_External_Potential(Vext)
# mmfa.Set_InitialCondition()
# mmfa.Calculate_Equilibrium(logoutput=False)
# #####
# MCdata = np.loadtxt('../MCdata/lj-hardwall-rhob0.65-T1.35.dat')
# xMC,rhoMC = MCdata[:,0], MCdata[:,1]
# plt.scatter(xMC+0.5,rhoMC,marker='o',edgecolors='C0',facecolors='none',label='MC')
# plt.plot(hs.z,hs.rho,':',color='grey',label='FMT')
# plt.plot(mfa.z,mfa.rho,':',color='C1',label='MFA')
# plt.plot(bfd.z,bfd.rho,'--',color='C2',label='BFD')
# plt.plot(wda.z,wda.rho,'-.',color='C3',label='WDA')
# plt.plot(mmfa.z,mmfa.rho,'-k',label='MMFA')
# plt.ylim(0.0,1.5)
# plt.xlim(0.0,6.0)
# plt.xlabel(r'$z/\sigma$')
# plt.ylabel(r'$\rho(z) \sigma^3$')
# plt.text(4.5,0.27,r'$k_B T/\epsilon = 1.35$',ha='center')
# plt.text(4.5,0.18,r'$\rho_b \sigma^3 = 0.65$',ha='center')
# plt.legend(loc='upper right',ncol=2)
# plt.savefig('../figures/lj1d-hardwall-rhob=0.65-T=1.35.png',dpi=200)
# plt.show()


# # Other example
# # Temperature and Density 
# kT = 1.35
# rhob = 0.82
# # pure FMT 
# hs.Set_Temperature(kT)
# hs.Set_BulkDensity(rhob)
# hs.Set_External_Potential(Vext)
# hs.Set_InitialCondition()
# hs.Calculate_Equilibrium(logoutput=False)
# # MFA
# mfa.Set_Temperature(kT)
# mfa.Set_BulkDensity(rhob)
# mfa.Set_External_Potential(Vext)
# mfa.Set_InitialCondition()
# mfa.Calculate_Equilibrium(logoutput=False)
# # BFD
# bfd.Set_Temperature(kT)
# bfd.Set_BulkDensity(rhob)
# bfd.Set_External_Potential(Vext)
# bfd.Set_InitialCondition()
# bfd.Calculate_Equilibrium(logoutput=False)
# # WDA
# wda.Set_Temperature(kT)
# wda.Set_BulkDensity(rhob)
# wda.Set_External_Potential(Vext)
# wda.Set_InitialCondition()
# wda.Calculate_Equilibrium(logoutput=False)
# # MMFA
# mmfa.Set_Temperature(kT)
# mmfa.Set_BulkDensity(rhob)
# mmfa.Set_External_Potential(Vext)
# mmfa.Set_InitialCondition()
# mmfa.Calculate_Equilibrium(logoutput=False)
# #####
# MCdata = np.loadtxt('../MCdata/lj-hardwall-rhob0.82-T1.35.dat')
# xMC,rhoMC = MCdata[:,0], MCdata[:,1]
# plt.scatter(xMC+0.5,rhoMC,marker='o',edgecolors='C0',facecolors='none',label='MC')
# plt.plot(hs.z,hs.rho,':',color='grey',label='FMT')
# plt.plot(mfa.z,mfa.rho,':',color='C1',label='MFA')
# plt.plot(bfd.z,bfd.rho,'--',color='C2',label='BFD')
# plt.plot(wda.z,wda.rho,'-.',color='C3',label='WDA')
# plt.plot(mmfa.z,mmfa.rho,'-k',label='MMFA')
# plt.ylim(0.0,3)
# plt.xlim(0.0,6.0)
# plt.xlabel(r'$z/\sigma$')
# plt.ylabel(r'$\rho(z) \sigma^3$')
# plt.text(4.5,0.5,r'$k_B T/\epsilon = 1.35$',ha='center')
# plt.text(4.5,0.18,r'$\rho_b \sigma^3 = 0.82$',ha='center')
# plt.legend(loc='upper right',ncol=2)
# plt.savefig('../figures/lj1d-hardwall-rhob=0.82-T=1.35.png',dpi=200)
# plt.show()
