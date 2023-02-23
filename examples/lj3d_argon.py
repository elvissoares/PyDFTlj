import numpy as np
import sys
sys.path.insert(0, '../src/')
from pydft1d import DFT1D
from pydft3d import DFT3D
import matplotlib.pyplot as plt
plt.style.use(['science'])

def ljpotential(r,eps=1.0,sigma=1.0):
    return 4*eps*((sigma/r)**(12)-(sigma/r)**(6))

kT = 0.71
rhob = 0.84

dft1d = DFT1D(fmtmethod='WBI',ljmethod='MMFA',geometry='Spherical')
dft1d.Set_Geometry(L=10.0)
dft1d.Set_FluidProperties(sigma=1.0,epsilon=1.0)
dft1d.Set_Temperature(kT)
dft1d.Set_BulkDensity(rhob)
Vext = ljpotential(dft1d.r)
dft1d.Set_External_Potential(Vext)
dft1d.Set_InitialCondition()
dft1d.Calculate_Equilibrium(logoutput=False)

import pandas as pd
df = pd.read_excel('../MCdata/MCdata-radialdistribution-lennardjones-Verlet1968.xls',sheet_name='Argon')
plt.scatter(df['r']/3.405,df['KT=0.71-rhob=0.84'],marker='o',edgecolors='C0',facecolors='none',label=r'${}^{36}$Ar @ 85K')
plt.plot(dft1d.r,dft1d.rho/dft1d.rhob,'-',color='k',label=r'$(\Delta = 0.01 d)^1$')
plt.xlim(0.0,5)
plt.ylim(0,3.5)
plt.xlabel(r'$r/\sigma$')
plt.ylabel(r'$g(r)$')
plt.text(3.5,0.6,r'$k_B T/\epsilon = 0.71$',ha='center')
plt.text(3.5,0.3,r'$\rho_b \sigma^3 = 0.84$',ha='center')
plt.legend(loc='upper right',ncol=1)
plt.savefig('../figures/lj3d-argon-1d.png',dpi=200)
plt.show()

fine = DFT3D(gridsize='fine',fmtmethod='WBI',ljmethod='MMFA')
fine.Set_Geometry(11.76)
fine.Set_FluidProperties(sigma=1.0,epsilon=1.0)
fine.Set_Temperature(kT)
fine.Set_BulkDensity(rhob)

X,Y,Z = np.meshgrid(fine.x,fine.y,fine.z,indexing ='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)
Vext = ljpotential(R)

fine.Set_External_Potential(Vext)
fine.Set_InitialCondition()
fine.Calculate_Equilibrium(logoutput=False)

plt.scatter(df['r']/3.405,df['KT=0.71-rhob=0.84'],marker='o',edgecolors='C0',facecolors='none',label=r'${}^{36}$Ar @ 85K')
plt.plot(dft1d.r,dft1d.rho/dft1d.rhob,'-',color='k',label=r'$(\Delta = 0.01 d)^1$')
plt.plot(fine.z,fine.rho[:,fine.N[1]//2,fine.N[2]//2]/fine.rhob,':',color='C1',label=r'$(\Delta = 0.05 d)^3$')
plt.xlim(0.0,5)
plt.ylim(0,3.5)
plt.xlabel(r'$r/\sigma$')
plt.ylabel(r'$g(r)$')
plt.text(3.5,0.6,r'$k_B T/\epsilon = 0.71$',ha='center')
plt.text(3.5,0.3,r'$\rho_b \sigma^3 = 0.84$',ha='center')
plt.legend(loc='upper right',ncol=1)
plt.savefig('../figures/lj3d-argon-fine.png',dpi=200)
plt.show()

medium = DFT3D(gridsize='medium',fmtmethod='WBI',ljmethod='MMFA')
medium.Set_Geometry(11.76)
medium.Set_FluidProperties(sigma=1.0,epsilon=1.0)
medium.Set_Temperature(kT)
medium.Set_BulkDensity(rhob)

X,Y,Z = np.meshgrid(medium.x,medium.y,medium.z,indexing ='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)
Vext = ljpotential(R)

medium.Set_External_Potential(Vext)
medium.Set_InitialCondition()
medium.Calculate_Equilibrium(logoutput=False)

plt.scatter(df['r']/3.405,df['KT=0.71-rhob=0.84'],marker='o',edgecolors='C0',facecolors='none',label=r'${}^{36}$Ar @ 85K')
plt.plot(dft1d.r,dft1d.rho/dft1d.rhob,'-',color='k',label=r'$(\Delta = 0.01 d)^1$')
plt.plot(fine.z,fine.rho[:,fine.N[1]//2,fine.N[2]//2]/fine.rhob,':',color='C1',label=r'$(\Delta = 0.05 d)^3$')
plt.plot(medium.z,medium.rho[:,medium.N[1]//2,medium.N[2]//2]/medium.rhob,'-.',color='C3',label=r'$(\Delta = 0.1 d)^3$')
plt.xlim(0.0,5)
plt.ylim(0,3.5)
plt.xlabel(r'$r/\sigma$')
plt.ylabel(r'$g(r)$')
plt.text(3.5,0.6,r'$k_B T/\epsilon = 0.71$',ha='center')
plt.text(3.5,0.3,r'$\rho_b \sigma^3 = 0.84$',ha='center')
plt.legend(loc='upper right',ncol=1)
plt.savefig('../figures/lj3d-argon-medium.png',dpi=200)
plt.show()

grained = DFT3D(gridsize='grained',fmtmethod='WBI',ljmethod='MMFA')
grained.Set_Geometry(11.76)
grained.Set_FluidProperties(sigma=1.0,epsilon=1.0)
grained.Set_Temperature(kT)
grained.Set_BulkDensity(rhob)

X,Y,Z = np.meshgrid(grained.x,grained.y,grained.z,indexing ='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)
Vext = ljpotential(R)

grained.Set_External_Potential(Vext)
grained.Set_InitialCondition()
grained.Calculate_Equilibrium(logoutput=False)

plt.scatter(df['r']/3.405,df['KT=0.71-rhob=0.84'],marker='o',edgecolors='C0',facecolors='none',label=r'${}^{36}$Ar @ 85K')
plt.plot(dft1d.r,dft1d.rho/dft1d.rhob,'-',color='k',label=r'$(\Delta = 0.01 d)^1$')
plt.plot(fine.z,fine.rho[:,fine.N[1]//2,fine.N[2]//2]/fine.rhob,':',color='C1',label=r'$(\Delta = 0.05 d)^3$')
plt.plot(medium.z,medium.rho[:,medium.N[1]//2,medium.N[2]//2]/medium.rhob,'-.',color='C3',label=r'$(\Delta = 0.1 d)^3$')
plt.plot(grained.z,grained.rho[:,grained.N[1]//2,grained.N[2]//2]/grained.rhob,'--',color='C2',label=r'$(\Delta = 0.2 d)^3$')
plt.xlim(0.0,5)
plt.ylim(0,3.5)
plt.xlabel(r'$r/\sigma$')
plt.ylabel(r'$g(r)$')
plt.text(3.5,0.6,r'$k_B T/\epsilon = 0.71$',ha='center')
plt.text(3.5,0.3,r'$\rho_b \sigma^3 = 0.84$',ha='center')
plt.legend(loc='upper right',ncol=1)
plt.savefig('../figures/lj3d-argon-grained.png',dpi=200)
plt.show()
