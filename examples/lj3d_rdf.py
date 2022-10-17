import numpy as np
import sys
sys.path.insert(0, '../src/')
from pydft3d import DFT3D
import matplotlib.pyplot as plt
plt.style.use(['science'])

kT = 0.71
rhob = 0.84

bfd = DFT3D(gridsize='grained',fmtmethod='WBI',ljmethod='BFD')
bfd.Set_Geometry(11.76)
bfd.Set_FluidProperties(sigma=1.0,epsilon=1.0)
bfd.Set_Temperature(kT)
bfd.Set_BulkDensity(rhob)

Vext = np.zeros_like(bfd.rho)

def ljpotential(r,eps=1.0,sigma=1.0):
    return 4*eps*((sigma/r)**(12)-(sigma/r)**(6))

X,Y,Z = np.meshgrid(bfd.x,bfd.y,bfd.z,indexing ='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)
Vext[:] = ljpotential(R)

bfd.Set_External_Potential(Vext)
bfd.Set_InitialCondition()
bfd.Calculate_Equilibrium(logoutput=False)

wda = DFT3D(gridsize='grained',fmtmethod='WBI',ljmethod='WDA')
wda.Set_Geometry(11.76)
wda.Set_FluidProperties(sigma=1.0,epsilon=1.0)
wda.Set_Temperature(kT)
wda.Set_BulkDensity(rhob)
wda.Set_External_Potential(Vext)
wda.Set_InitialCondition()
wda.Calculate_Equilibrium(logoutput=False)

mmfa = DFT3D(gridsize='grained',fmtmethod='WBI',ljmethod='MMFA')
mmfa.Set_Geometry(11.76)
mmfa.Set_FluidProperties(sigma=1.0,epsilon=1.0)
mmfa.Set_Temperature(kT)
mmfa.Set_BulkDensity(rhob)
mmfa.Set_External_Potential(Vext)
mmfa.Set_InitialCondition()
mmfa.Calculate_Equilibrium(logoutput=False)

import pandas as pd
df = pd.read_excel('../MCdata/MCdata-radialdistribution-lennardjones-Verlet1968.xls',sheet_name='Argon')
plt.scatter(df['r']/3.405,df['KT=0.71-rhob=0.84'],marker='o',edgecolors='C0',facecolors='none',label=r'MD')
plt.plot(mmfa.z,mmfa.rho[:,mmfa.N[1]//2,mmfa.N[2]//2]/mmfa.rhob,'-k',label='MMFA')
plt.plot(wda.z,wda.rho[:,wda.N[1]//2,wda.N[2]//2]/wda.rhob,'-.',color='C3',label='WDA')
plt.plot(bfd.z,bfd.rho[:,bfd.N[1]//2,bfd.N[2]//2]/bfd.rhob,'--',color='C2',label='BFD')
plt.xlim(0.0,5)
plt.ylim(0,3.5)
plt.xlabel(r'$r/\sigma$')
plt.ylabel(r'$g(r)$')
plt.text(2.5,2.0,r'$k_B T/\epsilon = 0.71$',ha='center')
plt.text(2.5,1.7,r'$\rho_b \sigma^3 = 0.84$',ha='center')
plt.legend(loc='upper right',ncol=1)
plt.savefig('../figures/lj3d-radialdistribution-rhob=0.84-T=0.71.png',dpi=200)
plt.show()
