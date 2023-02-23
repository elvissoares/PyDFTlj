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
#plt.plot(dft1d.r[90:],np.exp(-dft1d.beta*dft1d.Vext[90:]),'--',color='grey',label=r'no corr.')
# plt.plot(dft1d.r,dft1d.rho/dft1d.rhob,'-',color='k',label=r'w/corr.')
plt.xlim(0.0,5)
plt.ylim(0,3.5)
plt.xlabel(r'$r/\sigma$')
plt.ylabel(r'$g(r)$')
plt.text(3.5,0.6,r'$k_B T/\epsilon = 0.71$',ha='center')
plt.text(3.5,0.3,r'$\rho_b \sigma^3 = 0.84$',ha='center')
plt.legend(loc='upper right',ncol=1)
plt.savefig('../figures/lj1d-argon-data.png',dpi=200)
plt.show()

import pandas as pd
df = pd.read_excel('../MCdata/MCdata-radialdistribution-lennardjones-Verlet1968.xls',sheet_name='Argon')
plt.scatter(df['r']/3.405,df['KT=0.71-rhob=0.84'],marker='o',edgecolors='C0',facecolors='none',label=r'${}^{36}$Ar @ 85K')
plt.plot(dft1d.r[90:],np.exp(-dft1d.beta*dft1d.Vext[90:]),'--',color='grey',label=r'no corr.')
# plt.plot(dft1d.r,dft1d.rho/dft1d.rhob,'-',color='k',label=r'w/corr.')
plt.xlim(0.0,5)
plt.ylim(0,3.5)
plt.xlabel(r'$r/\sigma$')
plt.ylabel(r'$g(r)$')
plt.text(3.5,0.6,r'$k_B T/\epsilon = 0.71$',ha='center')
plt.text(3.5,0.3,r'$\rho_b \sigma^3 = 0.84$',ha='center')
plt.legend(loc='upper right',ncol=1)
plt.savefig('../figures/lj1d-argon-nocorrelation.png',dpi=200)
plt.show()

plt.scatter(df['r']/3.405,df['KT=0.71-rhob=0.84'],marker='o',edgecolors='C0',facecolors='none',label=r'${}^{36}$Ar @ 85K')
plt.plot(dft1d.r[90:],np.exp(-dft1d.beta*dft1d.Vext[90:]),'--',color='grey',label=r'no corr.')
plt.plot(dft1d.r,dft1d.rho/dft1d.rhob,'-',color='k',label=r'w/corr.')
plt.xlim(0.0,5)
plt.ylim(0,3.5)
plt.xlabel(r'$r/\sigma$')
plt.ylabel(r'$g(r)$')
plt.text(3.5,0.6,r'$k_B T/\epsilon = 0.71$',ha='center')
plt.text(3.5,0.3,r'$\rho_b \sigma^3 = 0.84$',ha='center')
plt.legend(loc='upper right',ncol=1)
plt.savefig('../figures/lj1d-argon-correlation.png',dpi=200)
plt.show()
