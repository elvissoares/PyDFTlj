import numpy as np
import sys
sys.path.insert(0, '../src/')
from pydft3d import DFT3D
import matplotlib.pyplot as plt

dft = DFT3D(gridsize='medium',fmtmethod='WBI',ljmethod='None')

dft.Set_Geometry(12.0)
dft.Set_FluidProperties(sigma=1.0,epsilon=1.0)
dft.Set_Temperature(1.0)
dft.Set_BulkDensity(0.8)

Vext = np.zeros_like(dft.rho)

X,Y,Z = np.meshgrid(dft.x,dft.y,dft.z,indexing ='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)
mask = R<=1.0
Vext[mask] = np.inf

dft.Set_External_Potential(Vext)
dft.Set_InitialCondition()

plt.plot(dft.x,dft.rho[:,dft.N[1]//2,dft.N[2]//2]/dft.rhob)
# plt.xlim(0.0,L/2)
# plt.ylim(0,5)
plt.show()

dft.Calculate_Equilibrium(alpha0=0.62,dt=20.0,rtol=1e-7,atol=1e-8,logoutput=True)

plt.plot(dft.z,dft.rho[:,dft.N[1]//2,dft.N[2]//2]/dft.rhob)
# plt.xlim(0.0,L/2)
# plt.ylim(0,5)
plt.show()

# np.save('results/densityfield-lj-fmsa-rhostar='+str(rhob)+'-Tstar='+str(kT)+'-N'+str(N)+'-delta'+'{0:.2f}'.format(delta/d)+'.npy',n)        