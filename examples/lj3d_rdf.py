import numpy as np
import sys
sys.path.insert(0, '../src/')
from pydft3d import DFT3D
import matplotlib.pyplot as plt

dft = DFT3D(gridsize='fine',fmtmethod='WBI',ljmethod='BFD')
dft.Set_Geometry(11.76)
dft.Set_FluidProperties(sigma=1.0,epsilon=1.0)
dft.Set_Temperature(0.71)
dft.Set_BulkDensity(0.84)

Vext = np.zeros_like(dft.rho)

def ljpotential(r,eps=1.0,sigma=1.0):
    return np.where(r<0.5*sigma, 16128*eps, 4*eps*((sigma/r)**(12)-(sigma/r)**(6)))

X,Y,Z = np.meshgrid(dft.x,dft.y,dft.z,indexing ='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)
Vext[:] = ljpotential(R)

dft.Set_External_Potential(Vext)
dft.Set_InitialCondition()

plt.plot(dft.x,dft.Vext[:,dft.N[1]//2,dft.N[2]//2])
# plt.xlim(0.0,L/2)
# plt.ylim(0,5)
plt.show()

# plt.plot(dft.x,dft.rho[:,dft.N[1]//2,dft.N[2]//2]/dft.rhob)
# # plt.xlim(0.0,L/2)
# # plt.ylim(0,5)
# plt.show()

dft.Calculate_Equilibrium(logoutput=True)

plt.plot(dft.z,dft.rho[:,dft.N[1]//2,dft.N[2]//2]/dft.rhob)
# plt.xlim(0.0,L/2)
# plt.ylim(0,5)
plt.show()

# np.save('results/densityfield-lj-fmsa-rhostar='+str(rhob)+'-Tstar='+str(kT)+'-N'+str(N)+'-delta'+'{0:.2f}'.format(delta/d)+'.npy',n)        