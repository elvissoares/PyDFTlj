import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis import structure_matcher

plt.style.use(['science'])


from mayavi import mlab
# mlab.options.backend = 'envisage'
mlab.figure(1, bgcolor=(1,1,1), size=(1980, 1024))
mlab.clf()

structure_name = 'aCarbon-Bhatia-id001'
structure = Structure.from_file('../structures/'+structure_name+'.cif')
print('formula:  ', structure.formula)
print('num_sites:', structure.num_sites)
Lsolid = np.array([l for l in structure.lattice.lengths])

coord = structure.cart_coords
cmp = {'H': (1,1,1), 'C': (0.5,0.5,0.5), 'O': (75/255,75/255,75/255), 'N': (0,0,1), 'Zn': (0.85,0.64,0.125)}


# X,Y,Z,rho = np.load('dft-profiles-aCarbon-Bhatia-id001-CH4-TraPPE-DREIDING-T=298.0K-N=128-ljmethod=MMFA-P=1.0bar.npy')

Ngrid = np.array([128,128,128])

mlab.points3d(coord[:,0]*Ngrid[0]/Lsolid[0], coord[:,1]*Ngrid[1]/Lsolid[1], coord[:,2]*Ngrid[2]/Lsolid[2], color=cmp['C'], scale_factor=8,resolution=20,scale_mode='none')

# rhob = rho[0,0,0]
# Npad = (np.array(rho.shape)-Ngrid)//2

# lnrho = np.log10(rho[Npad[0]:-Npad[0],Npad[1]:-Npad[1],Npad[2]:-Npad[2]]/rhob+1e-30)

# rho1 = 0.0
# rho2 = 1.0
# rho3 = 2.0

# mlab.contour3d(lnrho, contours=[rho1],color=(255/256,154/256,0/256),opacity=0.6)
# mlab.contour3d(lnrho, contours=[rho2],color=(255/256,0/256,0/256),opacity=0.8)
# mlab.contour3d(lnrho, contours=[rho3],color=(128/256,0/256,128/256),opacity=1)

mlab.show()
# mlab.close(all=True)
