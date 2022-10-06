import numpy as np
from scipy import optimize
import sys
sys.path.insert(0, '../src/')
from mbwr import LJEOS
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-10-06

eos = LJEOS(sigma=1.0,epsilon=1.0)

# Objective function to critical point
def objective_cr(x):
    [rho,kT] = x
    return [kT+eos.dpdrho(rho,kT),eos.d2pdrho2(rho,kT)]
# Objective function to vapor-liquid equilibria
def objective(x,kT):
    [rhov,rhol] = x
    return [kT*np.log(rhol)+eos.mu(rhol,kT)-kT*np.log(rhov)-eos.mu(rhov,kT),kT*rhol+eos.p(rhol,kT)-kT*rhov-eos.p(rhov,kT)]
# Vapor-liquid equilibrium
def vle():
    solcr = optimize.root(objective_cr,[0.3,1.25],method='lm')
    [rhoc,kTc] = solcr.x
    kTarray = np.array([kTc])
    rhovarray = np.array([rhoc])
    rholarray = np.array([rhoc])
    x = [rhoc-0.1,rhoc+0.1]
    rhol = rhoc
    kT = kTc
    while rhol < 0.87:
        kT = kT - 0.001*kTc
        sol = optimize.root(objective, x, args=(kT),method='lm')
        [rhov,rhol] = sol.x
        x = sol.x
        kTarray=np.append(kTarray,kT)
        rhovarray=np.append(rhovarray,rhov)
        rholarray=np.append(rholarray,rhol)
    return [rhoc,kTc,np.hstack((rhovarray[::-1],rholarray)),np.hstack((kTarray[::-1],kTarray))]


[rhoc,kTc,rho,kT] = vle()

import matplotlib.pyplot as plt
import pandas as pd
plt.style.use(['science'])

df = pd.read_excel('../MCdata/MCdata-lennardjones-phasediagram.xls',sheet_name='NIST') 
plt.scatter(df['rho1'],df['T'],marker='o',edgecolors='C0',facecolors='none',linewidth=1.2,label='MC')
plt.scatter(df['rho2'],df['T'],marker='o',edgecolors='C0',facecolors='none',linewidth=1.2)

plt.plot(rho,kT,linestyle='-',color='k',label='MBWR')
plt.ylabel(r'$k_B T /\epsilon$')
plt.xlabel(r'$\rho_b \sigma^3$')
plt.xlim(0.0,0.9)
plt.ylim(0.6,1.4)
plt.legend(loc='upper right',ncol=1)

plt.savefig('../figures/phasediagram_lennardjones.png', dpi=200)
plt.show()
plt.close()

plj = np.array([eos.p(rho[i],kT[i]) + kT[i]*rho[i] for i in range(rho[rho<rhoc].size)])
plt.scatter(1.0/df['T'],df['P'],marker='o',edgecolors='C0',facecolors='none',linewidth=1.2,label='MC')
plt.plot(1.0/kT[rho<rhoc],plj,linestyle='-',color='k',label='MBWR')
plt.yscale('log')
plt.xlabel(r'$\epsilon/k_B T$')
plt.ylabel(r'$p \sigma^3/\epsilon$')
# plt.xlim(0.7,1.5)
# plt.ylim(1e-3,0.3)
plt.legend(loc='upper right',ncol=1)
plt.savefig('../figures/pressure_lennardjones.png', dpi=200)
plt.show()