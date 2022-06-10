#!/usr/bin/env python3

# This script is the python implementation of the Density Functional Theory
# for Electrolyte Solution in the presence of an external electrostatic potential
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2021-06-02
# Updated: 2022-01-17
# Version: 2.0
#
import numpy as np
from numba import jit, njit, vectorize, prange, int32, float32, float64    # import the types
from scipy import optimize
import matplotlib.pyplot as plt

" Global variables for the FIRE algorithm"
Ndelay = 20
Nmax = 10000
finc = 1.1
fdec = 0.5
fa = 0.99
Nnegmax = 2000


def Optimize(dft,method='picard',logoutput=False):
    dft.Update_System()

    if method == 'picard':
        atol=1.e-3
        rtol = 1.e-5
        alpha = 0.02
        lnrho = np.log(dft.rho) 

        nsig = (0.5*dft.d/dft.delta).astype(int)  
        errorlast = np.inf

        for i in range(Nmax):
            lnrhonew = dft.c1 + dft.mu[:,np.newaxis] - dft.Vext
            lnrhonew[0,:nsig[0]] = np.log(1.0e-16)
            lnrhonew[1,:nsig[1]] = np.log(1.0e-16)
            
            lnrho[:] = (1-alpha)*lnrho + alpha*lnrhonew
            dft.rho[:] = np.exp(lnrho)
            dft.Update_System()

            F = (lnrho - lnrhonew)/(atol+lnrho*rtol)

            # error = max(abs(F.min()),F.max())
            error = np.linalg.norm(F)
            if errorlast > error:  alpha = min(0.02,alpha*finc)
            else: alpha = max(1.0e-3,alpha*fdec)
            # if error< 100.0: 
            #     alpha = alpha*1.001
            errorlast = error
            if error < 1.0: break
            if logoutput: print(i,dft.Omega,error)

    elif method == 'fire':
        alpha0 = 0.2
        atol = 1.e-5
        dt = 0.0001
        error = 10*atol 
        dtmax = 10*dt
        dtmin = 0.02*dt
        alpha = alpha0
        Npos = 0
        Nneg = 0

        lnrho = np.log(dft.rho)
        V = np.zeros((dft.species,dft.N),dtype=np.float32)
        F = -(lnrho -dft.c1 - dft.mu[:,np.newaxis]+dft.Vext)

        for i in range(Nmax):

            P = (F*V).sum() # dissipated power
            
            if (P>0):
                Npos = Npos + 1
                if Npos>Ndelay:
                    dt = min(dt*finc,dtmax)
                    alpha = alpha*fa
            else:
                Npos = 0
                Nneg = Nneg + 1
                if Nneg > Nnegmax: break
                if i> Ndelay:
                    dt = max(dt*fdec,dtmin)
                    alpha = alpha0
                lnrho[:] = lnrho - 0.5*dt*V
                V = np.zeros((dft.species,dft.N),dtype=np.float32)
                dft.rho[:] = np.exp(lnrho)
                dft.Update_System()

            V[:] = V + 0.5*dt*F
            V[:] = (1-alpha)*V + alpha*F*np.linalg.norm(V)/np.linalg.norm(F)
            lnrho[:] = lnrho + dt*V
            dft.rho[:] = np.exp(lnrho)
            dft.Update_System()
            F[:] = -(lnrho -dft.c1 - dft.mu[:,np.newaxis]+dft.Vext)
            V[:] = V + 0.5*dt*F

            error = max(np.abs(F.min()),F.max())
            if error < atol: break

            if logoutput: print(i,dft.Omega,error)

        del V, F  

    elif method == 'anderson':
        alpha0=0.01
        atol=1.e-6
        m = 3
        beta0 = (1.0/m)*np.ones(m)
        beta = beta0.copy()
        alpha = alpha0

        x = np.log(dft.rho)
        u = np.zeros_like(x)

        xstr = np.zeros((m,dft.species,dft.N),dtype=np.float32)
        ustr = np.zeros((m,dft.species,dft.N),dtype=np.float32)
        Fstr = np.zeros(m,dtype=np.float32)

        nsig = (0.5*dft.d/dft.delta).astype(int)

        for i in range(Nmax):

            u[:] = dft.c1exc + dft.mu[:,np.newaxis] - dft.Vext
            u[0,:nsig[0]] = np.log(1.0e-16)
            u[1,:nsig[1]] = np.log(1.0e-16)

            plt.plot(dft.x,dft.Psi)
            plt.show()

            plt.plot(dft.x,u[0,:])
            plt.plot(dft.x,u[1,:])
            plt.show()
            if i < m:
                xstr[i] = x
                Fstr[i] = np.linalg.norm(u-x)
                ustr[i] = u
            else:
                xstr[:m-1] = xstr[1:m]
                xstr[m-1] = x
                Fstr[:m-1] = Fstr[1:m]
                Fstr[m-1] = np.linalg.norm(u-x)
                ustr[:m-1] = ustr[1:m]
                ustr[m-1] = u

                def objective(alp):
                    return np.sum(alp**2*Fstr)

                res = optimize.minimize(objective, np.sqrt(beta), method='Nelder-Mead', tol=1e-3)
                beta[:] = res.x**2/np.sum(res.x**2)
                print(beta)

                x[:] = (1-alpha)*beta[0]*xstr[0] + alpha*beta[0]*ustr[0]
                for l in range(1,m):
                    x += (1-alpha)*beta[l]*xstr[l] + alpha*beta[l]*ustr[l]
                dft.rho[:] = np.exp(x + dft.c1coul)
                dft.Update_System()

            F = (u-x)
                
            error = max(np.abs(F.min()),F.max())
            # if i > 0:
            #     if errorlast > error:  alpha = max(0.001,alpha*1.1)
            #     else: alpha = min(1.0e-4,alpha*0.9)
            errorlast = error
            if logoutput: print(i,dft.Omega,error)
            if error < atol: break

        del F, xstr, Fstr, ustr

    return i