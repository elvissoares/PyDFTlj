import numpy as np
from matplotlib import pyplot as plt
# from scipy.fft import fft2, ifft2
import pyfftw
import multiprocessing
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
from fmt2d import RFMT2D
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-09-03

pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
print('Number of cpu cores:',multiprocessing.cpu_count())

"""
 The python script to solve the DDFT equation using
 an implicit pseudospectral algorithm
"""

N = 256
rho_hat = np.empty((N,N), dtype=np.complex64)
c1_hat = np.empty((N,N), dtype=np.complex64)
phi_hat = np.empty((2,N,N), dtype=np.complex64)

rho = np.empty((N,N), dtype=np.float32)
c1 = np.empty((N,N), dtype=np.float32)

dx = 0.01
L = N*dx

fmt = RFMT2D(N,dx)

noise = 4.1
rho0 = 1.15
rho[:] = rho0*(1.0 + noise*np.random.standard_normal(rho.shape))

print(rho.sum()/N**2)

Nsteps = 2000
dt = 1.e-4

D = 1.0 # diffusity

kx = ky = np.fft.fftfreq(N, d=dx)*2*np.pi
K = np.array(np.meshgrid(kx , ky ,indexing ='ij'), dtype=np.float32)

K2 = np.sum(K*K,axis=0, dtype=np.float32)

# The anti-aliasing factor  
kmax_dealias = kx.max()*2.0/3.0 # The Nyquist mode
dealias = np.array((np.abs(K[0]) < kmax_dealias )*(np.abs(K[1]) < kmax_dealias ),dtype =bool)

rho_last = np.empty((N,N), dtype=np.float32)
error = 1.0
log = 0

rho_hat[:] = fft2(rho)
rho_hat[:] = rho_hat*np.array((np.abs(K[0]) < np.pi )*(np.abs(K[1]) < np.pi ),dtype =bool)

rho = ifft2(rho_hat).real

plt.imshow(rho)
plt.colorbar(cmap='viridis')
# plt.title('$c_0=%.1f$'% c0)
# plt.savefig('cahn-hilliard-input.png')
plt.show()

print('rho0 = ',rho.sum()*dx**2/L**2)


F = np.sum(rho*(np.log(rho)-1)+fmt.Phi(rho_hat))*dx**2/L**2

tarray = np.array([0.0])
Farray = np.array([F])

t = 0.0

while error > 1e-12:
    rho_last[:] = rho
    rho_hat[:] = fft2(rho)
    c1_hat = fmt.c1_hat(rho_hat)
    phi_hat[0] = fft2(rho*ifft2(K[0]*c1_hat).real)
    phi_hat[1] = fft2(rho*ifft2(K[1]*c1_hat).real)
    # phi_hat[0] *= dealias
    # phi_hat[1] *= dealias
    rho_hat[:] = rho_hat*(1-dt*D*K2)+(K[0]*phi_hat[0]+K[1]*phi_hat[1])*D*dt # explicit method
    # rho_hat[:] = (rho_hat*(1-dt*D*K2)+(K[0]*phi_hat[0]+K[1]*phi_hat[1])*D*dt)/(1+dt*D*K2) # implicit method
    rho[:] = ifft2(rho_hat).real # inverse fourier transform
    
    error = np.linalg.norm(rho-rho_last)/np.linalg.norm(rho_last)
    log +=1
    t += dt
    if log == 200:
        print(error)
        F = np.sum(rho*(np.log(rho)-1)+fmt.Phi(rho_hat))*dx**2/L**2
        Farray = np.append(Farray,F)
        tarray = np.append(tarray,t) 
        print(tarray)
        plt.plot(tarray,Farray)
        plt.show()
        plt.imshow(rho)
        plt.colorbar(cmap='viridis')
        # plt.title('$c_0=%.1f$'% c0)
        plt.show()
        log = 0
    
print('c = ',rho.sum()*dx**2/L**2)

plt.imshow(rho)
plt.colorbar(cmap='viridis')
plt.title('$c_0=%.1f$'% c0)
plt.savefig('cahn-hilliard-c0-%.1f.png'% c0)
plt.show()