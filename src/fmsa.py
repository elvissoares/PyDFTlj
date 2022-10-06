import numpy as np
from mbwr import BHdiameter
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-05-05

" The Multiple Yukawa representation of LJ potential from FMSA"

def Lfunc(l,eta):
    return (1+0.5*eta)*l+1+2*eta

def Sfunc(l,eta):
    return (1-eta)**2*l**3+6*eta*(1-eta)*l**2+18*eta**2*l-12*eta*(1+2*eta)

def Qfunc(l,eta):
    return (Sfunc(l,eta)+12*eta*Lfunc(l,eta)*np.exp(-l))/((1-eta)**2*l**3)

def dLdeta(l,eta):
    return (0.5*l+2)

def dSdeta(l,eta):
    return -2*(1-eta)*l**3+6*(1-2*eta)*l**2+36*eta*l-12*(1+4*eta)

def dQdeta(l,eta):
    return (dSdeta(l,eta)+12*Lfunc(l,eta)*np.exp(-l)+12*eta*dLdeta(l,eta)*np.exp(-l))/((1-eta)**2*l**3) +2*Qfunc(l,eta)/(1-eta)

# Two Yukawa parameters of LJ direct correlation function
def DCF1d(r,rhob,kT,sigma=1.0,epsilon=1.0):

    beta = 1/kT
    # Baker-Henderson effective diameter
    d = BHdiameter(kT,sigma=sigma,epsilon=epsilon)
    
    l = np.array([2.64279,14.9677])*d/sigma
    eps = 1.94728*epsilon*(sigma/d)*np.array([1,-1])*np.exp(l*(sigma/d-1))
    # l = np.array([2.9637,14.0167])*d/sigma
    # eps = 2.1714*epsilon*(sigma/d)*np.array([1,-1])*np.exp(l*(sigma/d-1))

    eta = np.pi*rhob*d**3/6
    denom = ((1-eta)**4*l**6*Qfunc(l,eta)**2)
    A0 = -24*eta*Sfunc(l,eta)*Lfunc(l,eta)/denom
    A1 = 24*eta*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
    A2 = -12*eta*(Sfunc(l,eta)*Lfunc(l,eta)*l**2-(1-eta)**2*(1+0.5*eta)*l**6)/denom
    A4 = 0.5*eta*A1
    C1 = -Sfunc(l,eta)**2/denom
    C2 = -144*eta**2*Lfunc(l,eta)**2/denom

    c2 = np.zeros_like(r)

    for i in range(l.size):
        c2[:] += 2*np.pi*beta*eps[i]*d**2*np.exp(-l[i]*(np.abs(r/d)-1))/l[i] + beta*eps[i]*d**2*np.piecewise(r,[(np.abs(r/d)<=1),(np.abs(r/d)>1)],[lambda r: C1[i]*2*np.pi*(np.exp(-l[i]*(np.abs(r/d)-1))-1)/l[i]-C2[i]*2*np.pi*(np.exp(l[i]*(np.abs(r/d)-1))-1)/l[i]-A4[i]*0.4*np.pi*(np.abs(r/d)**5-1)-A2[i]*(2/3.0)*np.pi*(np.abs(r/d)**3-1) - A1[i]*np.pi*(np.abs(r/d)**2-1)-A0[i]*2*np.pi*(np.abs(r/d)-1),0.0] )
    
    return c2