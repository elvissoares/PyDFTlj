import numpy as np
from .eos import BHdiameter
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
    
    l = np.array([2.5449,15.4641])*d/sigma
    eps = 1.8577*epsilon*(sigma/d)*np.array([1,-1])*np.exp(l*(sigma/d-1))
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

def ljWCA1d(x,epsilon,sigma):
    r0 = 2**(1/6)*sigma
    l1 = 3.0465
    l2 = 15.4732
    eps1 = epsilon*(1+l2)/(l1-l2)
    eps2 = epsilon*(1+l1)/(l1-l2)
    return np.where(np.abs(x)<r0,np.pi*(epsilon*x**2+r0**2*(-epsilon+2*eps1/l1-2*eps2/l2)),2*np.pi*eps1*r0**2*(np.exp(-l1*(np.abs(x)/r0-1)))/l1-2*np.pi*eps2*r0**2*(np.exp(-l2*(np.abs(x)/r0-1)))/l2 )

def ljBH1d(x,epsilon,sigma):
    l = np.array([2.5449,15.4641])
    eps = 1.8577*epsilon
    # l = np.array([2.9637,14.0167])
    # eps = 2.1714*epsilon
    return np.where(np.abs(x)<sigma,-2*np.pi*sigma**2*eps*(1/l[0]-1/l[1]),-2*np.pi*eps*sigma**2*(np.exp(-l[0]*(np.abs(x)/sigma-1)))/l[0]+2*np.pi*eps*sigma**2*(np.exp(-l[1]*(np.abs(x)/sigma-1)))/l[1] )

# The 3D geometry
def A0funcFT(k,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[2*np.pi*sigma**3,lambda k: 4*np.pi*sigma*(1-np.cos(k*sigma))/k**2])

def A1funcFT(k,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[4*np.pi*sigma**3/3,lambda k: 4*np.pi*(np.sin(k*sigma)-(k*sigma)*np.cos(k*sigma))/k**3])

def A2funcFT(k,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[np.pi*sigma**3,lambda k: (4*np.pi/sigma)*(2*k*sigma*np.sin(k*sigma)+(2-k**2*sigma**2)*np.cos(k*sigma)-2)/k**4])

def A4funcFT(k,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[2*np.pi*sigma**3/3,lambda k: (4*np.pi/sigma**3)*(4*k*sigma*(k**2*sigma**2-6)*np.sin(k*sigma)-(24-12*k**2*sigma**2+k**4*sigma**4)*np.cos(k*sigma)+24)/k**6])

def YKcoreFT(k,l,sigma=1.0):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[-4*np.pi*sigma**3*(1+l-np.exp(l))/l**2,lambda k: -4*np.pi*sigma**2*(l*np.sin(k*sigma)+k*sigma*np.cos(k*sigma)-k*sigma*np.exp(l))/(k*(l**2+k**2*sigma**2))])

def YKFT(k,l,sigma=1.0):
    return 4*np.pi*sigma**3*np.piecewise(k,[k<=1e-6,k>1e-6],[(1+l)/l**2,lambda k: (k*sigma*np.cos(k*sigma)+l*np.sin(k*sigma))/(k*sigma*(l**2+(k*sigma)**2))])

def YKcutoffFT(k,l,rc=5.0,sigma=1.0):
    return 4*np.pi*sigma*np.exp(l)*np.piecewise(k,[k<=1e-6,k>1e-6],[(np.exp(-l*rc/sigma)*(rc**3*l**2/3+rc**2*l*sigma+rc*sigma**2)-rc*sigma**2)/(rc*l**2),lambda k: (np.exp(-l*rc/sigma)*((l**2+k**2*rc*l*sigma+k**2*sigma**2)*np.sin(k*rc)-k*rc*l**2*np.cos(k*rc))-k**3*rc*sigma**2)/(k**3*rc*(l**2+k**2*sigma**2))])

def DCF3dFT(K,rhob,kT,sigma=1.0,epsilon=1.0):

    beta = 1/kT
    # Baker-Henderson effective diameter
    d = BHdiameter(kT,sigma=sigma,epsilon=epsilon)

    # l = np.array([2.5449,15.4641])*d/sigma
    # eps = 1.8577*epsilon*(sigma/d)*np.array([1,-1])*np.exp(l*(sigma/d-1))
    l = np.array([2.9637,14.0167])*d/sigma
    eps = 2.1714*epsilon*(sigma/d)*np.array([1,-1])*np.exp(l*(sigma/d-1))

    rc = 5.0*d # cutoff radius        

    eta = np.pi*rhob*d**3/6
    denom = ((1-eta)**4*l**6*Qfunc(l,eta)**2)
    A0 = -24*eta*Sfunc(l,eta)*Lfunc(l,eta)/denom
    A1 = 24*eta*((1+2*eta)**2*l**4+(1-eta)*(1+2*eta)*l**5)/denom
    A2 = -12*eta*(Sfunc(l,eta)*Lfunc(l,eta)*l**2-(1-eta)**2*(1+0.5*eta)*l**6)/denom
    A4 = 0.5*eta*A1
    C1 = -Sfunc(l,eta)**2/denom
    C2 = -144*eta**2*Lfunc(l,eta)**2/denom

    c2_hat = np.zeros_like(K)

    for i in range(eps.size):
        c2_hat[:] += -beta*eps[i]*YKcutoffFT(K,l[i],rc=rc/d,sigma=d) + beta*eps[i]*(A0[i]*A0funcFT(K,sigma=d)+A1[i]*A1funcFT(K,sigma=d)+A2[i]*A2funcFT(K,sigma=d)+A4[i]*A4funcFT(K,sigma=d)+C1[i]*YKcoreFT(K,l[i],sigma=d)+C2[i]*YKcoreFT(K,-l[i],sigma=d))

    return c2_hat

def ljBH3dFT(K,kT,sigma,epsilon):
    l = np.array([2.5449,15.4641])
    eps = 1.8577*epsilon*np.array([-1,1])
    # d = BHdiameter(kT,sigma=sigma,epsilon=epsilon)
    # l = np.array([2.9637,14.0167])*d/sigma
    # eps = 2.1714*epsilon*(sigma/d)*np.array([1,-1])*np.exp(l*(sigma/d-1))
    return eps[0]*YKFT(K,l[0],sigma=sigma)+eps[1]*YKFT(K,l[1],sigma=sigma)