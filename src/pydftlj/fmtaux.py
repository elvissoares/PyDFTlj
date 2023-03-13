from numba import vectorize
from numpy import pi, sinc, exp, log, piecewise
from scipy.special import spherical_jn

def sigmaLancsozFT(kx,ky,kz,kcut):
    return sinc(kx/kcut[0])*sinc(ky/kcut[1])*sinc(kz/kcut[2])

def translationFT(kx,ky,kz,a):
    return exp(1.0j*(kx*a[0]+ky*a[1]+kz*a[2]))

def w3FT(k,sigma=1.0):
    return piecewise(k,[k<=1e-6,k>1e-6],[pi*sigma**3/6,lambda k: (pi*sigma**2/k)*spherical_jn(1,0.5*sigma*k)])

def w2FT(k,sigma=1.0):
    return pi*sigma**2*spherical_jn(0,0.5*sigma*k)

def wtensFT(k,sigma=1.0):
    return pi*sigma**2*spherical_jn(2,0.5*sigma*k)

def wtensoverk2FT(k,sigma=1.0):
    return piecewise(k,[k*sigma<=1e-3,k*sigma>1e-3],[pi*sigma**4/60,lambda k:(pi*sigma**2/k**2)*spherical_jn(2,0.5*sigma*k)])

@vectorize
def phi2func(eta):
    if eta <= 1e-3:
        return 1+eta**2/9
    else:
        return 1+(2*eta-eta**2+2*log(1-eta)*(1-eta))/(3*eta)
    
@vectorize
def phi3funcWBI(eta):
    if eta<=1e-3:
        return 1-2*eta/9-eta**2/18
    else: 
        return 2*(eta+log(1-eta)*(1-eta)**2)/(3*eta**2)

@vectorize
def phi3funcWBII(eta):
    if eta<=1e-3:
        return 1-4*eta/9 + eta**2/18
    else: 
        return 1-(2*eta-3*eta**2+2*eta**3+2*log(1-eta)*(1-eta)**2)/(3*eta**2)

@vectorize
def dphi2dnfunc(eta):
    if eta<=1e-3:
        return 2*eta/9+eta**2/6.0
    else: 
        return -(2*eta+eta**2+2*log(1-eta))/(3*eta**2)
    
@vectorize
def dphi3dnfuncWBI(eta):
    if eta<=1e-3:
        return -2/9-eta/9-eta**2/15.0
    else: 
        return (2*(eta-2)*eta+4*(eta-1)*log(1-eta))/(3*eta**3)

@vectorize
def dphi3dnfuncWBII(eta):
    if eta<=1e-3:
        return -4.0/9+eta/9+eta**2/15
    else: 
        return -2*(1-eta)*(eta*(2+eta)+2*log(1-eta))/(3*eta**3)