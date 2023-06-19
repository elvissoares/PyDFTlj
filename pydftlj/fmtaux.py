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

## phi1(n3)*n0
@vectorize
def phi1func(eta):
   return -log(1-eta)

@vectorize
def dphi1dnfunc(eta):
    return 1/(1-eta)

## phi2(n3)*(n1*n2-n1vec*n2vec)
@vectorize
def phi2func(eta):
    return 1/(1-eta)
    
@vectorize
def dphi2dnfunc(eta):
    return 1/(1-eta)**2

@vectorize
def phi2funcWBII(eta):
    if eta<=1e-3:
        return 1+eta+10*eta**2/9
    else: 
        return (1 + (2*eta - eta**2 + 2*(1-eta)*log(1-eta))/(3*eta))/(1-eta)
    
@vectorize
def dphi2dnfuncWBII(eta):
    if eta<=1e-3:
        return 1+20*eta/9+7*eta**2/2
    else: 
        return -2*(eta - 3*eta**2 + (1-eta)**2*log(1-eta))/(3*eta**2*(1-eta)**2)

## phi3(n3)*(n2**3-3*n2*n2vec*n2vec)  
@vectorize
def phi3func(eta):
    if eta<=1e-3:
        return (1+2*eta+3*eta**2)/(24*pi)
    else: 
        return 1/(24*pi*(1-eta)**2)
    
@vectorize
def dphi3dnfunc(eta):
    if eta<=1e-3:
        return (2+6*eta+12*eta**2)/(24*pi)
    else: 
        return 1/(12*pi*(1-eta)**3)
     
@vectorize
def phi3funcWBI(eta):
    if eta<=1e-3:
        return (1.5+8*eta/3+15*eta**2/4)/(36*pi)
    else: 
        return (eta+(1-eta)**2*log(1-eta))/(36*pi*eta**2*(1-eta)**2)
    
@vectorize
def dphi3dnfuncWBI(eta):
    if eta<=1e-3:
        return (8/3+7.5*eta+72*eta**2/5)/(36*pi)
    else: 
        return -(eta*(2-5*eta+eta**2)+2*(1-eta)**3*log(1-eta))/(36*pi*eta**3*(1-eta)**3)

@vectorize
def phi3funcWBII(eta):
    if eta<=1e-3:
        return (1.5+7*eta/3+13*eta**2/4)/(36*pi)
    else: 
        return -(eta+(eta-3)*eta**2+log(1-eta)*(1-eta)**2)/(36*pi*eta**2*(1-eta)**2)
    
@vectorize
def dphi3dnfuncWBII(eta):
    if eta<=1e-3:
        return (7/3+6.5*eta+63*eta**2/5)/(36*pi)
    else: 
        return -(eta*(-2+5*eta-6*eta**2+eta**3)-2*(1-eta)**3*log(1-eta))/(36*pi*eta**3*(1-eta)**3)