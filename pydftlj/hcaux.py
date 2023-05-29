from numba import vectorize
from numpy import pi, sinc, exp, log, piecewise

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