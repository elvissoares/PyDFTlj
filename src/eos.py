import numpy as np
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-05-05

def BHdiameter(kT,sigma=1.0,epsilon=1.0):
    kTstar = kT/epsilon
    return sigma*(1+0.2977*kTstar)/(1+0.33163*kTstar+1.0477e-3*kTstar**2)

xlj = np.array([0.862308,2.976218,-8.402230,0.105413,-0.856458,1.582759,0.763942,1.753173,2.798e3,-4.839e-2,0.996326,-3.698e1,2.084e1,8.305e1,-9.574e2,-1.478e2,6.398e1,1.604e1,6.806e1,-2.791e3,-6.245128,-8.117e3,1.489e1,-1.059e4,-1.131607e2,-8.867771e3,-3.986982e1,-4.689270e3,2.593535e2,-2.694523e3,-7.218487e2,1.721802e2])

def acoef(kTstar):
    return np.array([xlj[0]*kTstar+xlj[1]*np.sqrt(kTstar)+xlj[2]+xlj[3]/kTstar+xlj[4]/kTstar**2,xlj[5]*kTstar+xlj[6]+xlj[7]/kTstar+xlj[8]/kTstar**2,xlj[9]*kTstar+xlj[10]+xlj[11]/kTstar,xlj[12],xlj[13]/kTstar+xlj[14]/kTstar**2,xlj[15]/kTstar,xlj[16]/kTstar+xlj[17]/kTstar**2,xlj[18]/kTstar**2])

def bcoef(kTstar):
    return np.array([xlj[19]/kTstar**2+xlj[20]/kTstar**3,xlj[21]/kTstar**2+xlj[22]/kTstar**4,xlj[23]/kTstar**2+xlj[24]/kTstar**3,xlj[25]/kTstar**2+xlj[26]/kTstar**4,xlj[27]/kTstar**2+xlj[28]/kTstar**3,xlj[29]/kTstar**2+xlj[30]/kTstar**3+xlj[31]/kTstar**4])

def Gfunc(rhostar):
    gamma = 3.0
    F = np.exp(-gamma*rhostar**2)
    G1 = (1-F)/(2*gamma)
    G2 = -(F*rhostar**2-2*G1)/(2*gamma)
    G3 = -(F*rhostar**4-4*G2)/(2*gamma)
    G4 = -(F*rhostar**6-6*G3)/(2*gamma)
    G5 = -(F*rhostar**8-8*G4)/(2*gamma)
    G6 = -(F*rhostar**10-10*G5)/(2*gamma)
    return np.array([G1,G2,G3,G4,G5,G6])

def dGfuncdrhos(rhostar):
    gamma = 3.0
    F = np.exp(-gamma*rhostar**2)
    dFdrhos = -2*gamma*rhostar*F
    G1 = -dFdrhos/(2*gamma)
    G2 = -(dFdrhos*rhostar**2+2*F*rhostar-2*G1)/(2*gamma)
    G3 = -(dFdrhos*rhostar**4+4*F*rhostar**3-4*G2)/(2*gamma)
    G4 = -(dFdrhos*rhostar**6+6*F*rhostar**5-6*G3)/(2*gamma)
    G5 = -(dFdrhos*rhostar**8+8*F*rhostar**7-8*G4)/(2*gamma)
    G6 = -(dFdrhos*rhostar**10+10*F*rhostar**9-10*G5)/(2*gamma)
    return np.array([G1,G2,G3,G4,G5,G6])

def d2Gfuncdrhos(rhostar):
    gamma = 3.0
    F = np.exp(-gamma*rhostar**2)
    dFdrhos = -2*gamma*rhostar*F
    d2Fdrhos = 2*gamma*(-1+2*gamma*rhostar**2)*F
    G1 = -d2Fdrhos/(2*gamma)
    G2 = -(d2Fdrhos*rhostar**2+4*dFdrhos*rhostar+2*F-2*G1)/(2*gamma)
    G3 = -(d2Fdrhos*rhostar**4+8*dFdrhos*rhostar**3+12*F*rhostar**2-4*G2)/(2*gamma)
    G4 = -(d2Fdrhos*rhostar**6+12*dFdrhos*rhostar**5+30*F*rhostar**4-6*G3)/(2*gamma)
    G5 = -(d2Fdrhos*rhostar**8+16*dFdrhos*rhostar**7+56*F*rhostar**6-8*G4)/(2*gamma)
    G6 = -(d2Fdrhos*rhostar**10+20*dFdrhos*rhostar**9+90*F*rhostar**8-10*G5)/(2*gamma)
    return np.array([G1,G2,G3,G4,G5,G6])

def d3Gfuncdrhos(rhostar):
    gamma = 3.0
    F = np.exp(-gamma*rhostar**2)
    dFdrhos = -2*gamma*rhostar*F
    d2Fdrhos = 2*gamma*(-1+2*gamma*rhostar**2)*F
    d3Fdrhos = 4*gamma**2*rhostar*(3-2*gamma*rhostar**2)*F
    G1 = -d3Fdrhos/(2*gamma)
    G2 = -(d3Fdrhos*rhostar**2+6*d2Fdrhos*rhostar+6*dFdrhos-2*G1)/(2*gamma)
    G3 = -(d3Fdrhos*rhostar**4+12*d2Fdrhos*rhostar**3+36*dFdrhos*rhostar**2+24*F*rhostar-4*G2)/(2*gamma)
    G4 = -(d3Fdrhos*rhostar**6+18*d2Fdrhos*rhostar**5+90*dFdrhos*rhostar**4+120*F*rhostar**3-6*G3)/(2*gamma)
    G5 = -(d3Fdrhos*rhostar**8+24*d2Fdrhos*rhostar**7+168*dFdrhos*rhostar**6+336*F*rhostar**5-8*G4)/(2*gamma)
    G6 = -(d3Fdrhos*rhostar**10+30*d2Fdrhos*rhostar**9+270*dFdrhos*rhostar**8+720*F*rhostar**7-10*G5)/(2*gamma)
    return np.array([G1,G2,G3,G4,G5,G6])

" The Lennard Jones equation of state"

###############################################################
class LJEOS():
    def __init__(self,sigma=1.0,epsilon=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
        
    # the free enery density
    def f(self,rho,kT):
        kTstar = kT/self.epsilon
        rhostar = rho*self.sigma**3
        a = acoef(kTstar)
        b = bcoef(kTstar)
        G = Gfunc(rhostar)
        fLJ = a[0]*rhostar + a[1]*rhostar**2/2+ a[2]*rhostar**3/3+a[3]*rhostar**4/4+a[4]*rhostar**5/5+a[5]*rhostar**6/6+a[6]*rhostar**7/7 + a[7]*rhostar**8/8
        fLJ += b[0]*G[0]+b[1]*G[1]+b[2]*G[2]+b[3]*G[3]+b[4]*G[4]+b[5]*G[5]
        return self.epsilon*rho*fLJ

    # the chemical potential
    def mu(self,rho,kT):
        kTstar = kT/self.epsilon
        rhostar = rho*self.sigma**3
        a = acoef(kTstar)
        b = bcoef(kTstar)
        G = Gfunc(rhostar)
        dGdrhos = dGfuncdrhos(rhostar)
        fLJ = a[0]*rhostar + a[1]*rhostar**2/2+ a[2]*rhostar**3/3+a[3]*rhostar**4/4+a[4]*rhostar**5/5+a[5]*rhostar**6/6+a[6]*rhostar**7/7 + a[7]*rhostar**8/8
        fLJ += b[0]*G[0]+b[1]*G[1]+b[2]*G[2]+b[3]*G[3]+b[4]*G[4]+b[5]*G[5]
        dfLJdrhostar = a[0] + a[1]*rhostar+ a[2]*rhostar**2+a[3]*rhostar**3+a[4]*rhostar**4+a[5]*rhostar**5+a[6]*rhostar**6 + a[7]*rhostar**7
        dfLJdrhostar += b[0]*dGdrhos[0]+b[1]*dGdrhos[1]+b[2]*dGdrhos[2]+b[3]*dGdrhos[3]+b[4]*dGdrhos[4]+b[5]*dGdrhos[5]
        return self.epsilon*(fLJ+rhostar*dfLJdrhostar)

     # the pressure
    def p(self,rho,kT):
        return (-self.f(rho,kT)+self.mu(rho,kT)*rho)

    # the derivative of the pressure
    def dpdrho(self,rho,kT):
        kTstar = kT/self.epsilon
        rhostar = rho*self.sigma**3
        a = acoef(kTstar)
        b = bcoef(kTstar)
        dGdrhos = dGfuncdrhos(rhostar)
        d2Gdrhos = d2Gfuncdrhos(rhostar)
        dfLJdrhostar = a[0] + a[1]*rhostar+ a[2]*rhostar**2+a[3]*rhostar**3+a[4]*rhostar**4+a[5]*rhostar**5+a[6]*rhostar**6 + a[7]*rhostar**7
        dfLJdrhostar += b[0]*dGdrhos[0]+b[1]*dGdrhos[1]+b[2]*dGdrhos[2]+b[3]*dGdrhos[3]+b[4]*dGdrhos[4]+b[5]*dGdrhos[5]
        d2fLJdrhostar = a[1]+ 2*a[2]*rhostar+3*a[3]*rhostar**2+4*a[4]*rhostar**3+5*a[5]*rhostar**4+6*a[6]*rhostar**5 + 7*a[7]*rhostar**6
        d2fLJdrhostar += b[0]*d2Gdrhos[0]+b[1]*d2Gdrhos[1]+b[2]*d2Gdrhos[2]+b[3]*d2Gdrhos[3]+b[4]*d2Gdrhos[4]+b[5]*d2Gdrhos[5]
        return self.epsilon*(2*dfLJdrhostar+rhostar*d2fLJdrhostar)*rhostar

    # the derivative of the pressure
    def d2pdrho2(self,rho,kT):
        kTstar = kT/self.epsilon
        rhostar = rho*self.sigma**3
        a = acoef(kTstar)
        b = bcoef(kTstar)
        dGdrhos = dGfuncdrhos(rhostar)
        d2Gdrhos = d2Gfuncdrhos(rhostar)
        d3Gdrhos = d3Gfuncdrhos(rhostar)
        dfLJdrhostar = a[0] + a[1]*rhostar+ a[2]*rhostar**2+a[3]*rhostar**3+a[4]*rhostar**4+a[5]*rhostar**5+a[6]*rhostar**6 + a[7]*rhostar**7
        dfLJdrhostar += b[0]*dGdrhos[0]+b[1]*dGdrhos[1]+b[2]*dGdrhos[2]+b[3]*dGdrhos[3]+b[4]*dGdrhos[4]+b[5]*dGdrhos[5]
        d2fLJdrhostar = a[1]+ 2*a[2]*rhostar+3*a[3]*rhostar**2+4*a[4]*rhostar**3+5*a[5]*rhostar**4+6*a[6]*rhostar**5 + 7*a[7]*rhostar**6
        d2fLJdrhostar += b[0]*d2Gdrhos[0]+b[1]*d2Gdrhos[1]+b[2]*d2Gdrhos[2]+b[3]*d2Gdrhos[3]+b[4]*d2Gdrhos[4]+b[5]*d2Gdrhos[5]
        d3fLJdrhostar = 2*a[2]+6*a[3]*rhostar+12*a[4]*rhostar**2+20*a[5]*rhostar**3+30*a[6]*rhostar**4 + 42*a[7]*rhostar**5
        d3fLJdrhostar += b[0]*d3Gdrhos[0]+b[1]*d3Gdrhos[1]+b[2]*d3Gdrhos[2]+b[3]*d3Gdrhos[3]+b[4]*d3Gdrhos[4]+b[5]*d3Gdrhos[5]
        return self.epsilon*((3*d2fLJdrhostar+rhostar*d3fLJdrhostar)*rhostar+2*dfLJdrhostar+rhostar*d2fLJdrhostar)*self.sigma**3

    # the free enery density
    def fatt(self,rho,kT):
        d = BHdiameter(kT,sigma=self.sigma,epsilon=self.epsilon)
        eta = rho*np.pi*d**3/6
        return self.f(rho,kT) - kT*rho*(4*eta-3*eta**2)/((1-eta)**2)

    # the chemical potential
    def muatt(self,rho,kT):
        d = BHdiameter(kT,sigma=self.sigma,epsilon=self.epsilon)
        eta = rho*np.pi*d**3/6
        return self.mu(rho,kT) - kT*(3*eta**3-9*eta**2+8*eta)/((1-eta)**3)

    