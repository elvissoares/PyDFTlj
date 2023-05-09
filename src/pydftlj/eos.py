import numpy as np
from scipy import optimize
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2023-02-23

def BHdiameter(kT,sigma=1.0,epsilon=1.0):
    kTstar = kT/epsilon
    return sigma*(1+0.2977*kTstar)/(1+0.33163*kTstar+1.0477e-3*kTstar**2)

# Reference: Johnson, J. K., Zollweg, J. A., & Gubbins, K. E. (1993). The Lennard-Jones equation of state revisited. Molecular Physics, 78(3), 591–618. https://doi.org/10.1080/00268979300100411
xljMBWR = np.array([0.862308,2.976218,-8.402230,0.105413,-0.856458,1.582759,0.763942,1.753173,2.798e3,-4.839e-2,0.996326,-3.698e1,2.084e1,8.305e1,-9.574e2,-1.478e2,6.398e1,1.604e1,6.806e1,-2.791e3,-6.245128,-8.117e3,1.489e1,-1.059e4,-1.131607e2,-8.867771e3,-3.986982e1,-4.689270e3,2.593535e2,-2.694523e3,-7.218487e2,1.721802e2])

# Reference: May, H.-O., & Mausbach, P. (2012). Riemannian geometry study of vapor-liquid phase equilibria and supercritical behavior of the Lennard-Jones fluid. Physical Review E, 85(3), 031201. https://doi.org/10.1103/PhysRevE.85.031201
xljMay12 = np.array([0.8623085097507421,2.976218765822098,-8.402230115796038,0.1054136629203555,-0.8564583828174598,1.44787318813706322,-0.310267527929454501,3.26700773856663408,4402.40210429518902,0.0165375389359225696,7.42150201869250559,-40.7967106914122298,16.4537825382141350,12.8389071227935610,-1407.06580259642897,-33.2251738947705988,17.8209627529619184,-331.646081795803070,331.495131943892488,-4399.44711295106300,-3.05878673562233238,-12849.6469455607240,9.96912508326940738,-16399.8349720621627,-256.926076715047884,-14588.020393359636,88.3082960748521799,-6417.29842088150144,121.307436784732417,-4461.88332740913756,-507.183302372831804,37.2385794546305178])

def acoef(kTstar,model='NewMBWR'):
    if model == 'MBWR':
        xlj = xljMBWR
    elif model == 'NewMBWR':
        xlj = xljMay12
    return np.array([xlj[0]*kTstar+xlj[1]*np.sqrt(kTstar)+xlj[2]+xlj[3]/kTstar+xlj[4]/kTstar**2,xlj[5]*kTstar+xlj[6]+xlj[7]/kTstar+xlj[8]/kTstar**2,xlj[9]*kTstar+xlj[10]+xlj[11]/kTstar,xlj[12],xlj[13]/kTstar+xlj[14]/kTstar**2,xlj[15]/kTstar,xlj[16]/kTstar+xlj[17]/kTstar**2,xlj[18]/kTstar**2])

def bcoef(kTstar,model='NewMBWR'):
    if model == 'MBWR':
        xlj = xljMBWR
    elif model == 'NewMBWR':
        xlj = xljMay12
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
    def __init__(self,sigma=1.0,epsilon=1.0,model='NewMBWR'):
        self.sigma = sigma
        self.epsilon = epsilon
        self.model = model
        
    # the excess free enery density
    def fexc(self,rho,kT):
        kTstar = kT/self.epsilon
        rhostar = rho*self.sigma**3
        a = acoef(kTstar,self.model)
        b = bcoef(kTstar,self.model)
        G = Gfunc(rhostar)
        fLJ = a[0]*rhostar + a[1]*rhostar**2/2+ a[2]*rhostar**3/3+a[3]*rhostar**4/4+a[4]*rhostar**5/5+a[5]*rhostar**6/6+a[6]*rhostar**7/7 + a[7]*rhostar**8/8
        fLJ += b[0]*G[0]+b[1]*G[1]+b[2]*G[2]+b[3]*G[3]+b[4]*G[4]+b[5]*G[5]
        return self.epsilon*rho*fLJ

    # the excess chemical potential
    def muexc(self,rho,kT):
        kTstar = kT/self.epsilon
        rhostar = rho*self.sigma**3
        a = acoef(kTstar,self.model)
        b = bcoef(kTstar,self.model)
        G = Gfunc(rhostar)
        dGdrhos = dGfuncdrhos(rhostar)
        fLJ = a[0]*rhostar + a[1]*rhostar**2/2+ a[2]*rhostar**3/3+a[3]*rhostar**4/4+a[4]*rhostar**5/5+a[5]*rhostar**6/6+a[6]*rhostar**7/7 + a[7]*rhostar**8/8
        fLJ += b[0]*G[0]+b[1]*G[1]+b[2]*G[2]+b[3]*G[3]+b[4]*G[4]+b[5]*G[5]
        dfLJdrhostar = a[0] + a[1]*rhostar+ a[2]*rhostar**2+a[3]*rhostar**3+a[4]*rhostar**4+a[5]*rhostar**5+a[6]*rhostar**6 + a[7]*rhostar**7
        dfLJdrhostar += b[0]*dGdrhos[0]+b[1]*dGdrhos[1]+b[2]*dGdrhos[2]+b[3]*dGdrhos[3]+b[4]*dGdrhos[4]+b[5]*dGdrhos[5]
        return self.epsilon*(fLJ+rhostar*dfLJdrhostar)

     # the excess pressure
    def pexc(self,rho,kT):
        return (-self.fexc(rho,kT)+self.muexc(rho,kT)*rho)

    # the derivative of the excess pressure
    def dpexcdrho(self,rho,kT):
        kTstar = kT/self.epsilon
        rhostar = rho*self.sigma**3
        a = acoef(kTstar,self.model)
        b = bcoef(kTstar,self.model)
        dGdrhos = dGfuncdrhos(rhostar)
        d2Gdrhos = d2Gfuncdrhos(rhostar)
        dfLJdrhostar = a[0] + a[1]*rhostar+ a[2]*rhostar**2+a[3]*rhostar**3+a[4]*rhostar**4+a[5]*rhostar**5+a[6]*rhostar**6 + a[7]*rhostar**7
        dfLJdrhostar += b[0]*dGdrhos[0]+b[1]*dGdrhos[1]+b[2]*dGdrhos[2]+b[3]*dGdrhos[3]+b[4]*dGdrhos[4]+b[5]*dGdrhos[5]
        d2fLJdrhostar = a[1]+ 2*a[2]*rhostar+3*a[3]*rhostar**2+4*a[4]*rhostar**3+5*a[5]*rhostar**4+6*a[6]*rhostar**5 + 7*a[7]*rhostar**6
        d2fLJdrhostar += b[0]*d2Gdrhos[0]+b[1]*d2Gdrhos[1]+b[2]*d2Gdrhos[2]+b[3]*d2Gdrhos[3]+b[4]*d2Gdrhos[4]+b[5]*d2Gdrhos[5]
        return self.epsilon*(2*dfLJdrhostar+rhostar*d2fLJdrhostar)*rhostar

    # the derivative of the excess pressure
    def d2pexcdrho2(self,rho,kT):
        kTstar = kT/self.epsilon
        rhostar = rho*self.sigma**3
        a = acoef(kTstar,self.model)
        b = bcoef(kTstar,self.model)
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
        return self.fexc(rho,kT) - kT*rho*(4*eta-3*eta**2)/((1-eta)**2)

    # the chemical potential
    def muatt(self,rho,kT):
        d = BHdiameter(kT,sigma=self.sigma,epsilon=self.epsilon)
        eta = rho*np.pi*d**3/6
        return self.muexc(rho,kT) - kT*(3*eta**3-9*eta**2+8*eta)/((1-eta)**3)

    def f(self,rho,kT):
        return kT*rho*(np.log(rho)-1) + self.fexc(rho,kT)

    def mu(self,rho,kT):
        return kT*np.log(rho) + self.muexc(rho,kT)

    def p(self,rho,kT):
        return kT*rho + self.pexc(rho,kT)
        
    def dpdrho(self,rho,kT): 
        return kT + self.dpexcdrho(rho,kT)

    def d2pdrho2(self,rho,kT): 
        return self.d2pexcdrho2(rho,kT)

# Objective function to critical point
def objective_cr(x,eos):
    [rho,kT] = x
    return [eos.dpdrho(rho,kT),eos.d2pdrho2(rho,kT)]

# Objective function to vapor-liquid equilibria
def objective(x,kT,eos):
    [rhov,rhol] = x
    return [eos.mu(rhol,kT)-eos.mu(rhov,kT),eos.p(rhol,kT)-eos.p(rhov,kT)]

# Vapor-liquid equilibrium
def Calculate_VaporLiquidEquilibria(eos,kTmin=0.7):
    solcr = optimize.root(objective_cr,[0.3/eos.sigma**3,1.3*eos.epsilon], args=(eos),method='lm')
    [rhoc,kTc] = solcr.x
    kTarray = np.array([kTc])
    rhovarray = np.array([rhoc])
    rholarray = np.array([rhoc])
    x = [0.8*rhoc,rhoc*1.2]
    rhol = rhoc
    kT = kTc
    while kT > kTmin*eos.epsilon:
        kT = kT - 0.001*(kTc-kTmin)
        sol = optimize.root(objective, x, args=(kT,eos),method='lm')
        [rhov,rhol] = sol.x
        x = sol.x
        kTarray=np.append(kTarray,kT)
        rhovarray=np.append(rhovarray,rhov)
        rholarray=np.append(rholarray,rhol)
    return [rhoc,kTc,np.hstack((rhovarray[::-1],rholarray)),np.hstack((kTarray[::-1],kTarray))]


