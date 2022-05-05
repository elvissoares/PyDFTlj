import numpy as np
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-05-05

" The Lennard Jones potential with the WDA approximation"

xlj = np.array([0.862308,2.976218,-8.402230,0.105413,-0.856458,1.582759,0.763942,1.753173,2.798e3,-4.839e-2,0.996326,-3.698e1,2.084e1,8.305e1,-9.574e2,-1.478e2,6.398e1,1.604e1,6.806e1,-2.791e3,-6.245128,-8.117e3,1.489e1,-1.059e4,-1.131607e2,-8.867771e3,-3.986982e1,-4.689270e3,2.593535e2,-2.694523e3,-7.218487e2,1.721802e2])

def acoef(T):
    return np.array([xlj[0]*T+xlj[1]*np.sqrt(T)+xlj[2]+xlj[3]/T+xlj[4]/T**2,xlj[5]*T+xlj[6]+xlj[7]/T+xlj[8]/T**2,xlj[9]*T+xlj[10]+xlj[11]/T,xlj[12],xlj[13]/T+xlj[14]/T**2,xlj[15]/T,xlj[16]/T+xlj[17]/T**2,xlj[18]/T**2])

def bcoef(T):
    return np.array([xlj[19]/T**2+xlj[20]/T**3,xlj[21]/T**2+xlj[22]/T**4,xlj[23]/T**2+xlj[24]/T**3,xlj[25]/T**2+xlj[26]/T**4,xlj[27]/T**2+xlj[28]/T**3,xlj[29]/T**2+xlj[30]/T**3+xlj[31]/T**4])

def Gfunc(rhos,T):
    gamma = 3.0
    F = np.exp(-gamma*rhos**2)
    G1 = (1-F)/(2*gamma)
    G2 = -(F*rhos**2-2*G1)/(2*gamma)
    G3 = -(F*rhos**4-4*G2)/(2*gamma)
    G4 = -(F*rhos**6-6*G3)/(2*gamma)
    G5 = -(F*rhos**8-8*G4)/(2*gamma)
    G6 = -(F*rhos**10-10*G5)/(2*gamma)
    return np.array([G1,G2,G3,G4,G5,G6])

def dGfuncdrhos(rhos,T):
    gamma = 3.0
    F = np.exp(-gamma*rhos**2)
    dFdrhos = -2*gamma*rhos*F
    G1 = -dFdrhos/(2*gamma)
    G2 = -(dFdrhos*rhos**2+2*F*rhos-2*G1)/(2*gamma)
    G3 = -(dFdrhos*rhos**4+4*F*rhos**3-4*G2)/(2*gamma)
    G4 = -(dFdrhos*rhos**6+6*F*rhos**5-6*G3)/(2*gamma)
    G5 = -(dFdrhos*rhos**8+8*F*rhos**7-8*G4)/(2*gamma)
    G6 = -(dFdrhos*rhos**10+10*F*rhos**9-10*G5)/(2*gamma)
    return np.array([G1,G2,G3,G4,G5,G6])

class LJEOS():
    def __init__(self,sigma=1.0,epsilon=1.0,method='MBWR'):
        self.sigma = sigma
        self.epsilon = epsilon
        self.method = method

    def diameter(self,beta):
        c1=1.1287
        c2 = -0.05536
        c3=0.0007278
        return 2**(1/6)*self.sigma*(1+np.sqrt(1+(T+c2*T**2+c3*T**4)/c1))**(-1.0/6.0)

    # the free enery per particle
    def f(self,rho,beta):
        T = 1.0/beta
        rhos = rho
        a = acoef(T)
        b = bcoef(T)
        G = Gfunc(rhos,T)
        fLJ = a[0]*rhos + a[1]*rhos**2/2+ a[2]*rhos**3/3+a[3]*rhos**4/4+a[4]*rhos**5/5+a[5]*rhos**6/6+a[6]*rhos**7/7 + a[7]*rhos**8/8
        fLJ += b[0]*G[0]+b[1]*G[1]+b[2]*G[2]+b[3]*G[3]+b[4]*G[4]+b[5]*G[5]
        eta = np.pi*rho/6.0
        fCS = eta*(4-3*eta)/((1-eta)**2)
        return (fLJ - fCS/beta)

    def dfdrho(self,rho,beta):
        T = 1.0/beta
        rhos = rho
        a = acoef(T)
        b = bcoef(T)
        dGdrhos = dGfuncdrhos(rhos,T)
        dfLJdrho = a[0] + a[1]*rhos+ a[2]*rhos**2+a[3]*rhos**3+a[4]*rhos**4+a[5]*rhos**5+a[6]*rhos**6 + a[7]*rhos**7
        dfLJdrho += b[0]*dGdrhos[0]+b[1]*dGdrhos[1]+b[2]*dGdrhos[2]+b[3]*dGdrhos[3]+b[4]*dGdrhos[4]+b[5]*dGdrhos[5]
        dfLJdrho = dfLJdrho
        eta = np.pi*rho/6.0
        dfCSdrho = 2*(2-eta)*(np.pi/6.0)/((1-eta)**3)
        return (dfLJdrho - dfCSdrho/beta)

if __name__ == "__main__":
    test1 = True # the liquid-vapor bulk phase diagram

    import matplotlib.pyplot as plt
    from fire import optimize_fire2

    if test1: 
        N = 1
        delta = 0.05
        L = N*delta

        def fexcCS(n,sigma):
            eta = np.pi*n*sigma**3/6.0
            return n*eta*(4-3*eta)/((1-eta)**2)

        def dfexcCSdn(n,sigma):
            eta = np.pi*n*sigma**3/6.0
            return (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)

        print('N=',N)
        print('L=',L)

        muarray = np.linspace(-4.7352,-4.73518,10,endpoint=True)

        Tarray = np.array([0.5])

        output = False

        n = np.ones(N,dtype=np.float32)

        lnn1 = np.log(n) + np.log(0.05)
        lnn2 = np.log(n) + np.log(0.8)

        for j in range(Tarray.size):

            T = Tarray[j]
            print('#######################################')
            print('T=',T)
            print("mu\trho\trho2\tOmega1\tOmega2")

            beta = 1.0/T #1.0/0.004 # in kBT units
            betainv = T

            sigma=  (1+0.2977*T)/(1+0.33163*T+1.0477e-3*T**2)
            print("sigma = ",sigma)

            LJM = LJPotential(N,delta,sigma=sigma)

            for i in range(muarray.size):

                mu = muarray[i]
                
                ## The Grand Canonical Potential
                def Omega(lnn,mu):
                    n = np.exp(lnn)
                    Omegak = (betainv*n*(lnn-1) + betainv*fexcCS(n,1.0) + n*LJM.f(n,beta)- mu*n)
                    return Omegak.sum()

                def dOmegadnR(lnn,mu):
                    n = np.exp(lnn)
                    return n*(betainv*(lnn) + betainv*dfexcCSdn(n,1.0) + LJM.f(n,beta)+n*LJM.dfdrho(n,beta) - mu)

                [lnnsol,Omegasol,Niter] = optimize_fire2(lnn1,Omega,dOmegadnR,mu,1.0e-12,0.01,output)
                [lnnsol2,Omegasol2,Niter] = optimize_fire2(lnn2,Omega,dOmegadnR,mu,1.0e-12,0.01,output)

                rhomean = np.exp(lnnsol).sum()
                rhomean2 = np.exp(lnnsol2).sum()

                print(mu,rhomean,rhomean2,Omegasol,Omegasol2)