import numpy as np
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-05-05

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

" The Lennard Jones equation of state"

###############################################################
class LJEOS():
    def __init__(self,sigma=1.0,epsilon=1.0,method='MBWR'):
        self.sigma = sigma
        self.epsilon = epsilon
        self.method = method

    def diameter(self,kT):
        kTstar = kT/self.epsilon
        return (1+0.2977*kTstar)/(1+0.33163*kTstar+1.0477e-3*kTstar**2)

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
        dfLJdrho = a[0] + a[1]*rhostar+ a[2]*rhostar**2+a[3]*rhostar**3+a[4]*rhostar**4+a[5]*rhostar**5+a[6]*rhostar**6 + a[7]*rhostar**7
        dfLJdrho += b[0]*dGdrhos[0]+b[1]*dGdrhos[1]+b[2]*dGdrhos[2]+b[3]*dGdrhos[3]+b[4]*dGdrhos[4]+b[5]*dGdrhos[5]
        return self.epsilon*(fLJ+rhostar*dfLJdrho)

    # the free enery density
    def fatt(self,rho,kT):
        d = self.diameter(kT)
        eta = rho*np.pi*d**3/6
        return self.f(rho,kT) - kT*rho*(4*eta-3*eta**2)/((1-eta)**2)

    # the chemical potential
    def muatt(self,rho,kT):
        d = self.diameter(kT)
        eta = rho*np.pi*d**3/6
        return self.mu(rho,kT) - kT*(3*eta**3-9*eta**2+8*eta)/((1-eta)**3)

     # the pressure
    def p(self,rho,kT):
        return (-self.f(rho,kT)+self.mu(rho,kT)*rho)

if __name__ == "__main__":
    test1 = False # the liquid-vapor bulk phase diagram
    test2 = True

    import matplotlib.pyplot as plt
    from fire import optimize_fire2

    if test1: 

        ljeos = LJEOS(sigma=1.0,epsilon=1.0,method='MBWR')

        kTarray = np.arange(1.299,1.313,0.001)

        filename = 'results/phasediagram_lennardjones_MBWR.dat'

        def writeFile(kT,rhov,rhol,mu,omega):
            '''Writes the energy to a file.'''
            with open(filename, 'a') as f:
                f.write('{:.3f} {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(kT,rhov,rhol,mu,omega))

        rhomin = 0.2027
        rhomax = 0.4279

        output = False

        lnn1 = np.log(0.01)
        lnn2 = np.log(0.9)

        print('#######################################')
        print("kT\tmu\trho\trho2\tOmega1\tOmega2")

        for k in range(kTarray.size):

            kT = kTarray[k]
            beta = 1/kT
            
            mumin = np.log(rhomin) + beta*ljeos.mu(rhomin,kT) 
            mumax = np.log(rhomax) + beta*ljeos.mu(rhomax,kT) 
            # print(mumax,mumin)

            ## The Grand Canonical Potential
            def Omega(lnn,mu):
                n = np.exp(lnn)
                return (n*(lnn-1) + beta*ljeos.f(n,kT) - mu*n)

            def dOmegadnR(lnn,mu):
                n = np.exp(lnn)
                return n*(lnn + beta*ljeos.mu(n,kT) - mu)

            error = 1.0

            while error> 1.e-4:

                muarray = np.linspace(mumin,mumax,5)

                for i in range(muarray.size):

                    mu = muarray[i]

                    [lnnsol,Omegasol,Niter] = optimize_fire2(lnn1,Omega,dOmegadnR,mu,alpha0=0.2,rtol=1.0e-4,dt=0.1,logoutput=output)
                    [lnnsol2,Omegasol2,Niter] = optimize_fire2(lnn2,Omega,dOmegadnR,mu,alpha0=0.2,rtol=1.0e-4,dt=0.1,logoutput=output)

                    rhomean = np.exp(lnnsol)
                    rhomean2 = np.exp(lnnsol2)
                    # print(mu,rhomean,rhomean2,Omegasol,Omegasol2)

                    if (abs(rhomean2-rhomean)> 1.e-3):
                        if Omegasol>Omegasol2: 
                            error = min(error,abs(Omegasol2-Omegasol))
                            mumax = min(mumax,mu)
                        else: 
                            error = min(error,abs(Omegasol2-Omegasol))
                            mumin = max(mumin,mu)
                
                # print(mumax,mumin,error)

            mu = (mumax+mumin)*0.5           
            [lnnsol,Omegasol,Niter] = optimize_fire2(lnn1,Omega,dOmegadnR,mu,rtol=1.0e-4,dt=0.1,logoutput=output)
            [lnnsol2,Omegasol2,Niter] = optimize_fire2(lnn2,Omega,dOmegadnR,mu,rtol=1.0e-4,dt=0.1,logoutput=output)

            rhov = np.exp(lnnsol)
            rhol = np.exp(lnnsol2)
            omega = (Omegasol+Omegasol2)*0.5
            error = abs(Omegasol-Omegasol2)
        
            # print('---------------------')
            print(kT,rhov,rhol,mu,omega)
            writeFile(kT,rhov,rhol,mu,omega)

            rhomin = rhov
            rhomax = rhol
    
    if test2:

        epsilon = 36.7 # kelvin
        sigma = 2.96 # angstrom

        ljeos = LJEOS(sigma=sigma,epsilon=epsilon,method='MBWR')

        rhostar = np.arange(0.1,2.0,0.1)

        kT =  77 # kelvin
        beta = 1.0/kT

        pstar = (kT*rhostar+ljeos.p(rhostar/sigma**3,kT)*sigma**3)/epsilon

        for i in range(rhostar.size):
            print(rhostar[i],pstar[i],1e-5*pstar[i]*(1.38e-23*epsilon)/(sigma*1e-9)**3)

        plt.plot(rhostar,pstar,label='77 K')

        kT =  243 # kelvin
        beta = 1.0/kT

        pstar = (kT*rhostar+ljeos.p(rhostar/sigma**3,kT)*sigma**3)/epsilon
        plt.plot(rhostar,pstar,label='243 K')

        kT =  298 # kelvin
        beta = 1.0/kT

        pstar = (kT*rhostar+ljeos.p(rhostar/sigma**3,kT)*sigma**3)/epsilon
        plt.plot(rhostar,pstar,label='298 K')

        plt.legend(loc='best')
        plt.xlim(0,2)
        plt.ylim(0,1000)
        plt.xlabel(r'$\rho \sigma^3$')
        plt.ylabel(r'$P \sigma^3/\epsilon$')
        plt.show()
