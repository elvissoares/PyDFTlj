import numpy as np
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-05-12
# Updated: 2022-06-08

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

" The Multiple Yukawa equation of state"

class YKEOS():
    def __init__(self,sigma=1.0,epsilon=np.array([1.0]),l=np.array([1.0]),method='FMSA'):
        self.sigma = sigma
        self.epsilon = epsilon
        self.l = l
        self.method = method

    # the free-energy density
    def f(self,rho,kT):
        eta = np.pi*rho*self.sigma**3/6
        beta = 1/kT
        a1 = np.zeros_like(rho)
        a2 = np.zeros_like(rho)
        for i in range(self.epsilon.size):
            a1 += -12*rho*eta*self.epsilon[i]*Lfunc(self.l[i],eta)/(self.l[i]**2*(1-eta)**2*Qfunc(self.l[i],eta))
            for j in range(self.epsilon.size):
                a2 += -6*eta*rho*beta*self.epsilon[i]*self.epsilon[j]/((self.l[i]+self.l[j])*Qfunc(self.l[i],eta)**2*Qfunc(self.l[j],eta)**2)
        return a1+a2

    # the chemical potential
    def mu(self,rho,kT):
        eta = np.pi*rho*self.sigma**3/6
        beta = 1/kT
        da1drho = np.zeros_like(rho)
        da2drho = np.zeros_like(rho)
        for i in range(self.epsilon.size):
            da1drho += -24*eta*self.epsilon[i]*Lfunc(self.l[i],eta)/(self.l[i]**2*(1-eta)**2*Qfunc(self.l[i],eta)) - 12*eta**2*self.epsilon[i]*dLdeta(self.l[i],eta)/(self.l[i]**2*(1-eta)**2*Qfunc(self.l[i],eta)) - 24*eta**2*self.epsilon[i]*Lfunc(self.l[i],eta)/(self.l[i]**2*(1-eta)**3*Qfunc(self.l[i],eta)) + 12*eta**2*self.epsilon[i]*Lfunc(self.l[i],eta)*dQdeta(self.l[i],eta)/(self.l[i]**2*(1-eta)**2*Qfunc(self.l[i],eta)**2)
            for j in range(self.epsilon.size):
                da2drho += -12*eta*beta*self.epsilon[i]*self.epsilon[j]/((self.l[i]+self.l[j])*Qfunc(self.l[i],eta)**2*Qfunc(self.l[j],eta)**2) + 12*eta**2*beta*self.epsilon[i]*self.epsilon[j]*dQdeta(self.l[i],eta)/((self.l[i]+self.l[j])*Qfunc(self.l[i],eta)**3*Qfunc(self.l[j],eta)**2)+ 12*eta**2*beta*self.epsilon[i]*self.epsilon[j]*dQdeta(self.l[j],eta)/((self.l[i]+self.l[j])*Qfunc(self.l[i],eta)**2*Qfunc(self.l[j],eta)**3)
        return da1drho+da2drho

    # the pressure
    def p(self,rho,kT):
        return (-self.f(rho,kT)+self.mu(rho,kT)*rho)

################################################################################
if __name__ == "__main__":
    test0 = False
    test1 = False # the compressibility factor plot
    test2 = False # attractive yukawa phase diagram
    test3 = True 
    test4 = False # lennard-jones phase diagram

    import matplotlib.pyplot as plt
    from fire import optimize_fire2

    if test0:
        eta = np.linspace(0.001,0.8,100)*np.pi/6

        plt.plot(eta,Sfunc(1.8,eta))
        plt.show()

    if test1:

        ykeos = YKEOS(sigma=1.0,epsilon=np.array([1.0]),l=np.array([1.8]),method='FMSA')

        def fhs(rho):
            eta = rho*np.pi/6
            return rho*(4*eta-3*eta**2)/((1-eta)**2)

        def muhs(rho):
            eta = rho*np.pi/6
            return (3*eta**3-9*eta**2+8*eta)/((1-eta)**3)

        def Zhs(rho):
            eta = rho*np.pi/6
            return (1+eta+eta**2-eta**3)/((1-eta)**3)

        def Z(n,kT):
            beta = 1.0/kT
            return (Zhs(n) + beta*ykeos.p(n,kT)/n)

        rho= np.linspace(0.001,0.9,100)

        rhoMC = np.array([0.1,0.3,0.4,0.5,0.6,0.7,0.8])
        Z1MC = np.array([0.621,0.021,-0.216,-0.295,-0.278,0.165,1.19])
        Z15MC = np.array([0.847,0.623,0.693,0.815,1.219,1.999,3.318])
        Z2MC = np.array([0.946,0.977,1.128,1.437,1.984,2.898,4.429])

        plt.scatter(rhoMC,Z1MC,marker='o',color='C0')
        plt.scatter(rhoMC,Z15MC,marker='o',color='C1')
        plt.scatter(rhoMC,Z2MC,marker='o',color='C2')
        plt.plot(rho,Z(rho,1.0),color='C0',label=r'$k T/\epsilon = 1.0$')
        plt.plot(rho,Z(rho,1.5),color='C1',label=r'$1.5$')
        plt.plot(rho,Z(rho,2.0),color='C2',label=r'$2.0$')
        plt.xlim(0,0.9)
        plt.ylim(-1,5)
        plt.legend(loc='best')
        plt.xlabel(r'$\rho \sigma^3$')
        plt.ylabel(r'$Z$')
        plt.show()

    if test2: 
        l = 1.8
        ykeos = YKEOS(sigma=1.0,epsilon=np.array([1.0]),l=np.array([l]),method='FMSA')

        def fhs(rho):
            eta = rho*np.pi/6
            return rho*(4*eta-3*eta**2)/((1-eta)**2)
        
        def muhs(rho):
            eta = rho*np.pi/6
            return (3*eta**3-9*eta**2+8*eta)/((1-eta)**3)

        kTarray = np.arange(1.244,1.26,0.004)

        filename = 'results/phasediagram_yukawa_l'+str(l)+'_FMSA.dat'

        def writeFile(kT,rhov,rhol,mu,omega):
            '''Writes the energy to a file.'''
            with open(filename, 'a') as f:
                f.write('{:.4f} {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(kT,rhov,rhol,mu,omega))

        rhomin = 0.20
        rhomax = 0.42

        output = False

        lnn1 = np.log(0.001)
        lnn2 = np.log(0.9)

        ## The Grand Canonical Potential
        def Omega(lnn,mu):
            n = np.exp(lnn)
            return (n*(lnn-1) + fhs(n) + beta*ykeos.f(n,kT) - mu*n)

        def dOmegadnR(lnn,mu):
            n = np.exp(lnn)
            return n*(lnn + muhs(n) + beta*ykeos.mu(n,kT) - mu)

        print('#######################################')
        print("kT\tmu\trho\trho2\tOmega1\tOmega2")

        for k in range(kTarray.size):

            kT = kTarray[k]
            beta = 1/kT
            
            mumin = np.log(rhomin) + beta*ykeos.mu(rhomin,kT) + muhs(rhomin)
            mumax = np.log(rhomax) + beta*ykeos.mu(rhomax,kT) + muhs(rhomax)
            # print(mumax,mumin)

            error = 1.0

            while error> 1.e-4:

                muarray = np.linspace(mumin,mumax,10)

                for i in range(muarray.size):

                    mu = muarray[i]

                    [lnnsol,Omegasol,Niter] = optimize_fire2(lnn1,Omega,dOmegadnR,mu,alpha0=0.2,rtol=1.0e-4,dt=0.1,logoutput=output)
                    [lnnsol2,Omegasol2,Niter] = optimize_fire2(lnn2,Omega,dOmegadnR,mu,alpha0=0.2,rtol=1.0e-4,dt=0.1,logoutput=output)

                    rhomean = np.exp(lnnsol)
                    rhomean2 = np.exp(lnnsol2)
                    # print(mu,rhomean,rhomean2,Omegasol,Omegasol2)

                    if (abs(rhomean2-rhomean)> 1.e-2):
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

    if test3: 
        sigma = 1.0
        epsilon = 1.0

        color = ['C0','C1','C2','C3']

        rhoMC = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.84])
        ZMCp75 = np.array([0.23,-0.29,-0.78,-1.2,-1.69,-2.05,-1.71,-0.53,0.37])

        plt.scatter(rhoMC,ZMCp75)

        rhoMC = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.75,0.85,0.92])
        ZMC1p15 = np.array([0.61,0.35,0.12,-0.09,-0.13,0.07,0.31,1.17,2.86,4.72])

        plt.scatter(rhoMC,ZMC1p15)

        rhoMC = np.array([0.1,0.2,0.3,0.4,0.55,0.7,0.8,0.9,1.0,1.08])
        ZMC2p74 = np.array([0.97,0.99,1.04,1.2,1.65,2.64,3.60,5.14,7.39,9.58])

        plt.scatter(rhoMC,ZMC2p74)

        rho = np.linspace(0.01,1.1,100)

        kTarray = np.array([0.75,1.15,1.35,2.74])

        def Zhs(rho):
                eta = rho*np.pi/6
                return (1+eta+eta**2-eta**3)/((1-eta)**3)

        for k in range(kTarray.size):

            kT = kTarray[k]/epsilon
            beta = 1.0/kT

            d = sigma*(1+0.2977*kT)/(1+0.33163*kT+0.0010477*kT**2)

            # parameters of Lennard-Jones potential described by two Yukawa
            
            l = np.array([2.6428,14.96767])*d/sigma
            eps = 1.94728*epsilon*(sigma/d)*np.array([1,-1])*np.exp(l*(sigma/d-1))
            ykeos = YKEOS(sigma=d,epsilon=eps,l=l,method='FMSA')

            plt.plot(rho,(Zhs(rho) + beta*ykeos.p(rho,kT)/rho),'--',color=color[k])

            l = np.array([2.9637,14.0167])*d/sigma
            eps = 2.1714*epsilon*(sigma/d)*np.array([1,-1])*np.exp(l*(sigma/d-1))
            ykeos2 = YKEOS(sigma=d,epsilon=eps,l=l,method='FMSA')

            plt.plot(rho,(Zhs(rho) + beta*ykeos2.p(rho,kT)/rho),color=color[k],label=r'$k_B T /\epsilon = $'+str(kT))
        
        plt.xlabel(r'$\rho \sigma^3$')
        plt.ylabel(r'$Z$')
        plt.ylim(-2.5,10)
        plt.xlim(0,1.1)
        plt.legend(loc='best')
        plt.show()

            

    if test4: 
        sigma = 1.0
        epsilon = 1.0

        kTarray = np.hstack((np.arange(0.70,1.2,0.01),np.arange(1.2,1.3,0.001)))

        filename = 'results/phasediagram_lennardjones_FMSA.dat'

        def writeFile(kT,rhov,rhol,mu,omega):
            '''Writes the energy to a file.'''
            with open(filename, 'a') as f:
                f.write('{:.4f} {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(kT,rhov,rhol,mu,omega))

        print('#######################################')
        print("kT\tmu\trho\trho2\tOmega1\tOmega2")

        rhomin = 0.001
        rhomax = 0.87

        for k in range(kTarray.size):

            kT = kTarray[k]/epsilon
            beta = 1.0/kT

            d = sigma*(1+0.2977*kT)/(1+0.33163*kT+0.0010477*kT**2)
            print(d)

            # parameters of Lennard-Jones potential described by two Yukawa
            l = np.array([2.9637,14.0167])*d/sigma
            eps = 2.1714*epsilon*(sigma/d)*np.array([1,-1])*np.exp(l*(sigma/d-1))
            rc = 5.0

            ykeos = YKEOS(sigma=d,epsilon=eps,l=l,method='FMSA')

            def fhs(rho):
                eta = rho*np.pi*d**3/6
                return rho*(4*eta-3*eta**2)/((1-eta)**2)
            
            def muhs(rho):
                eta = rho*np.pi*d**3/6
                return (3*eta**3-9*eta**2+8*eta)/((1-eta)**3)

            output = False

            lnn1 = np.log(0.001)
            lnn2 = np.log(0.9)

            ## The Grand Canonical Potential
            def Omega(lnn,mu):
                n = np.exp(lnn)
                return (n*(lnn-1) + fhs(n) + beta*ykeos.f(n,kT) - mu*n)

            def dOmegadnR(lnn,mu):
                n = np.exp(lnn)
                return n*(lnn + muhs(n) + beta*ykeos.mu(n,kT) - mu)
            
            mumin = np.log(rhomin) + beta*ykeos.mu(rhomin,kT) + muhs(rhomin)
            mumax = np.log(rhomax) + beta*ykeos.mu(rhomax,kT) + muhs(rhomax)

            error = 1.0

            while error> 1.e-4:

                muarray = np.linspace(mumin,mumax,4)

                for i in range(muarray.size):

                    mu = muarray[i]

                    [lnnsol,Omegasol,Niter] = optimize_fire2(lnn1,Omega,dOmegadnR,mu,alpha0=0.2,rtol=1.0e-4,dt=0.1,logoutput=output)
                    [lnnsol2,Omegasol2,Niter] = optimize_fire2(lnn2,Omega,dOmegadnR,mu,alpha0=0.2,rtol=1.0e-4,dt=0.1,logoutput=output)

                    rhomean = np.exp(lnnsol)
                    rhomean2 = np.exp(lnnsol2)
                    print(mu,rhomean,rhomean2,Omegasol,Omegasol2)

                    if (abs(rhomean2-rhomean)> 1.e-2):
                        if Omegasol>Omegasol2: 
                            error = min(error,abs(Omegasol2-Omegasol))
                            mumax = min(mumax,mu)
                        else: 
                            error = min(error,abs(Omegasol2-Omegasol))
                            mumin = max(mumin,mu)

            mu = (mumax+mumin)*0.5           
            [lnnsol,Omegasol,Niter] = optimize_fire2(lnn1,Omega,dOmegadnR,mu,rtol=1.0e-4,dt=0.1,logoutput=output)
            [lnnsol2,Omegasol2,Niter] = optimize_fire2(lnn2,Omega,dOmegadnR,mu,rtol=1.0e-4,dt=0.1,logoutput=output)

            rhov = np.exp(lnnsol)
            rhol = np.exp(lnnsol2)
            omega = (Omegasol+Omegasol2)*0.5
            error = abs(Omegasol-Omegasol2)
        
            # print('---------------------')
            print(kT,rhov,rhol,kT*mu,kT*omega)
            writeFile(kT,rhov,rhol,kT*mu,kT*omega)

            rhomin = rhov
            rhomax = rhol