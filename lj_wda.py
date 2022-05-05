import numpy as np
from scipy.special import spherical_jn
# from scipy.fft import fftn, ifftn
import pyfftw
import multiprocessing
from pyfftw.interfaces.scipy_fftpack import fftn, ifftn
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-11-04
# Updated: 2020-11-05
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
print('Number of cpu cores:',multiprocessing.cpu_count())

" The Lennard Jones potential with the WDA approximation"

xlj = np.array([0.862308,2.976218,-8.402230,0.105413,-0.856458,1.582759,0.763942,1.753173,2.798e3,-4.839e-2,0.996326,-3.698e1,2.084e1,8.305e1,-9.574e2,-1.478e2,6.398e1,1.604e1,6.806e1,-2.791e3,-6.245128,-8.117e3,1.489e1,-1.059e4,-1.131607e2,-8.867771e3,-3.986982e1,-4.689270e3,2.593535e2,-2.694523e3,-7.218487e2,1.721802e2])

# defining the attractive potential 
def uatt(r,sigma):
    rc = 3*sigma
    return np.where(r<=sigma, 0.0, np.where(r<=rc,4*((sigma/r)**12-(sigma/r)**6)-4*((sigma/rc)**12-(sigma/rc)**6),0.0))

def weight(r,sigma):
    return np.where(r<=sigma,(3/(np.pi*sigma**3))*(1-r/sigma),0.0)

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

class LJPotential():
    def __init__(self,N,delta,sigma=1.0):
        self.N = N
        self.delta = delta
        self.L = delta*N
        self.sigma = sigma

        self.w_hat = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)
        self.phi_hat = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.complex64)
        self.rhobar = np.empty((self.N[0],self.N[1],self.N[2]),dtype=np.float32)

        x = np.linspace(-self.L[0]/2,self.L[0]/2,self.N[0])
        y = np.linspace(-self.L[1]/2,self.L[1]/2,self.N[1])
        z = np.linspace(-self.L[2]/2,self.L[2]/2,self.N[2])
        X,Y,Z = np.meshgrid(x,y,z)
        R = np.sqrt(X**2+Y**2+Z**2)
        u = uatt(R,self.sigma)
        w = weight(R,self.sigma)
        self.phi_hat[:] = fftn(u)*delta**3
        self.w_hat[:] = fftn(w)*delta**3
        print('w=',ifftn(self.w_hat).sum().real)
        del x, u, X, Y, Z, w

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

    def phi_integral(self,n_hat):
        return ifftn((n_hat-n_hat*self.w_hat)*self.phi_hat).real

    def free_energy(self,n_hat,beta):
        self.rhobar[:] = ifftn(n_hat*self.w_hat).real
        rho = ifftn(n_hat).real
        F1 = np.sum(rho*self.psi(self.rhobar,beta))*self.delta**3
        F2 = 0.5*np.sum((rho-self.rhobar)*self.phi_integral(n_hat))*self.delta**3
        return (F1+F2)
    
    def c1(self,n_hat,beta):
        self.rhobar[:] = ifftn(n_hat*self.w_hat).real
        # plt.imshow(self.rhobar[N//2].real, cmap='Greys_r')
        # plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
        # plt.xlabel('$x$')
        # plt.ylabel('$y$')
        # plt.show()
        rho = ifftn(n_hat).real
        aux = fftn((np.pi/6)*self.sigma**3*rho*self.dpsideta(self.rhobar,beta))
        c11 = beta*(-self.psi(self.rhobar,beta)-ifftn(aux*self.w_hat)).real
        c12 = -beta*ifftn((n_hat-n_hat*self.w_hat)*self.phi_hat*(1-self.w_hat)).real
        return (c11+c12)
        # aux = fftn((np.pi/6)*self.sigma**3*self.dpsideta(rhobar,beta))
        # return beta*(-self.psi(rhobar,beta)-rho*ifftn(aux*self.w_hat)).real


if __name__ == "__main__":
    test1 = False #plot the potential
    test2 = True # the phase diagram by DFT
    test3 = False # the liquid-vapor bulk phase diagram
    test4 = False # density profile
    test5 = False # the fluid-solid bulk phase diagram

    import matplotlib.pyplot as plt
    import sys
    sys.path.append("../PyDFT/")
    from fire import optimize_fire2
    from fmt import FMT

    if test1:
        l = 1.8
        N = 128
        delta = 0.05
        L = N*delta
        ykm = YKPotential(l,N,delta)

        r = np.linspace(0.0,ykm.L/2,N//2)
        rcut = np.linspace(1.0,ykm.L/2,N)

        w = ifftn(ykm.w_hat).real
        plt.plot(np.linspace(-ykm.L/2,ykm.L/2,N),w[N//2,N//2],label='IFT')
        # plt.plot(rcut,yukawa(rcut,l),'k',label='ana')
        # plt.plot(-rcut,yukawa(rcut,l),'k')
        plt.legend(loc='upper left')
        # plt.ylim(-1.2,0.1)
        plt.xlim(-L/2.0,L/2.0)
        plt.xlabel('$r/\\sigma$')
        plt.ylabel('$V(r)/\\epsilon$')
        # plt.savefig('yukawa_FT.png')
        plt.show()        

    if test2: 
        delta = 0.025
        N = np.array([64,64,64])
        L = N*delta

        FMT = FMT(N,delta)
        LJM = LJPotential(N,delta)

        muarray = np.linspace(-3.72,3.71,10,endpoint=True)

        Tarray = np.array([1.1])

        output = True

        n = np.ones((N[0],N[1],N[2]),dtype=np.float32)

        lnn1 = np.log(n) + np.log(0.05)
        lnn2 = np.log(n) + np.log(0.8)

        Vol = L[0]*L[1]*L[2]
        dV = delta[0]*delta[1]*delta[2]

        for j in range(Tarray.size):

            T = Tarray[j]
            print('#######################################')
            print('T=',T)
            print("mu\trho\trho2\tOmega1\tOmega2")

            beta = 1.0/T #1.0/0.004 # in kBT units
            betainv = T

            for i in range(muarray.size):

                mu = muarray[i]
                
                ## The Grand Canonical Potential
                def Omega(lnn,mu):
                    n[:] = np.float32(np.exp(lnn))
                    n_hat = fftn(n)
                    phi = FMT.Phi(n_hat)
                    FHS = betainv*np.sum(phi)*dV
                    Fid = betainv*np.sum(n*(lnn-1.0))*dV
                    Fykm = YKM.free_energy(n_hat,beta)
                    N = n.sum()*dV
                    return (Fid+FHS+Fykm-mu*N)/Vol

                def dOmegadnR(lnn,mu):
                    n[:] = np.float32(np.exp(lnn))
                    n_hat = fftn(n)
                    dphidn = FMT.dPhidn(n_hat)
                    c1YKM = YKM.c1(n_hat,beta)
                    return n*(betainv*(lnn)  + betainv*dphidn - betainv*c1YKM - mu)*delta**3/Vol

                [lnnsol,Omegasol,Niter] = optimize_fire2(lnn1,Omega,dOmegadnR,mu,1.0e-13,100.0,output)
                [lnnsol2,Omegasol2,Niter] = optimize_fire2(lnn2,Omega,dOmegadnR,mu,1.0e-13,0.1,output)

                rhomean = np.exp(lnnsol).sum()*delta**3/Vol 
                rhomean2 = np.exp(lnnsol2).sum()*delta**3/Vol

                print(mu,rhomean,rhomean2,Omegasol,Omegasol2)

                # np.save('ykm3d-densityfield1-lambda'+str(l)+'-T'+str(T)+'-mu'+str(mu)+'-N-'+str(N)+'.npy',np.exp(lnnsol))
                # np.save('ykm3d-densityfield2-lambda'+str(l)+'-T'+str(T)+'-mu'+str(mu)+'-N-'+str(N)+'.npy',np.exp(lnnsol2))

    if test3: 
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

    if test4: 
        l = 1.8
        L = 6.4
        N = 256
        delta = L/N

        FMT = FMT(N,delta)
        YKM = YKPotential(l,N,delta)

        # def mu_bulk(rhob):
        #     eta = (np.pi/6.0)*rhob
        #     a = YKM.integral_of_potential() #integral of Yukawa potential
        #     return (betainv*np.log(rhob) + betainv*(8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3) + a*rhob)

        print('N=',N)
        print('L=',L)

        #mu = 7.92 # mu_b = 7.620733333333334 #rho_b = 0.8 and T = 2.0
        # mu = -2.22 # mu_b = -2.2915 #rho_b = 0.8 and T = 1.0
        # mu = -2.745 # mu_b = -2.75946 #rho_b = 0.3 and T = 2.0
        rhob = 0.3 #rho_b = 0.8 and T = 1.0

        darray = [[7.92,0.8,2.0],[-2.22,0.8,1.0],[-2.745,0.3,2.0]]

        T = 2.0
        for i in range(3):
            [mu,rhob,T] = darray[i]

            print('################')
            print('T = ',T)
            print('rhob = ',rhob)
            beta = 1.0/T #1.0/0.004 # in kBT units
            betainv = T

            output = False

            n = 1.0e-16*np.ones((N,N,N),dtype=np.float32)
            Vext = np.zeros((N,N,N),dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        r2 = delta**2*((i-N/2)**2+(j-N/2)**2+(k-N/2)**2)
                        if r2>=1.0: 
                            n[i,j,k] = rhob*(1.0+ 0.1*np.random.randn())
                            Vext[i,j,k] = uatt((i-N/2)*delta,(j-N/2)*delta,(k-N/2)*delta,l,1.0)
            # n[:] = 1.0 + 0.1*np.random.randn(N,N,N)
            # n[:] = 1.0
            # plt.imshow(n[N//2].real, cmap='Greys_r')
            # plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
            # plt.xlabel('$x$')
            # plt.ylabel('$y$')
            # plt.show()

            lnn1 = np.log(n)

            del n

            Vol = L**3
                    
            ## The Grand Canonical Potential
            def Omega(lnn,mu):
                n = np.float32(np.exp(lnn))
                n_hat = fftn(n)
                phi = FMT.Phi(n_hat)
                FHS = (betainv)*np.sum(phi)*delta**3
                Fid = betainv*np.sum(n*(lnn-1.0))*delta**3
                Fykm = YKM.free_energy(n_hat,beta)
                N = n.sum()*delta**3
                Vextint = np.sum(Vext*n)*delta**3
                return (Fid+FHS+Fykm-mu*N+Vextint)/Vol

            def dOmegadnR(lnn,mu):
                n = np.float32(np.exp(lnn))
                n_hat = fftn(n)
                dphidn = FMT.dPhidn(n_hat)
                c1YKM = YKM.c1(n_hat,beta)
                return n*(betainv*(lnn)  + (betainv)*dphidn - betainv*c1YKM - mu + Vext)*delta**3/Vol

            [lnnsol,Omegasol,Niter] = optimize_fire2(lnn1,Omega,dOmegadnR,mu,1.0e-14,50.0,output)

            rho = np.exp(lnnsol)
            rhomean = rho.sum()*delta**3/Vol 

            print(mu,rhomean,Omegasol)

            # plt.imshow(rho[N//2].real/rhomean, cmap='Greys_r')
            # plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
            # plt.xlabel('$x$')
            # plt.ylabel('$y$')
            # plt.show()

            r = np.linspace(-L/2,L/2,N)
            np.save('ykm3d-radialdistribution-lambda'+str(l)+'-T'+str(T)+'-rho'+str(rhob)+'-N-'+str(N)+'.npy',[r,rho[N//2,N//2].real/rhomean])

    if test5: 
        l = 4.0
        L = 2*np.sqrt(2) #fcc unit cell
        a = np.sqrt(2)
        N = 64
        delta = L/N
        Vol = L**3

        Narray = np.array([N,N,N])
        deltaarray = np.array([delta,delta,delta])

        FMT = FMT(Narray,deltaarray)
        YKM = YKPotential(l,N,delta)

        print('The fluid-solid phase diagram using a gaussian parametrization')
        print('N=',N)
        print('L=',L)
        print('delta=',delta)

        # define the variables to the gaussian parametrization
        # the fcc lattice
        # lattice = a*np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,1],[0,0,-1],[0,0,1],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]])
        R = 0.5*a*np.array([[0,1,1],[1,0,1],[1,1,0]])
        def gaussian(alpha,x,y,z):
            # x = y = z = np.linspace(-L/2,L/2,N)
            rho = np.zeros((N,N,N),dtype=np.float32)
            for n1 in range(-2,3):
                for n2 in range(-2,3):
                    for n3 in range(-2,3):
                        rho += np.power(alpha/np.pi,1.5)*np.exp(-alpha*((x-n1*R[0,0]-n2*R[1,0]-n3*R[2,0])**2+(y-n1*R[0,1]-n2*R[1,1]-n3*R[2,1])**2+(z-n1*R[0,2]-n2*R[1,2]-n3*R[2,2])**2))
            return rho 

        n = np.empty((N,N,N),dtype=np.float32)
        n_hat = np.empty((N,N,N),dtype=np.complex64)

        alphacrystal = np.array([8.0])
        x = np.linspace(-L/2,L/2,N)
        X,Y,Z = np.meshgrid(x,x,x)
        
        n[:] = gaussian(alphacrystal,X,Y,Z)
        rhomean = n.sum()*delta**3/Vol 
        print(rhomean)
        nsig = int(0.5*1.0/delta)
        plt.imshow(n[N//2].real, cmap='viridis')
        plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.show()

        lnnsol = np.log(n)
        lnnflu = np.log(0.7)*np.ones((N,N,N),dtype=np.float32)

        muarray = np.linspace(5.0,10.0,10,endpoint=True)

        Tarray = np.array([0.6])

        output = True

        ## The Grand Canonical Potential
        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fftn(n)
            phi = FMT.Phi(n_hat)
            FHS = (betainv)*np.sum(phi)*delta**3
            Fid = betainv*np.sum(n*(lnn-1.0))*delta**3
            Fykm = YKM.free_energy(n_hat,beta)
            N = n.sum()*delta**3
            return (Fid+FHS+Fykm-mu*N)/Vol

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fftn(n)
            dphidn = FMT.dPhidn(n_hat)
            c1YKM = YKM.c1(n_hat,beta)
            return n*(betainv*lnn + betainv*dphidn - betainv*c1YKM - mu)*delta**3/Vol

        for j in range(Tarray.size):

            T = Tarray[j]
            print('#######################################')
            print('T=',T)
            print("mu\trho\trho2\tOmega1\tOmega2")

            beta = 1.0/T #1.0/0.004 # in kBT units
            betainv = T

            for i in range(muarray.size):

                mu = muarray[i]

                [lnn2,Omegasol2,Niter] = optimize_fire2(lnnsol,Omega,dOmegadnR,mu,6.0e-11,0.2,output)
                
                [lnn1,Omegasol,Niter] = optimize_fire2(lnnflu,Omega,dOmegadnR,mu,6.0e-11,0.2,output)
                

                rhomean = np.exp(lnn1).sum()*delta**3/Vol
                rhomean2 = np.exp(lnn2).sum()*delta**3/Vol

                # n[:] = gaussian(alpha2,X,Y,Z)
                # plt.imshow(n[0].real, cmap='Greys_r')
                # plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
                # plt.xlabel('$x$')
                # plt.ylabel('$y$')
                # plt.show()

                print(mu,rhomean,rhomean2,Omegasol,Omegasol2)