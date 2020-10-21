import numpy as np
from scipy.special import jv
# from scipy.fft import fft2, ifft2
import pyfftw
import multiprocessing
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-16
# Updated: 2020-09-18
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

def sigmaLancsozFT(kx,ky,kcut):
    return np.sinc(kx/kcut[0])*np.sinc(ky/kcut[1])

def translationFT(kx,ky,a):
    return np.exp(1.0j*(kx*a[0]+ky*a[1]))

def w2FT(k,sigma):
    return np.piecewise(k,[k<=1e-12,k>1e-12],[np.pi*sigma**2/4,lambda k: (np.pi*sigma/k)*jv(1,0.5*k*sigma)])

def w1FT(k,sigma):
    return np.pi*sigma*jv(0,0.5*k*sigma)

def w1tensFT(k,sigma):
    return np.piecewise(k,[k<=1e-6,k>1e-6],[np.pi*sigma**2/32,lambda k: (np.pi*sigma/k**2)*jv(2,0.5*k*sigma)])

# The Roth Functional
class RFMT2D():
    def __init__(self,N,L,sigma=1.0):
        self.dim = 2
        self.N = N
        self.L = L
        self.delta = L/N
        self.sigma = sigma

        self.w2_hat = np.empty((self.N[0],self.N[1]),dtype=np.complex64)
        self.w1_hat = np.empty((self.N[0],self.N[1]),dtype=np.complex64)
        self.w1vec_hat = np.empty((2,self.N[0],self.N[1]),dtype=np.complex64)
        self.w1tens_hat = np.empty((2,2,self.N[0],self.N[1]),dtype=np.complex64)

        self.n2 = np.empty((self.N[0],self.N[1]),dtype=np.float32)
        self.n1 = np.empty((self.N[0],self.N[1]),dtype=np.float32)
        self.n0 = np.empty((self.N[0],self.N[1]),dtype=np.float32)
        self.n1vec = np.empty((2,self.N[0],self.N[1]),dtype=np.float32)
        self.n1tens = np.empty((2,2,self.N[0],self.N[1]),dtype=np.float32)

        kx = np.fft.fftfreq(self.N[0], d=self.delta[0])*2*np.pi
        ky = np.fft.fftfreq(self.N[1], d=self.delta[1])*2*np.pi
        # kcut = np.pi/self.delta
        kcut = np.array([kx.max()*2/3,ky.max()*2/3])
        Kx,Ky = np.meshgrid(kx,ky,indexing ='ij')
        # k = np.array(np.meshgrid(kx,ky,indexing ='ij'), dtype=np.float32)
        # K2 = np.sum(k*k,axis=0, dtype=np.float32)
        K = np.sqrt(Kx**2+Ky**2)
        dealias = np.array((np.abs(Kx) < kcut[0] )*(np.abs(Ky) < kcut[1] ),dtype =bool)

        # self.w2_hat[:] = w2FT(K,self.sigma)*sigmaLancsozFT(Kx,Ky,kcut)*translationFT(Kx,Ky,0.5*self.L)

        # self.w1_hat[:] = w1FT(K,self.sigma)*sigmaLancsozFT(Kx,Ky,kcut)*translationFT(Kx,Ky,0.5*self.L)

        # w1tens_aux = w1tensFT(K,self.sigma)*sigmaLancsozFT(Kx,Ky,kcut)*translationFT(Kx,Ky,0.5*self.L)

        self.w2_hat[:] = w2FT(K,self.sigma)*dealias*translationFT(Kx,Ky,0.5*self.L)

        self.w1_hat[:] = w1FT(K,self.sigma)*dealias*translationFT(Kx,Ky,0.5*self.L)

        w1tens_aux = w1tensFT(K,self.sigma)*dealias*translationFT(Kx,Ky,0.5*self.L)

        self.w1vec_hat[0] = -1.0j*Kx*self.w2_hat
        self.w1vec_hat[1] = -1.0j*Ky*self.w2_hat
        self.w1tens_hat[0,0] = -Kx*Kx*w1tens_aux + 2*self.w2_hat/self.sigma
        self.w1tens_hat[0,1] = -Kx*Ky*w1tens_aux
        self.w1tens_hat[1,1] = -Ky*Ky*w1tens_aux + 2*self.w2_hat/self.sigma
        self.w1tens_hat[1,0] = -Ky*Kx*w1tens_aux

        del w1tens_aux, kx, ky, Kx, Ky, K, dealias

    def weighted_densities(self,n_hat):
        self.n2[:] = ifft2(n_hat*self.w2_hat).real
        self.n1[:] = ifft2(n_hat*self.w1_hat).real
        self.n1vec[0] = ifft2(n_hat*self.w1vec_hat[0]).real
        self.n1vec[1] = ifft2(n_hat*self.w1vec_hat[1]).real
        self.n1tens[0,0] = ifft2(n_hat*self.w1tens_hat[0,0]).real
        self.n1tens[0,1] = ifft2(n_hat*self.w1tens_hat[0,1]).real
        self.n1tens[1,1] = ifft2(n_hat*self.w1tens_hat[1,1]).real
        self.n1tens[1,0] = ifft2(n_hat*self.w1tens_hat[1,0]).real
        # plt.imshow(ifft2(self.w1vec_hat[0]).real)
        # plt.imshow(self.n2)
        # plt.colorbar()
        # plt.show()

        self.n0[:] = self.n1/(np.pi*self.sigma)
        self.oneminusn2 = 1-self.n2 

    def Phi(self,n_hat):
        self.weighted_densities(n_hat)
        return (-self.n0*np.log(self.oneminusn2)+((19/12.)*self.n1**2-(5.0/12.0)*(self.n1vec[0]*self.n1vec[0]+self.n1vec[1]*self.n1vec[1])-(7/6.)*(self.n1tens[0,0]*self.n1tens[0,0]+self.n1tens[0,1]*self.n1tens[0,1]+self.n1tens[1,1]*self.n1tens[1,1]+self.n1tens[1,0]*self.n1tens[1,0]))/(4*np.pi*self.oneminusn2)).real

    def c1_hat(self,n_hat):
        self.weighted_densities(n_hat)

        denom = 1.0/(24*np.pi*self.oneminusn2)
        self.dPhidn0 = fft2(-np.log(self.oneminusn2))
        self.dPhidn1 = fft2(19*self.n1*denom)
        self.dPhidn2 = fft2( self.n0/self.oneminusn2 + ((19/12.)*self.n1**2-(5.0/12.0)*(self.n1vec[0]*self.n1vec[0]+self.n1vec[1]*self.n1vec[1])-(7/6.)*(self.n1tens[0,0]*self.n1tens[0,0]+self.n1tens[0,1]*self.n1tens[0,1]+self.n1tens[1,1]*self.n1tens[1,1]+self.n1tens[1,0]*self.n1tens[1,0]))/(4*np.pi*self.oneminusn2**2))

        self.dPhidn1vec0 = fft2( -5*self.n1vec[0]*denom)
        self.dPhidn1vec1 = fft2( -5*self.n1vec[1]*denom)
        self.dPhidn1tens00 = fft2( -14*self.n1tens[0,0]*denom)
        self.dPhidn1tens01 = fft2( -28*self.n1tens[0,1]*denom)
        self.dPhidn1tens11 = fft2( -14*self.n1tens[1,1]*denom)

        self.dPhidn_hat = self.dPhidn2*self.w2_hat
        self.dPhidn_hat += self.dPhidn1*self.w1_hat + self.dPhidn0*self.w1_hat/np.pi
        self.dPhidn_hat += (-1.0)*(self.dPhidn1vec0*self.w1vec_hat[0]+self.dPhidn1vec1*self.w1vec_hat[1])

        self.dPhidn_hat += self.dPhidn1tens00*self.w1tens_hat[0,0]+self.dPhidn1tens01*self.w1tens_hat[0,1]+self.dPhidn1tens11*self.w1tens_hat[1,1]

        del self.dPhidn0,self.dPhidn1,self.dPhidn2,self.dPhidn1vec0,self.dPhidn1vec1,self.dPhidn1tens00 ,self.dPhidn1tens01,self.dPhidn1tens11
        
        return (-self.dPhidn_hat)

    def dPhidn(self,n_hat):
        return ifft2(-self.c1_hat(n_hat)).real


##### Take a example using FMT ######
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from fire import optimize_fire2

    #############################
    test1 = False
    test2 = False # slit-like pore
    test3 = False # radial distribution
    test4 = False # pressure
    test5 = True # crystalization phase diagram
    test6 = False # crystalization

    if test1:
        delta = 0.015625
        N = 1024
        L = N*delta
        eta = 0.4783
        fmt = RFMT2D(N,delta)

        w = ifft2(fmt.w2_hat)
        x = np.linspace(-L/2,L/2,N)
        y = np.linspace(-L/2,L/2,N)
        X, Y = np.meshgrid(x,y)

        print(w.real.sum(),np.pi/4)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        cmap = plt.get_cmap('hot')
        cp = ax.contourf(X, Y, w.real/np.max(w.real),100, cmap=cmap)
        ax.set_title(r'$\omega_2(r)=\Theta(\sigma/2-r)$')
        fig.colorbar(cp, ticks=[0, 0.2,0.4,0.6,0.8,1.0]) 
        ax.set_xlabel(r'$x/\sigma$')
        ax.set_ylabel(r'$y/\sigma$')
        fig.savefig('omega2-2d-N%d.png'% N, bbox_inches='tight')
        plt.show()

        w = ifft2(fmt.w1_hat)

        print(w.real.sum(),np.pi)
        cmap = plt.get_cmap('hot')
        cp2 = ax.contourf(X, Y, w.real/np.max(w.real),100, cmap=cmap)
        ax.set_title(r'$\omega_1(r)=\delta(\sigma/2-r)$')
        fig.colorbar(cp2, ticks=[0, 0.2,0.4,0.6,0.8,1.0]) 
        ax.set_xlabel(r'$x/\sigma$')
        ax.set_ylabel(r'$y/\sigma$')
        fig.savefig('omega1-2d-N%d.png'% N, bbox_inches='tight')
        plt.show()
        plt.close()

    ######################################################
    if test2:
        delta = 0.01
        N = 1024
        L = N*delta
        fmt = RFMT2D(N,delta)
        eta = 0.5
        rhob = eta/(np.pi/4.0)

        nsig = int(0.5/delta)

        n = np.asarray(rhob*(1+0.1*np.random.randn(N,N)),dtype=np.float32)
        n[:nsig,:] = 1.0e-12
        n[N-nsig:,:] = 1.0e-12
        n_hat = np.empty((N,N),dtype=np.complex64)

        lnn = np.log(n)

        z = np.linspace(-L/2,L/2,N)
            
        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fft2(n)
            phi = fmt.Phi(n_hat)
            Omegak = n*(lnn-1.0) + phi - mu*n
            return Omegak.sum()*delta**2/L**2

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fft2(n)
            dphidn = fmt.dPhidn(n_hat)
            return n*(lnn + dphidn - mu)*delta**2/L**2

        print("Doing the N=%d"% N) 
        mu = np.log(rhob) + (3*eta - 2*eta*eta)/((1-eta)**2) - np.log(1-eta)
        
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,1.0e-13,5.0,True)

        n[:] = np.exp(nsol)
        Nint = n.sum()*delta**2
        print('rhob=',rhob,'\t F/N =',(Omegasol*L**2)/N+mu)

        np.save('fmt2d-slitpore-eta'+str(eta)+'-N-'+str(N)+'.npy',[z,n[:,N//2]/rhob])

        # [zRF,rhoRF] = np.load('fmt2d-wbii-slitpore-eta'+str(eta)+'-N-'+str(N)+'.npy') 

        plt.plot(z,n[:,N//2]/rhob,label='RF')
        # plt.plot(zRF,rhoRF/rhob,label='WBII')
        plt.plot(z,((1+eta+eta**2)/(1-eta)**3)*np.ones(z.size),'--',color='grey')
        plt.xlabel(r'$x/\sigma$')
        plt.ylabel(r'$\rho(x)/\rho_b$')
        plt.xlim(-L/2,L/2)
        # plt.ylim(0.0,10)
        plt.legend()
        plt.savefig('slitpore2d-eta%.2f-N%d.pdf'% (eta,N), bbox_inches='tight')
        plt.show()
        plt.close()

    if test3:
        N = 512
        delta = 0.02
        L = N*delta
        fmt = RFMT2D(N,delta)
        eta = 0.735
        rhob = eta/(np.pi/4)

        n = 1e-16*np.ones((N,N),dtype=np.float32)
        n_hat = np.empty((N,N),dtype=np.complex64)
        for i in range(N):
            for j in range(N):
                r2 = delta**2*((i-N/2)**2+(j-N/2)**2)
                if r2>=1.0: n[i,j] = 1.0 + 0.1*np.random.randn()
        
        # plt.imshow(n.real, cmap='Greys_r')
        # plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
        # plt.xlabel('$x$')
        # plt.ylabel('$y$')
        # plt.show()

        lnn = np.log(n*rhob)
            
        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fft2(n)
            phi = fmt.Phi(n_hat)
            Omegak = n*(lnn-1.0) + phi - mu*n
            return Omegak.sum()*delta**2/L**2

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fft2(n)
            dphidn = fmt.dPhidn(n_hat)
            return n*(lnn + dphidn - mu)*delta**2/L**2

        print("Doing the N=%d"% N) 

        mu = np.log(rhob) + 3*eta/(1-eta) + (eta/(1-eta))**2 - np.log(1-eta)
        
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,1.0e-13,1.0,True)

        n[:] = np.exp(nsol.real)
        rhob = n.sum()*delta**2/(L**2-np.pi/4)
        print('eta = ',rhob*np.pi/4)

        z = np.linspace(0,L/2,N//2)

        np.save('radialdistribution-fmt2d-eta'+str(eta)+'-N-'+str(N)+'.npy',[z,n[N//2:,N//2]/rhob])

        plt.plot(z,n[N//2:,N//2]/rhob)
        plt.xlabel('$r$')
        plt.ylabel('$g(r)$')
        plt.savefig('radialdistribution-fmt2d-eta'+str(eta)+'-N-'+str(N)+'.pdf', bbox_inches='tight')
        plt.show()


    # Pressure
    if test4:
        N = 512
        L = np.array([10.24,0.5*np.sqrt(3)*10.24])
        delta = L/N
        fmt = RFMT2D(N,L)

        n = 1e-16*np.ones((N,N),dtype=np.float32)
        for i in range(N):
            for j in range(N):
                r2 = ((i-N/2)*delta[0])**2+((j-N/2)*delta[1])**2
                if r2>=1.0: n[i,j] = 1.0 + 0.1*np.random.randn()
        n_hat = np.empty((N,N),dtype=np.complex64)

        A = L[0]*L[1]
        dA = delta[0]*delta[1]

        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fft2(n)
            phi = fmt.Phi(n_hat)
            Omegak = n*(lnn-1.0) + phi - mu*n
            return Omegak.sum()*dA/A

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fft2(n)
            dphidn = fmt.dPhidn(n_hat)
            return n*(lnn + dphidn - mu)*dA/A

        print("Doing the N=%d"% N) 
        print(" eta PA/rhokBT")

        etaarray = np.arange(0.69,0.725,0.005)

        for eta in etaarray:

            rhob = eta/(np.pi/4)

            lnn = np.log(n*rhob)

            mu = np.log(rhob) + 3*eta/(1-eta) + (eta/(1-eta))**2 - np.log(1-eta)
        
            [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,1.0e-13,1.0,False)

            n[:] = np.exp(nsol.real)
            print(eta,(1+0.5*np.pi*n[N//2+50,N//2])*rhob)

            plt.imshow(n.real, cmap='Greys_r')
            plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.show()

    # fluid-solid transition
    if test5:
        delta = 0.025*np.ones(2)
        # L = 2*np.array([2,np.sqrt(3)])
        N = np.array([512,512])
        L = N*delta
        fmt = RFMT2D(N,L)

        n = np.empty((N[0],N[1]),dtype=np.float32)
        n_hat = np.empty((N[0],N[1]),dtype=np.complex64)

        A = L[0]*L[1]
        dA = delta[0]*delta[1]

        x = np.linspace(-L[0]/2,L[0]/2,N[0])
        y = np.linspace(-L[1]/2,L[1]/2,N[1])
        X,Y = np.meshgrid(x,y,indexing ='ij')

        # fcc lattice 
        # a1 = np.array([np.sqrt(2)*0.5,np.sqrt(2)*0.5])
        # a2 = np.array([np.sqrt(2)*0.5,-np.sqrt(2)*0.5])
        
        # honey comb lattice
        a1 = np.array([0.5,-0.5*np.sqrt(3)])
        a2 = np.array([0.5,0.5*np.sqrt(3)])
        def gaussian(alpha,rhob):
            rho = rhob*np.ones((N[0],N[1]),dtype=np.float32)
            for n1 in range(-1,2):
                for n2 in range(-1,2):
                    # if (abs(n1)>1) and (abs(n2)>1): R = n1*a1*(1.1) + n2*a2*(1.1)
                    R = n1*a1 + n2*a2
                    rho += (alpha/np.pi-rhob)*np.exp(-alpha*((X-R[0])**2+(Y-R[1])**2))
            rho -= (alpha/np.pi-rhob)*np.exp(-alpha*((X-0)**2+(Y-np.sqrt(3))**2))+(alpha/np.pi-rhob)*np.exp(-alpha*((X-0)**2+(Y+np.sqrt(3))**2))
            return rho

        n[:] = gaussian(4.0,0.1)
        # n[:] = np.abs(1.3*np.random.randn(N[0],N[1]))
        # n_hat = fft2(n)
        # kx = np.fft.fftfreq(N[0], d=delta[0])*2*np.pi
        # ky = np.fft.fftfreq(N[1], d=delta[1])*2*np.pi
        # kmax_dealias = np.pi
        # Kx,Ky = np.meshgrid(kx,ky,indexing ='ij')
        # dealias = np.array((np.abs(Kx) < kmax_dealias )*(np.abs(Ky) < kmax_dealias ),dtype =bool)
        # n_hat *= dealias
        # n[:] = ifft2(n_hat).real
        # n[:] += gaussian(4.0)

        print(n.mean())
        lnncr = np.log(n)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cp2 = ax.contourf(X, Y, n, 20,cmap='RdGy_r')
        cbar = fig.colorbar(cp2) 
        cbar.ax.set_title('$\\rho(x,y) \\sigma^2$')
        ax.set_xlabel('$x/\\sigma$')
        ax.set_ylabel('$y/\\sigma$')
        ax.set_aspect('equal')
        plt.show()
        plt.close()

        n[:] = (4.0/np.pi-0.1)*np.exp(-4.0*((X)**2+(Y)**2)) + 0.1 ##gaussian(4.0,0.1)
        # n[:] = 0.7 + 0.01*np.random.randn(N[0],N[1])
        # n_hatfl = fft2(n)
        # kx = np.fft.fftfreq(N[0], d=delta[0])*2*np.pi
        # ky = np.fft.fftfreq(N[1], d=delta[1])*2*np.pi
        # kmax_dealias = 2*np.pi
        # Kx,Ky = np.meshgrid(kx,ky,indexing ='ij')
        # dealias = np.array((np.abs(Kx) < kmax_dealias )*(np.abs(Ky) < kmax_dealias ),dtype =bool)
        # n_hatfl *= dealias
        # n[:] = ifft2(n_hatfl).real
        lnnfl = np.log(n)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cp2 = ax.contourf(X, Y, n, 20,cmap='RdGy_r')
        # cbar = fig.colorbar(cp2) 
        # cbar.ax.set_title('$\\rho(x,y) \\sigma^2$')
        # ax.set_xlabel('$x/\\sigma$')
        # ax.set_ylabel('$y/\\sigma$')
        # ax.set_aspect('equal')
        # plt.show()
        # plt.close()

        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fft2(n)
            phi = fmt.Phi(n_hat)
            Omegak = n*(lnn-1.0) + phi - mu*n
            return Omegak.sum()*dA/A

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fft2(n)
            dphidn = fmt.dPhidn(n_hat)
            return n*(lnn + dphidn - mu)*dA/A

        # def Omega(n_hat,mu):
        #     n[:] = ifft2(n_hat).real
        #     phi = fmt.Phi(n_hat)
        #     Omegak = n*(np.log(n)-1.0) + phi - mu*n
        #     return Omegak.sum()*dA/A

        # def dOmegadnR(n_hat,mu):
        #     n[:] = ifft2(n_hat).real
        #     dphidn = fmt.dPhidn(n_hat)
        #     return fft2((np.log(n) + dphidn - mu)*dA/A)

        print("============= Doing the fluid-solid transition line ===============") 
        print("eta mu nfl ncr Omegafl  Omegacr")
        
        eta = 0.69 #np.pi*np.sqrt(3)/6
        rhob = eta/(np.pi/4)
        mu1 = np.log(rhob) + 3*eta/(1-eta) + (eta/(1-eta))**2 - np.log(1-eta)
        eta = np.pi*np.sqrt(3)/6
        rhob = eta/(np.pi/4)
        mu2 = np.log(rhob) + 3*eta/(1-eta) + (eta/(1-eta))**2 - np.log(1-eta)

        # muarray = np.linspace(23.055,23.666,10)
        muarray = np.linspace(mu1,mu2,10)

        output = False

        for mu in muarray:
            # [nfl,Omegafl,Niter] = optimize_fire2(n_hatfl,Omega,dOmegadnR,mu,5.0e-6,1.0,output)
            # [ncr,Omegacr,Niter] = optimize_fire2(n_hat,Omega,dOmegadnR,mu,1.0e-6,1.0,output)
            [nfl,Omegafl,Niter] = optimize_fire2(lnnfl,Omega,dOmegadnR,mu,1.0e-12,0.1,output)
            [ncr,Omegacr,Niter] = optimize_fire2(lnncr,Omega,dOmegadnR,mu,1.0e-12,0.1,output)

            # ncr = ifft2(ncr).real
            ncr = np.exp(ncr)
            ncrmean = np.mean(ncr)
            nfl = np.exp(nfl)
            # nfl = ifft2(nfl).real
            nflmean = np.mean(nfl)

            print(nflmean*np.pi/4, mu, nflmean, ncrmean, Omegafl, Omegacr)

            # # np.save('crystal-fmt2d-eta'+str(eta)+'-N-'+str(N)+'.npy',[X,Y,n])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            cp = ax.contourf(X, Y, nfl, 20, cmap='RdGy_r')
            cbar = fig.colorbar(cp) 
            cbar.ax.set_title('$\\rho(x,y) \\sigma^2$')
            ax.set_xlabel('$x/\\sigma$')
            ax.set_ylabel('$y/\\sigma$')
            ax.set_aspect('equal')
            fig.savefig('densitymap-fluid-mu'+str(mu)+'.pdf')
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            cp = ax.contourf(X, Y, ncr, 20, cmap='RdGy_r')
            cbar = fig.colorbar(cp) 
            cbar.ax.set_title('$\\rho(x,y) \\sigma^2$')
            ax.set_xlabel('$x/\\sigma$')
            ax.set_ylabel('$y/\\sigma$')
            ax.set_aspect('equal')
            fig.savefig('densitymap-crystal-mu'+str(mu)+'.pdf')
            plt.close()

    if test6:
        delta = 0.01*np.ones(2)
        # L = 2*np.array([2,np.sqrt(3)])
        N = np.array([1024,1024])
        L = N*delta
        fmt = RFMT2D(N,L)

        n = np.empty((N[0],N[1]),dtype=np.float32)
        n_hat = np.empty((N[0],N[1]),dtype=np.complex64)

        A = L[0]*L[1]
        dA = delta[0]*delta[1]

        x = np.linspace(-L[0]/2,L[0]/2,N[0])
        y = np.linspace(-L[1]/2,L[1]/2,N[1])
        X,Y = np.meshgrid(x,y,indexing ='ij')

        # fcc lattice 
        a1 = np.array([np.sqrt(2)*0.5,np.sqrt(2)*0.5])
        a2 = np.array([np.sqrt(2)*0.5,-np.sqrt(2)*0.5])
        
        # honey comb lattice
        # a1 = np.array([0.5,-0.5*np.sqrt(3)])
        # a2 = np.array([0.5,0.5*np.sqrt(3)])
        def gaussian(alpha):
            rho = np.zeros((N[0],N[1]),dtype=np.float32)
            for n1 in range(-9,10):
                for n2 in range(-9,10):
                    R = n1*a1 + n2*a2
                    rho += alpha/np.pi*np.exp(-alpha*((X-R[0])**2+(Y-R[1])**2))
            return rho

        n[:] = gaussian(5.0)

        print(n.mean())
        lnncr = np.log(n)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cp2 = ax.contourf(X, Y, n, 20,cmap='RdGy_r')
        cbar = fig.colorbar(cp2) 
        cbar.ax.set_title('$\\rho(x,y) \\sigma^2$')
        ax.set_xlabel('$x/\\sigma$')
        ax.set_ylabel('$y/\\sigma$')
        ax.set_aspect('equal')
        plt.show()
        plt.close()

        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fft2(n)
            phi = fmt.Phi(n_hat)
            Omegak = n*(lnn-1.0) + phi 
            return Omegak.sum()*dA/(n.sum()*dA)
        
        print(Omega(lnncr,1.0))

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fft2(n)
            phi = fmt.Phi(n_hat)
            dphidn = fmt.dPhidn(n_hat)
            Omegak = n*(lnn-1.0) + phi 
            Nn = n.sum()*dA
            return n*((lnn + dphidn)-Omegak.sum()*dA/Nn)*dA/Nn

        print("============= Doing the fluid-solid transition line ===============") 
        print("mu ncr Omegafl  Omegacr")
        
        eta = 0.695 #np.pi*np.sqrt(3)/6
        rhob = eta/(np.pi/4)
        mu1 = np.log(rhob) + 3*eta/(1-eta) + (eta/(1-eta))**2 - np.log(1-eta)
        eta = 0.715
        rhob = eta/(np.pi/4)
        mu2 = np.log(rhob) + 3*eta/(1-eta) + (eta/(1-eta))**2 - np.log(1-eta)

        # muarray = np.linspace(23.055,23.666,10)
        muarray = np.linspace(mu1,mu2,10)

        output = True

        for mu in muarray:
            [ncr,Omegacr,Niter] = optimize_fire2(lnncr,Omega,dOmegadnR,mu,1.0e-12,1.0,output)

            # ncr = ifft2(ncr).real
            ncr = np.exp(ncr)
            ncrmean = np.mean(ncr)

            print(mu, ncrmean, Omegacr)

            # # np.save('crystal-fmt2d-eta'+str(eta)+'-N-'+str(N)+'.npy',[X,Y,n])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            cp = ax.contourf(X, Y, ncr, 20, cmap='RdGy_r')
            cbar = fig.colorbar(cp) 
            cbar.ax.set_title('$\\rho(x,y) \\sigma^2$')
            ax.set_xlabel('$x/\\sigma$')
            ax.set_ylabel('$y/\\sigma$')
            ax.set_aspect('equal')
            plt.show()
            plt.close()
