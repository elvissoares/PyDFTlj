import numpy as np
from scipy.special import jv
# from scipy.fft import fft2, ifft2
import pyfftw
import multiprocessing
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-16
# Updated: 2020-09-04

def sigmaLancsozFT(kx,ky,kcut):
    return np.sinc(kx/kcut)*np.sinc(ky/kcut)

def translationFT(kx,ky,a):
    return np.exp(1.0j*(kx+ky)*a)

def w2FT(kx,ky,sigma):
    k = np.sqrt(kx**2 + ky**2)
    return np.piecewise(k,[k<=1e-6,k>1e-6],[np.pi*sigma**2/4,lambda k: (np.pi*sigma/k)*jv(1,0.5*k*sigma)])

def w1FT(kx,ky,sigma):
    k = np.sqrt(kx**2 + ky**2)
    return np.pi*sigma*jv(0,0.5*k)

def w1tensFT(kx,ky,sigma):
    k = np.sqrt(kx**2 + ky**2)
    return np.piecewise(k,[k<=1e-6,k>1e-6],[sigma*2*np.pi/32,lambda k: (np.pi*sigma/k**2)*jv(2,0.5*k)])

# The Roth Functional
# The spatial generation of the wweiths is ok!
class RFMT2D():
    def __init__(self,N,delta,sigma=1.0):
        self.dim = 2
        self.N = N
        self.delta = delta
        self.L = N*delta
        self.sigma = sigma

        self.w2_hat = np.zeros((self.N,self.N),dtype=np.complex64)
        self.w1_hat = np.zeros((self.N,self.N),dtype=np.complex64)
        self.w1vec_hat = np.zeros((2,self.N,self.N),dtype=np.complex64)
        self.w1tens_hat = np.zeros((2,2,self.N,self.N),dtype=np.complex64)

        self.n2 = np.zeros((self.N,self.N),dtype=np.float32)
        self.n1 = np.zeros((self.N,self.N),dtype=np.float32)
        self.n1vec = np.zeros((2,self.N,self.N),dtype=np.float32)
        self.n1tens = np.zeros((2,2,self.N,self.N),dtype=np.float32)

        kx = np.fft.fftfreq(self.N, d=self.delta)*2*np.pi
        ky = np.fft.fftfreq(self.N, d=self.delta)*2*np.pi
        kcut = np.pi/self.delta
        Kx,Ky = np.meshgrid(kx,ky)

        self.w2_hat = w2FT(Kx,Ky,self.sigma)*sigmaLancsozFT(Kx,Ky,kcut)*translationFT(Kx,Ky,0.5*self.L)

        self.w1_hat = w1FT(Kx,Ky,self.sigma)*sigmaLancsozFT(Kx,Ky,kcut)*translationFT(Kx,Ky,0.5*self.L)

        w1tens_aux = w1tensFT(Kx,Ky,self.sigma)*sigmaLancsozFT(Kx,Ky,kcut)*translationFT(Kx,Ky,0.5*self.L)

        self.w1vec_hat[0] = -1.0j*Kx*self.w2_hat
        self.w1vec_hat[1] = -1.0j*Ky*self.w2_hat
        self.w1tens_hat[0,0] = -Kx*Kx*w1tens_aux + 2*self.w2_hat/self.sigma
        self.w1tens_hat[0,1] = -Kx*Ky*w1tens_aux
        self.w1tens_hat[1,1] = -Ky*Ky*w1tens_aux + 2*self.w2_hat/self.sigma
        self.w1tens_hat[1,0] = -Ky*Kx*w1tens_aux

    def weighted_densities(self,n_hat):
        self.n2 = ifft2(n_hat*self.w2_hat).real
        self.n1 = ifft2(n_hat*self.w1_hat).real
        self.n1vec[0] = ifft2(n_hat*self.w1vec_hat[0]).real
        self.n1vec[1] = ifft2(n_hat*self.w1vec_hat[1]).real
        self.n1tens[0,0] = ifft2(n_hat*self.w1tens_hat[0,0]).real
        self.n1tens[0,1] = ifft2(n_hat*self.w1tens_hat[0,1]).real
        self.n1tens[1,1] = ifft2(n_hat*self.w1tens_hat[1,1]).real
        self.n1tens[1,0] = ifft2(n_hat*self.w1tens_hat[1,0]).real

        self.n0 = self.n1/(np.pi*self.sigma)
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
    test2 = True # slit-like pore
    test3 = False
    test4 = False

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
        rhob = eta/(np.pi/6.0)

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
        mu = np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)
        
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,1.0e-12,2.0,True)

        n[:] = np.exp(nsol)
        Nint = n.sum()*delta**2
        print('rhob=',rhob,'\t F/N =',(Omegasol*L**2)/N+mu)

        np.save('fmt2d-rf-slitpore-eta'+str(eta)+'-N-'+str(N)+'.npy',[z,n[:,N//2]/rhob])

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
        delta = 0.01
        L = N*delta
        fmt = RFMT2D(N,delta)
        etaarray = np.array([0.75,0.77])
        rhobarray = etaarray/(np.pi/4)

        n0 = np.ones((N,N),dtype=np.float32)
        # for i in range(N):
        #     for j in range(N):
        #         for k in range(N):
        #             r2 = delta**2*((i-N/2)**2+(j-N/2)**2+(k-N/2)**2)
        #             if r2>=1.0: n0[i,j,k] = 1.0 + 0.01*np.random.randn()
        # rhohat = fft2(n0)
        rhohat = np.zeros((N,N),dtype=np.complex64)
        kx = np.fft.fftfreq(N, d=delta)*2*np.pi
        ky = np.fft.fftfreq(N, d=delta)*2*np.pi
        kcut = kx.max()/40
        Kx,Ky = np.meshgrid(kx,ky)
        def Pk(kx,ky):
            k = np.sqrt(kx**2+ky**2)
            return np.where(k>kcut,0.0, np.where(k>0,N**2*0.005*np.random.randn(N,N),1.0*N**2))
        rhohat[:] = Pk(Kx,Ky)
        n0[:] = ifft2(rhohat).real
        # n0[:] = np.abs(ifft2(rhohat))
        # n0[:] = n0/n0.max()

        plt.imshow(n0.real, cmap='Greys_r')
        plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.show()

        # n0 = 1.0e-12*np.ones((N,N,N),dtype=np.float32)
        # for i in range(N):
        #     for j in range(N):
        #         for k in range(N):
        #             r2 = delta**2*((i-N/2)**2+(j-N/2)**2+(k-N/2)**2)
        #             if r2>=1.0: n0[i,j,k] = 1.0
        # n0 = np.ones((N,N,N),dtype=np.float32)
        # n0[27:38,:,:] = 1.0e-12
        lnn = np.log(n0)
        del n0

        z = np.linspace(-L/2,L/2,N)
            
        def Omega(lnn,mu):
            n = np.exp(lnn)
            n_hat = fft2(n)
            phi = fmt.Phi(n_hat)
            del n_hat
            Omegak = n*(lnn-1.0) + phi - mu*n
            return Omegak.sum()*delta**3/L**3

        def dOmegadnR(lnn,mu):
            n = np.exp(lnn)
            n_hat = fft2(n)
            dphidn = fmt.dPhidn(n_hat)
            del n_hat
            return n*(lnn + dphidn - mu)*delta**3/L**3

        for i in range(rhobarray.size):
            print("Doing the N=%d"% N) 
            # eta = etaarray[i]
            rhob = rhobarray[i]
            eta = rhob*(np.pi/4.0)
            mu = np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)
            
            x = lnn 
            
            [nsol,Omegasol,Niter] = optimize_fire2(x,Omega,dOmegadnR,mu,1.0e-20,0.1,True)

            n = np.exp(nsol)

            np.save('fmt2d-rf-densityfield-eta'+str(eta)+'-N-'+str(N)+'.npy',nsol)

            plt.imshow(n.real/rhob, cmap='viridis')
            plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.savefig('densityprofile-fmt2d-eta'+str(eta)+'-N-'+str(N)+'.pdf', bbox_inches='tight')
            plt.show()
                
        # plt.xlabel(r'$x/\sigma$')
        # plt.ylabel(r'$\rho(x)\sigma^3$')
        # # plt.xlim(0.5,3)
        # # plt.ylim(0.0,11)
        # plt.savefig('densityprofile-N%d.pdf'% N, bbox_inches='tight')
        # plt.show()
        # plt.close()

    if test4:
        L = 6.4
        eta = 0.4783
        rhob = eta/(np.pi/6.0)
        mu = np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)

        Narray = np.array([64,128,256])

        for N in Narray:
            print("Doing the N=%d"% N) 
            delta = L/N
            
            fmt = WBIIFFT(N,delta)

            n0 = 1.0e-12*np.ones((N,N,N),dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        r2 = delta**2*((i-N/2)**2+(j-N/2)**2+(k-N/2)**2)
                        if r2>=1.0: n0[i,j,k] = rhob

            # n0 = rhob*np.ones((N,N,N),dtype=np.float32)
            # n0[27:37,:,:] = 1.0e-12
            # n0 = rhob + 0.2*np.random.randn(N,N,N)

            # n_hat = fft2(n0)
            # plt.imshow(n0[:,:,N//2], cmap='Greys_r')
            # plt.colorbar(label='$\\rho(x,y,0)/\\rho_b$')
            # # plt.xlabel('$x$')
            # # plt.ylabel('$y$')
            # plt.show()

            z = np.linspace(-L/2,L/2,N)
            
            def Omega(lnn,mu):
                n = np.exp(lnn)
                n_hat = fft2(n)
                phi = fmt.Phi(n_hat)
                del n_hat
                Omegak = n*(lnn-1.0) + phi - mu*n
                return Omegak.sum()*delta**3/L**3

            def dOmegadnR(lnn,mu):
                n = np.exp(lnn)
                n_hat = fft2(n)
                dphidn = fmt.dPhidn(n_hat)
                del n_hat
                return n*(lnn + dphidn - mu)*delta**3/L**3
            
            lnn = np.log(n0)
            
            [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,1.0e-12,1.0,True)

            n = np.exp(nsol)

            # plt.plot(np.arange(0,(N//2+1)*0.5/L,0.5/L),Sk, 'k')
            # plt.xlabel('$k$')
            # plt.ylabel('$S(k)$')
            # plt.show()
            # plt.close()

            rhobcalc = n.mean()

            np.save('fmt-wbii-densityfield-eta'+str(eta)+'-N-'+str(N)+'.npy',n)

            plt.imshow(n[:,:,N//2]/rhobcalc, cmap='Greys_r')
            plt.colorbar(label=r'$\rho(x,y,0)/\rho_b$')
            plt.xlabel('$x/\\sigma$')
            plt.ylabel('$y/\\sigma$')
            plt.savefig('densitymap-N%d.pdf'% N, bbox_inches='tight')
            # plt.show()
            plt.close()

            plt.plot(z,n[:,N//2,N//2]/rhobcalc)
            plt.xlabel(r'$x/\sigma$')
            plt.ylabel(r'$\rho(x,0,0)/\rho_b$')
            # plt.xlim(0.5,3)
            # plt.ylim(0.0,5)
            plt.savefig('densityprofile-N%d.pdf'% N, bbox_inches='tight')
            # plt.show()
            plt.close()