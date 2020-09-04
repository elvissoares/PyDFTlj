import numpy as np
from scipy.special import spherical_jn
# from scipy.fft import fftn, ifftn
import pyfftw
import multiprocessing
from pyfftw.interfaces.scipy_fftpack import fftn, ifftn
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-16
# Updated: 2020-07-25
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

twopi = 2*np.pi

def sigmaLancsozFT(kx,ky,kz,kcut):
    return np.sinc(kx/kcut)*np.sinc(ky/kcut)*np.sinc(kz/kcut)

def translationFT(kx,ky,kz,a):
    return np.exp(1.0j*(kx+ky+kz)*a)

def w3FT(kx,ky,kz,sigma=1.0):
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    return np.piecewise(k,[k<=1e-6,k>1e-6],[np.pi*sigma**3/6,lambda k: (np.pi*sigma**2/k)*spherical_jn(1,0.5*sigma*k)])

def w2FT(kx,ky,kz,sigma=1.0):
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    return np.pi*sigma**2*spherical_jn(0,0.5*sigma*k)

# The Rosenfeld Functional
class RFFFT():
    def __init__(self,N,delta,sigma=1.0):
        self.dim = 3
        self.N = N
        self.delta = delta
        self.L = N*delta
        self.sigma = sigma

        self.w3_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)
        self.w2_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)
        self.w2vec_hat = np.zeros((3,self.N,self.N,self.N),dtype=np.complex64)

        self.n3 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2vec = np.zeros((3,self.N,self.N,self.N),dtype=np.float32)

        kx = ky = kz = np.fft.fftfreq(self.N, d=self.delta)*twopi
        kcut = np.pi/self.delta
        Kx,Ky,Kz = np.array(np.meshgrid(kx,ky,kz),dtype=np.complex64)

        self.w3_hat = w3FT(Kx,Ky,Kz,sigma)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)

        self.w2_hat = w2FT(Kx,Ky,Kz,sigma)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)

        self.w2vec_hat[0] = -1.0j*Kx*self.w3_hat
        self.w2vec_hat[1] = -1.0j*Ky*self.w3_hat
        self.w2vec_hat[2] = -1.0j*Kz*self.w3_hat

    def weighted_densities(self,n_hat):
        self.n3 = ifftn(n_hat*self.w3_hat).real
        self.n2 = ifftn(n_hat*self.w2_hat).real
        self.n2vec[0] = ifftn(n_hat*self.w2vec_hat[0]).real
        self.n2vec[1] = ifftn(n_hat*self.w2vec_hat[1]).real
        self.n2vec[2] = ifftn(n_hat*self.w2vec_hat[2]).real
        self.n0 = self.n2/(np.pi*self.sigma**2)
        self.n1 = self.n2/(twopi*self.sigma)
        self.n1vec = self.n2vec/(twopi*self.sigma)

    def Phi(self,n_hat):
        self.weighted_densities(n_hat)
        aux = 1-self.n3
        return (-self.n0*np.log(aux)+(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2]))/aux+(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))/(24*np.pi*aux**2)).real

    def dPhidn(self,n_hat):
        self.weighted_densities(n_hat)
        aux = 1-self.n3
        self.dPhidn0 = fftn(-np.log(aux))
        self.dPhidn1 = fftn(self.n2/aux)
        self.dPhidn2 = fftn(self.n1/aux + (3*self.n2*self.n2-3*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))/(24*np.pi*aux**2))

        self.dPhidn3 = fftn(self.n0/(aux) +(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2]))/((aux)*(aux)) + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))/(12*np.pi*(aux)*(aux)*(aux)))

        self.dPhidn1vec0 = (-1)*fftn(self.n2vec[0]/aux)
        self.dPhidn1vec1 = (-1)*fftn(self.n2vec[1]/aux)
        self.dPhidn1vec2 = (-1)*fftn(self.n2vec[2]/aux)
        self.dPhidn2vec0 = fftn( -self.n1vec[0]/aux - self.n2*self.n2vec[0]/(4*np.pi*aux**2))
        self.dPhidn2vec1 = fftn(-self.n1vec[1]/aux- self.n2*self.n2vec[1]/(4*np.pi*aux**2))
        self.dPhidn2vec2 = fftn(-self.n1vec[2]/aux- self.n2*self.n2vec[2]/(4*np.pi*aux**2))

        dPhidn = ifftn((self.dPhidn2 + self.dPhidn1/(twopi*self.sigma) + self.dPhidn0/(np.pi*self.sigma**2))*self.w2_hat)

        dPhidn += ifftn(self.dPhidn3*self.w3_hat)

        dPhidn -= ifftn((self.dPhidn2vec0+self.dPhidn1vec0/(twopi*self.sigma))*self.w2vec_hat[0] +(self.dPhidn2vec1+self.dPhidn1vec1/(twopi*self.sigma))*self.w2vec_hat[1] + (self.dPhidn2vec2+self.dPhidn1vec2/(twopi*self.sigma))*self.w2vec_hat[2])

        del self.dPhidn0,self.dPhidn1,self.dPhidn2,self.dPhidn3,self.dPhidn1vec0,self.dPhidn1vec1,self.dPhidn1vec2,self.dPhidn2vec0,self.dPhidn2vec1,self.dPhidn2vec2
        
        return dPhidn.real


class WBIIFFT():
    def __init__(self,N,delta,sigma=1.0):
        self.dim = 3
        self.N = N
        self.delta = delta
        self.L = N*delta
        self.sigma = sigma

        self.w3_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)
        self.w2_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)
        self.w2vec_hat = np.empty((3,self.N,self.N,self.N),dtype=np.complex64)

        self.n3 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2vec = np.empty((3,self.N,self.N,self.N),dtype=np.float32)
        self.n1vec = np.empty((3,self.N,self.N,self.N),dtype=np.float32)
        
        kx = ky = kz = np.fft.fftfreq(self.N, d=self.delta)*twopi
        kcut = np.pi/self.delta
        Kx,Ky,Kz = np.meshgrid(kx,ky,kz)
        del kx,ky,kz

        self.w3_hat[:] = self.sigma**2*w3FT(Kx,Ky,Kz)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)
        self.w3_hat[0,0,0] = np.pi*self.sigma**3/6

        self.w2_hat[:] = self.sigma**2*w2FT(Kx,Ky,Kz)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)

        self.w2vec_hat[0] = -1.0j*Kx*self.w3_hat
        self.w2vec_hat[1] = -1.0j*Ky*self.w3_hat
        self.w2vec_hat[2] = -1.0j*Kz*self.w3_hat

        del Kx,Ky,Kz

    def weighted_densities(self,n_hat):
        self.n3[:] = ifftn(n_hat*self.w3_hat).real
        self.n2[:] = ifftn(n_hat*self.w2_hat).real
        self.n2vec[0] = ifftn(n_hat*self.w2vec_hat[0]).real
        self.n2vec[1] = ifftn(n_hat*self.w2vec_hat[1]).real
        self.n2vec[2] = ifftn(n_hat*self.w2vec_hat[2]).real
        self.n0 = self.n2/(np.pi*self.sigma**2)
        self.n1 = self.n2/(twopi*self.sigma)
        self.n1vec[0] = self.n2vec[0]/(twopi*self.sigma)
        self.n1vec[1] = self.n2vec[1]/(twopi*self.sigma)
        self.n1vec[2] = self.n2vec[2]/(twopi*self.sigma)
        self.oneminusn3 = 1-self.n3

        self.phi2 = 1+(2*self.n3-self.n3*self.n3+2*self.oneminusn3*np.log(self.oneminusn3))/(3*self.n3)

        self.phi3 = 1-(2*self.n3-3*self.n3*self.n3+2*self.n3*self.n3*self.n3+2*np.log(self.oneminusn3)*self.oneminusn3**2)/(3*self.n3*self.n3)

        self.dphi2dn3 = -(self.n3*(2+self.n3)+2*np.log(self.oneminusn3))/(3*self.n3*self.n3)
        self.dphi3dn3 = 2*self.oneminusn3*self.dphi2dn3/self.n3

    def Phi(self,n_hat):
        self.weighted_densities(n_hat)
        return (-self.n0*np.log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2])) + (self.phi3/(24*np.pi*self.oneminusn3**2))*(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2])) ).real

    def dPhidn(self,n_hat):
        self.weighted_densities(n_hat)

        self.dPhidn0 = fftn(-np.log(self.oneminusn3 ))
        self.dPhidn1 = fftn(self.n2*self.phi2/self.oneminusn3 )
        self.dPhidn2 = fftn(self.n1*self.phi2/self.oneminusn3  + (3*self.n2*self.n2-3*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))*self.phi3/(24*np.pi*self.oneminusn3**2) )

        self.dPhidn3 = fftn(self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec[0]*self.n2vec[0]+self.n1vec[1]*self.n2vec[1]+self.n1vec[2]*self.n2vec[2]))*(self.dphi2dn3 + self.phi2/self.oneminusn3)/self.oneminusn3 + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec[0]*self.n2vec[0]+self.n2vec[1]*self.n2vec[1]+self.n2vec[2]*self.n2vec[2]))*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/(24*np.pi*self.oneminusn3**2) ) 

        self.dPhidn1vec0 = fftn( -self.n2vec[0]*self.phi2/self.oneminusn3 )
        self.dPhidn1vec1 = fftn( -self.n2vec[1]*self.phi2/self.oneminusn3 )
        self.dPhidn1vec2 = fftn( -self.n2vec[2]*self.phi2/self.oneminusn3 )
        self.dPhidn2vec0 = fftn( -self.n1vec[0]*self.phi2/self.oneminusn3  - self.n2*self.n2vec[0]*self.phi3/(4*np.pi*self.oneminusn3**2))
        self.dPhidn2vec1 = fftn(-self.n1vec[1]*self.phi2/self.oneminusn3 - self.n2*self.n2vec[1]*self.phi3/(4*np.pi*self.oneminusn3**2))
        self.dPhidn2vec2 = fftn(-self.n1vec[2]*self.phi2/self.oneminusn3 - self.n2*self.n2vec[2]*self.phi3/(4*np.pi*self.oneminusn3**2))

        dPhidn = ifftn((self.dPhidn2 + self.dPhidn1/(twopi*self.sigma) + self.dPhidn0/(np.pi*self.sigma**2))*self.w2_hat)
        dPhidn += ifftn(self.dPhidn3*self.w3_hat)
        dPhidn += ifftn((self.dPhidn2vec0+self.dPhidn1vec0/(twopi*self.sigma))*self.w2vec_hat[0] +(self.dPhidn2vec1+self.dPhidn1vec1/(twopi*self.sigma))*self.w2vec_hat[1] + (self.dPhidn2vec2+self.dPhidn1vec2/(twopi*self.sigma))*self.w2vec_hat[2])

        del self.dPhidn0,self.dPhidn1,self.dPhidn2,self.dPhidn3,self.dPhidn1vec0,self.dPhidn1vec1,self.dPhidn1vec2,self.dPhidn2vec0,self.dPhidn2vec1,self.dPhidn2vec2
        
        return dPhidn.real



##### Take a example using FMT ######
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    
    #############################
    test1 = False
    test2 = False # slit-like pore
    test3 = False
    test4 = False
    test5 = True # solid-fluid phase diagram gaussian parametrization

    if test1:
        delta = 0.05
        N = 128
        L = N*delta
        fmt = WBIIFFT(N,delta)

        w = ifftn(fmt.w3_hat)
        x = np.linspace(-L/2,L/2,N)
        y = np.linspace(-L/2,L/2,N)
        X, Y = np.meshgrid(x,y)

        print(w.real.sum(),np.pi/6)

        wmax = np.max(w[:,:,N//2].real)

        cmap = plt.get_cmap('jet')
        # cp = ax.contourf(X, Y, w[:,:,N//2].real/wmax,20, cmap=cmap)
        # # ax.set_title(r'$\omega_3(r)=\Theta(\sigma/2-r)$')
        # fig.colorbar(cp,ticks=[0,0.2,0.4,0.6,0.8,1.0]) 
        # ax.set_xlabel(r'$x/\sigma$')
        # ax.set_ylabel(r'$y/\sigma$')
        # fig.savefig('omega3-N%d.pdf'% N, bbox_inches='tight')
        # plt.show()
        # plt.close()

        w = ifftn(fmt.w2_hat)

        wmax = np.max(w[:,:,N//2].real)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        print(w.real.sum(),np.pi)
        cp2 = ax.contourf(X, Y, w[:,:,N//2].real/wmax,20, cmap=cmap)
        # ax.set_title(r'$\omega_2(r)=\delta(\sigma/2-r)$')
        fig.colorbar(cp2,ticks=[0,0.2,0.4,0.6,0.8,1.0]) 
        ax.set_xlabel(r'$x/\sigma$')
        ax.set_ylabel(r'$y/\sigma$')
        fig.savefig('omega2-N%d.pdf'% N, bbox_inches='tight')
        plt.show()
        plt.close()

    ######################################################
    if test2:
        L = 8.0
        N = 256
        delta = L/N
        fmt = RFFFT(N,delta)
        eta = 0.42
        rhob = eta/(np.pi/6.0)

        nsig = int(0.5/delta)

        n0 = rhob*np.ones((N,N,N),dtype=np.float32)
        n0[:nsig,:,:] = 1.0e-12
        n0[N-nsig:,:,:] = 1.0e-12
        n = np.empty((N,N,N),dtype=np.float32)
        n_hat = np.empty((N,N,N),dtype=np.complex64)

        # plt.imshow(n0[:,:,N//2].real, cmap='Greys_r')
        # plt.colorbar(label='$\\rho(x,y,0)/\\rho_b$')
        # # plt.xlabel('$x$')
        # # plt.ylabel('$y$')
        # plt.show()

        lnn = np.log(n0)
        del n0

        z = np.linspace(-L/2,L/2,N)
            
        def Omega(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fftn(n)
            phi = fmt.Phi(n_hat)
            Omegak = n*(lnn-1.0) + phi - mu*n
            return Omegak.sum()*delta**3/L**3

        def dOmegadnR(lnn,mu):
            n[:] = np.exp(lnn)
            n_hat[:] = fftn(n)
            dphidn = fmt.dPhidn(n_hat)
            return n*(lnn + dphidn - mu)*delta**3/L**3

        print("Doing the N=%d"% N) 
        mu = np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)
        
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,1.0e-16,100.0,True)

        n[:] = np.exp(nsol)
        Nint = n.sum()*delta**3
        print('rhob=',rhob,'\t F/N =',(Omegasol*L**3)/N+mu)

        np.save('fmt-rf-slitpore-eta'+str(eta)+'-N-'+str(N)+'.npy',[z,n[:,N//2,N//2]/rhob])

        # [zRF,rhoRF] = np.load('fmt-wbii-slitpore-eta'+str(eta)+'-N-'+str(N)+'.npy') 

        plt.plot(z,n[:,N//2,N//2]/rhob,label='RF')
        # plt.plot(zRF,rhoRF/rhob,label='WBII')
        plt.plot(z,((1+eta+eta**2)/(1-eta)**3)*np.ones(z.size),'--',color='grey')
        plt.xlabel(r'$x/\sigma$')
        plt.ylabel(r'$\rho(x)/\rho_b$')
        plt.xlim(-L/2,L/2)
        plt.ylim(0.0,10)
        plt.legend()
        plt.savefig('slitpore-eta%.2f-N%d.pdf'% (eta,N), bbox_inches='tight')
        plt.show()
        plt.close()

    if test3:
        N = 128
        delta = 0.05
        L = N*delta
        fmt = RFFFT(N,delta)
        # etaarray = np.array([0.2,0.3,0.4257,0.4783])
        rhobarray = np.array([0.05,0.8])

        n0 = 1e-12*np.ones((N,N,N),dtype=np.float32)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    r2 = delta**2*((i-N/2)**2+(j-N/2)**2+(k-N/2)**2)
                    if r2>=0.25: n0[i,j,k] = 1.0 + 0.1*np.random.randn()
        lnn = np.log(n0)
        del n0

        z = np.linspace(0,L/2,N//2)
            
        def Omega(lnn,mu):
            n = np.exp(lnn)
            n_hat = fftn(n)
            phi = fmt.Phi(n_hat)
            del n_hat
            Omegak = n*(lnn-1.0) + phi - mu*n
            return Omegak.sum()*delta**3/L**3

        def dOmegadnR(lnn,mu):
            n = np.exp(lnn)
            n_hat = fftn(n)
            dphidn = fmt.dPhidn(n_hat)
            del n_hat
            return n*(lnn + dphidn - mu)*delta**3/L**3

        for i in range(rhobarray.size):
            print("Doing the N=%d"% N) 
            # eta = etaarray[i]
            rhob = rhobarray[i]
            eta = rhob*(np.pi/6.0)
            mu = np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)
            
            x = lnn + np.log(rhob)
            
            [nsol,Omegasol,Niter] = optimize_fire2(x,Omega,dOmegadnR,mu,1.0e-20,100.0,True)

            n = np.exp(nsol)

            np.save('fmt-rf-densityfield-rho'+str(rhob)+'-N-'+str(N)+'.npy',nsol)

            eta = (np.pi/6)*rhob
            
        #     Nint = n.sum()*delta**3
        #     print('rhob=',rhob,'\t F/N =',(Omegasol*L**3)/N+mu)
            plt.plot(z,n[N//2:,N//2,N//2]/rhob)
            plt.plot(z,((1+eta+eta**2)/(1-eta)**3)*np.ones(z.size),'--',color='grey')
            plt.savefig('densityprofile-rf-rho%.2f-N%d.pdf'% (rhob,N), bbox_inches='tight')
            plt.close()
            

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

            # n_hat = fftn(n0)
            # plt.imshow(n0[:,:,N//2], cmap='Greys_r')
            # plt.colorbar(label='$\\rho(x,y,0)/\\rho_b$')
            # # plt.xlabel('$x$')
            # # plt.ylabel('$y$')
            # plt.show()

            z = np.linspace(-L/2,L/2,N)
            
            def Omega(lnn,mu):
                n = np.exp(lnn)
                n_hat = fftn(n)
                phi = fmt.Phi(n_hat)
                del n_hat
                Omegak = n*(lnn-1.0) + phi - mu*n
                return Omegak.sum()*delta**3/L**3

            def dOmegadnR(lnn,mu):
                n = np.exp(lnn)
                n_hat = fftn(n)
                dphidn = fmt.dPhidn(n_hat)
                del n_hat
                return n*(lnn + dphidn - mu)*delta**3/L**3
            
            lnn = np.log(n0)
            
            [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,1.0e-12,1.0,True)

            n = np.exp(nsol)

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
    
    if test5: 
        a = np.sqrt(2) #fcc unit cell
        N = 64

        print('The fluid-solid phase diagram for Hard-Sphere using a gaussian parametrization')
        print('N=',N)

        # define the variables to the gaussian parametrization
        # the fcc lattice
        def gaussian(alpha,L):
            x = np.linspace(-L/2,L/2,N)
            X,Y,Z = np.meshgrid(x,x,x)
            lattice = 0.5*L*np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,1],[0,0,-1],[0,0,1],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]])
            rho = np.zeros((N,N,N),dtype=np.float32)
            for R in lattice:
                rho += np.power(alpha/np.pi,1.5)*np.exp(-alpha*((X-R[0])**2+(Y-R[1])**2+(Z-R[2])**2))
            return rho

        def dgaussiandalpha(alpha,L):
            x = np.linspace(-L/2,L/2,N)
            X,Y,Z = np.meshgrid(x,x,x)
            lattice = 0.5*L*np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,1],[0,0,-1],[0,0,1],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]])
            drhodalpha = np.zeros((N,N,N),dtype=np.float32)
            for R in lattice:
                drhodalpha += np.power(alpha/np.pi,1.5)*(1.5/alpha-((X-R[0])**2+(Y-R[1])**2+(Z-R[2])**2))*np.exp(-alpha*((X-R[0])**2+(Y-R[1])**2+(Z-R[2])**2))
            return drhodalpha

        alphaliq = np.array([0.01])
        alphacrystal = 6.0
        n = np.empty((N,N,N),dtype=np.float32)
        n_hat = np.empty((N,N,N),dtype=np.complex64)
        dndalpha = np.empty((N,N,N),dtype=np.float32)
        lnnf = np.array(np.log(0.7+0.1*np.random.randn(N,N,N)),dtype=np.float32)
        n[:] = gaussian(alphacrystal,a)
        print(n.sum()/N**3)
        plt.imshow(n[0].real, cmap='viridis')
        plt.colorbar(label='$\\rho(x,y,-L/2)$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.show()

        rhob = 0.8
        eta = rhob*np.pi/6
        mul = np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)

        rhob = 1.4
        eta = rhob*np.pi/6
        mus = np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)

        muarray = np.linspace(15.87,25.77,5,endpoint=True)

        output = True

        print('#######################################')
        print("mu\trho\trho2\tOmega1\tOmega2")

        p0 = np.array([alphacrystal,2*a])

        Ll = a
        deltal = Ll/N
        FMTl = WBIIFFT(N,deltal)

        for i in range(muarray.size):

            mu = muarray[i]
            
            ## The Grand Canonical Potential
            def Omegaliq(lnn,mu):
                n[:] = np.exp(lnn)
                n_hat[:] = fftn(n)
                phi = FMTl.Phi(n_hat)
                Omegak = n*(lnn-1.0) + phi - mu*n
                return Omegak.sum()*deltal**3/Ll**3

            def dOmegadnRliq(lnn,mu):
                n[:] = np.exp(lnn)
                n_hat[:] = fftn(n)
                dphidn = FMTl.dPhidn(n_hat)
                return n*(lnn + dphidn - mu)*deltal**3/Ll**3

            def Omega(p,mu):
                [alpha,L] = [np.abs(p[0]),np.abs(p[1])]
                delta = L/N
                FMT = WBIIFFT(N,delta)
                n[:] = gaussian(alpha,L)
                n_hat[:] = fftn(n)
                phi = FMT.Phi(n_hat)
                FHS = np.sum(phi)*delta**3
                Fid = np.sum(n*(np.log(n)-1.0))*delta**3
                Nn = n.sum()*delta**3
                return (Fid+FHS-mu*Nn)

            def dOmegadnR(p,mu):
                [alpha,L] = [np.abs(p[0]),np.abs(p[1])]
                delta = L/N
                FMT = WBIIFFT(N,delta)
                n[:] = gaussian(alpha,L)
                n_hat[:] = fftn(n)
                dphidn = FMT.dPhidn(n_hat)
                dndalpha[:] = dgaussiandalpha(alpha,L)
                phi = FMT.Phi(n_hat)
                FHS = np.sum(phi)*delta**3
                Fid = np.sum(n*(np.log(n)-1.0))*delta**3
                Nn = n.sum()*delta**3
                return np.array([np.sum(dndalpha*(np.log(n) + dphidn - mu)*delta**3),(Fid+FHS-mu*Nn)*3/L])

            # [lnnsol1,Omegasol,Niter] = optimize_fire2(lnnf,Omegaliq,dOmegadnRliq,mu,1.0e-12,1.0,output)
            [p,Omegasol2,Niter] = optimize_fire2(p0,Omega,dOmegadnR,mu,1.0e-12,0.00001,output)

            [alphasol,Lsol] = [np.abs(p[0]),np.abs(p[1])]
            rhomean = np.exp(lnnsol1).sum()/N**3
            rhomean2 = gaussian(alphasol,Lsol).sum()/N**3

            # n[:] = np.exp(lnnsol2)
            # plt.imshow(n[0].real, cmap='Greys_r')
            # plt.colorbar(label='$\\rho(x,y)/\\rho_b$')
            # plt.xlabel('$x$')
            # plt.ylabel('$y$')
            # plt.show()

            print(mu,rhomean,rhomean2,Omegasol,Omegasol2/Lsol**3)