import numpy as np
from scipy.special import jv
from scipy.fft import fftn, ifftn
# import pycuda.autoinit
# from pycuda import gpuarray
# from skcuda import fft
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-16
# Updated: 2020-07-13

def sigmaLancsozFT(kx,ky,kcut):
    return np.sinc(kx/kcut)*np.sinc(ky/kcut)

def translationFT(kx,ky,a):
    return np.exp(1.0j*(kx+ky)*a)

def w2FT(kx,ky):
    k = np.sqrt(kx**2 + ky**2)
    return (np.pi/k)*jv(1,0.5*k)

def w1FT(kx,ky):
    k = np.sqrt(kx**2 + ky**2)
    return np.pi*jv(0,0.5*k)

def w1tensFT(kx,ky):
    k = np.sqrt(kx**2 + ky**2)
    return (np.pi/k**2)*jv(2,0.5*k)


# The Rosenfeld Functional
# The spatial generation of the wweiths is ok!
class RFFFT2D():
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

        self.w2_hat = self.sigma*w2FT(Kx,Ky)*sigmaLancsozFT(Kx,Ky,kcut)*translationFT(Kx,Ky,0.5*self.L)
        self.w2_hat[0,0] = self.sigma*np.pi/4

        self.w1_hat = self.sigma*w1FT(Kx,Ky)*sigmaLancsozFT(Kx,Ky,kcut)*translationFT(Kx,Ky,0.5*self.L)

        w1tens_aux = self.sigma*w1tensFT(Kx,Ky)*sigmaLancsozFT(Kx,Ky,kcut)*translationFT(Kx,Ky,0.5*self.L)
        w1tens_aux[0,0] = self.sigma*np.pi/128

        self.w1vec_hat[0] = -1.0j*Kx*self.w2_hat
        self.w1vec_hat[1] = -1.0j*Ky*self.w2_hat
        self.w1tens_hat[0,0] = -Kx*Kx*w1tens_aux + 2*self.w2_hat
        self.w1tens_hat[0,1] = -Kx*Ky*w1tens_aux
        self.w1tens_hat[1,1] = -Ky*Ky*w1tens_aux + 2*self.w2_hat
        self.w1tens_hat[1,0] = -Ky*Kx*w1tens_aux

        # # # the GPU Fourier Transform
        # # x_gpu = gpuarray.to_gpu(np.zeros((self.N,self.N)))
        # # self.plan = Plan(x_gpu.shape,np.float32,np.complex64)
        # # self.inverse_plan = Plan(x.shape, in_dtype=np.complex64, out_dtype=np.float32)

        # w2 = ifftn(self.w2_hat).real
        # print(w2.sum()/(np.pi/4))
        # # w2 = ifftn(self.w2_hat).real
        # # print(w2.sum()/(np.pi))
        # # plt.imshow(w2[N//2])
        # plt.imshow(w2)
        # plt.colorbar(cmap='viridis')
        # plt.show()

    def weighted_densities(self,n_hat):
        self.n2 = ifftn(n_hat*self.w2_hat).real
        self.n1 = ifftn(n_hat*self.w1_hat).real
        self.n1vec[0] = ifftn(n_hat*self.w1vec_hat[0]).real
        self.n1vec[1] = ifftn(n_hat*self.w1vec_hat[1]).real
        self.n1tens[0,0] = ifftn(n_hat*self.w1tens_hat[0,0]).real
        self.n1tens[0,1] = ifftn(n_hat*self.w1tens_hat[0,1]).real
        self.n1tens[1,1] = ifftn(n_hat*self.w1tens_hat[1,1]).real
        self.n1tens[1,0] = ifftn(n_hat*self.w1tens_hat[1,0]).real

        self.n0 = self.n1/(np.pi*self.sigma)
        self.oneminusn2 = 1-self.n2 

    def Phi(self,n_hat):
        self.weighted_densities(n_hat)
        return (-self.n0*np.log(self.oneminusn2)+((19/12.)*self.n1**2-(5.0/12.0)*(self.n1vec[0]*self.n1vec[0]+self.n1vec[1]*self.n1vec[1])-(7/6.)*(self.n1tens[0,0]*self.n1tens[0,0]+self.n1tens[0,1]*self.n1tens[0,1]+self.n1tens[1,1]*self.n1tens[1,1]+self.n1tens[1,0]*self.n1tens[1,0]))/(4*np.pi*self.oneminusn2)).real

    def dPhidn(self,n_hat):
        self.weighted_densities(n_hat)

        denom = 1.0/(24*np.pi*self.oneminusn2)
        self.dPhidn0 = fftn(-np.log(self.oneminusn2))
        self.dPhidn1 = fftn(19*self.n1*denom)
        self.dPhidn2 = fftn( self.n0/self.oneminusn2 + ((19/12.)*self.n1**2-(5.0/12.0)*(self.n1vec[0]*self.n1vec[0]+self.n1vec[1]*self.n1vec[1])-(7/6.)*(self.n1tens[0,0]*self.n1tens[0,0]+self.n1tens[0,1]*self.n1tens[0,1]+self.n1tens[1,1]*self.n1tens[1,1]+self.n1tens[1,0]*self.n1tens[1,0]))/(4*np.pi*self.oneminusn2**2))

        self.dPhidn1vec0 = fftn( -5*self.n1vec[0]*denom)
        self.dPhidn1vec1 = fftn( -5*self.n1vec[1]*denom)
        self.dPhidn1tens00 = fftn( -14*self.n1tens[0,0]*denom)
        self.dPhidn1tens01 = fftn( -28*self.n1tens[0,1]*denom)
        self.dPhidn1tens11 = fftn( -14*self.n1tens[1,1]*denom)

        self.dPhidn2conv = ifftn(self.dPhidn2*self.w2_hat)
        self.dPhidn1conv = ifftn(self.dPhidn1*self.w1_hat + self.dPhidn0*self.w1_hat/np.pi)
        self.dPhidn1vecconv = (-1.0)*ifftn(self.dPhidn1vec0*self.w1vec_hat[0]+self.dPhidn1vec1*self.w1vec_hat[1])

        self.dPhidn1tensconv = ifftn(self.dPhidn1tens00*self.w1tens_hat[0,0]+2*self.dPhidn1tens01*self.w1tens_hat[0,1]+self.dPhidn1tens11*self.w1tens_hat[1,1])

        del self.dPhidn0,self.dPhidn1,self.dPhidn2,self.dPhidn1vec0,self.dPhidn1vec1,self.dPhidn1tens00 ,self.dPhidn1tens01,self.dPhidn1tens11
        
        return (self.dPhidn2conv+self.dPhidn1conv+self.dPhidn1vecconv+self.dPhidn1tensconv).real


##### Take a example using FMT ######
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from fire import optimize_fire2

        # http://wiki.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
    # this is a latex constant, don't change it.
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 246.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 9/12.0 #0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize,golden_ratio * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 9
    legend_size = inverse_latex_scale * 8
    # learn how to configure:
    # http://matplotlib.sourceforge.net/users/customizing.html
    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': legend_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            # 'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            'font.serif': ['Computer Modern Roman'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                                    ],
            }
    plt.rcParams.update(params)
    plt.ioff()
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.96, bottom=0.18,
                        hspace=0.02, wspace=0.02)
    ax = fig.add_subplot(111)

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
        fmt = RFFFT2D(N,delta)

        w = ifftn(fmt.w2_hat)
        x = np.linspace(-L/2,L/2,N)
        y = np.linspace(-L/2,L/2,N)
        X, Y = np.meshgrid(x,y)

        print(w.real.sum(),np.pi/4)

        cmap = plt.get_cmap('hot')
        cp = ax.contourf(X, Y, w.real/np.max(w.real),100, cmap=cmap)
        ax.set_title(r'$\omega_2(r)=\Theta(\sigma/2-r)$')
        fig.colorbar(cp, ticks=[0, 0.2,0.4,0.6,0.8,1.0]) 
        ax.set_xlabel(r'$x/\sigma$')
        ax.set_ylabel(r'$y/\sigma$')
        fig.savefig('omega2-2d-N%d.png'% N, bbox_inches='tight')
        plt.show()

        w = ifftn(fmt.w1_hat)

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
        fmt = RFFFT2D(N,delta)
        eta = 0.6
        rhob = eta/(np.pi/4.0)

        nsig = int(0.5/delta)

        n0 = np.asarray(rhob*(1+0.1*np.random.randn(N,N)),dtype=np.float32)
        n0[:nsig,:] = 1.0e-12
        n0[N-nsig:,:] = 1.0e-12

        # plt.imshow(n0[:,:,N//2].real, cmap='Greys_r')
        # plt.colorbar(label='$\\rho(x,y,0)/\\rho_b$')
        # # plt.xlabel('$x$')
        # # plt.ylabel('$y$')
        # plt.show()

        lnn = np.log(n0)
        del n0

        z = np.linspace(-L/2,L/2,N)
            
        def Omega(lnn,mu):
            n = np.exp(lnn)
            n_hat = fftn(n)
            phi = fmt.Phi(n_hat)
            del n_hat
            Omegak = n*(lnn-1.0) + phi - mu*n
            return Omegak.sum()*delta**2/L**2

        def dOmegadnR(lnn,mu):
            n = np.exp(lnn)
            n_hat = fftn(n)
            dphidn = fmt.dPhidn(n_hat)
            del n_hat
            return n*(lnn + dphidn - mu)*delta**2/L**2

        print("Doing the N=%d"% N) 
        mu = np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)
        
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,1.0e-12,1.0,True)

        n = np.exp(nsol)
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
        plt.ylim(0.0,10)
        plt.legend()
        plt.savefig('slitpore2d-eta%.2f-N%d.pdf'% (eta,N), bbox_inches='tight')
        plt.show()
        plt.close()

    if test3:
        N = 512
        delta = 0.01
        L = N*delta
        fmt = RFFFT2D(N,delta)
        etaarray = np.array([0.75,0.77])
        rhobarray = etaarray/(np.pi/4)

        n0 = np.ones((N,N),dtype=np.float32)
        # for i in range(N):
        #     for j in range(N):
        #         for k in range(N):
        #             r2 = delta**2*((i-N/2)**2+(j-N/2)**2+(k-N/2)**2)
        #             if r2>=1.0: n0[i,j,k] = 1.0 + 0.01*np.random.randn()
        # rhohat = fftn(n0)
        rhohat = np.zeros((N,N),dtype=np.complex64)
        kx = np.fft.fftfreq(N, d=delta)*2*np.pi
        ky = np.fft.fftfreq(N, d=delta)*2*np.pi
        kcut = kx.max()/40
        Kx,Ky = np.meshgrid(kx,ky)
        def Pk(kx,ky):
            k = np.sqrt(kx**2+ky**2)
            return np.where(k>kcut,0.0, np.where(k>0,N**2*0.005*np.random.randn(N,N),1.0*N**2))
        rhohat[:] = Pk(Kx,Ky)
        n0[:] = ifftn(rhohat).real
        # n0[:] = np.abs(ifftn(rhohat))
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