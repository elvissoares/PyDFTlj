import numpy as np
from scipy.special import spherical_jn
# from scipy.fft import fftn, ifftn
import pyfftw
import multiprocessing
from pyfftw.interfaces.scipy_fftpack import fftn, ifftn
# import pycuda.autoinit
# from pycuda import gpuarray
# from skcuda import fft
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
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

def w3FT(kx,ky,kz):
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    return (np.pi/k)*spherical_jn(1,0.5*k)

def w2FT(kx,ky,kz):
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    return np.pi*spherical_jn(0,0.5*k)

def wFT(kx,ky,kz,lambdaD,delta):
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    # return np.where(k < 1.e-3 , -4*np.pi/3-(4*np.pi/lambdaD**2)*(1+lambdaD), -(4*np.pi/k)*spherical_jn(1,k)-(4*np.pi/(lambdaD**2+k**2))*(np.cos(k)+lambdaD*spherical_jn(0,k)))
    return (-4*np.pi/(lambdaD**2+k**2))*(np.cos(k)+lambdaD*spherical_jn(0,k))


# The Rosenfeld Functional
# The spatial generation of the wweiths is ok!
class RFFFT():
    def __init__(self,N,delta):
        self.dim = 3
        self.N = N
        self.delta = delta
        self.L = N*delta

        self.w3_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)
        self.w2_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)
        self.w2vec0_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)
        self.w2vec1_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)
        self.w2vec2_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)

        self.n3 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2vec0 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2vec1 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2vec2 = np.empty((self.N,self.N,self.N),dtype=np.float32)

        kx = np.fft.fftfreq(self.N, d=self.delta)*twopi
        ky = np.fft.fftfreq(self.N, d=self.delta)*twopi
        kz = np.fft.fftfreq(self.N, d=self.delta)*twopi
        kcut = np.pi/self.delta
        Kx,Ky,Kz = np.meshgrid(kx,ky,kz)

        self.w3_hat = w3FT(Kx,Ky,Kz)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)
        self.w3_hat[0,0,0] = np.pi/6

        self.w2_hat = w2FT(Kx,Ky,Kz)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)

        self.w2vec0_hat = -1.0j*Kx*self.w3_hat
        self.w2vec1_hat = -1.0j*Ky*self.w3_hat
        self.w2vec2_hat = -1.0j*Kz*self.w3_hat

        # # # the GPU Fourier Transform
        # # x_gpu = gpuarray.to_gpu(np.zeros((self.N,self.N,self.N)))
        # # self.plan = Plan(x_gpu.shape,np.float32,np.complex64)
        # # self.inverse_plan = Plan(x.shape, in_dtype=np.complex64, out_dtype=np.float32)

    def weighted_densities(self,n_hat):
        self.n3 = ifftn(n_hat*self.w3_hat)
        self.n2 = ifftn(n_hat*self.w2_hat)
        self.n2vec0 = ifftn(n_hat*self.w2vec0_hat)
        self.n2vec1 = ifftn(n_hat*self.w2vec1_hat)
        self.n2vec2 = ifftn(n_hat*self.w2vec2_hat)
        self.n0 = self.n2/np.pi
        self.n1 = self.n2/twopi
        self.n1vec0 = self.n2vec0/twopi
        self.n1vec1 = self.n2vec1/twopi
        self.n1vec2 = self.n2vec2/twopi

    def Phi(self,n_hat):
        self.weighted_densities(n_hat)
        aux = 1-self.n3
        return (-self.n0*np.log(aux)+(self.n1*self.n2-(self.n1vec0*self.n2vec0+self.n1vec1*self.n2vec1+self.n1vec2*self.n2vec2))/aux+(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec0*self.n2vec0+self.n2vec1*self.n2vec1+self.n2vec2*self.n2vec2))/(24*np.pi*aux**2)).real

    def dPhidn(self,n_hat):
        self.weighted_densities(n_hat)
        aux = 1-self.n3
        self.dPhidn0 = fftn(-np.log(aux))
        self.dPhidn1 = fftn(self.n2/aux)
        self.dPhidn2 = fftn(self.n1/aux + (3*self.n2*self.n2-3*(self.n2vec0*self.n2vec0+self.n2vec1*self.n2vec1+self.n2vec2*self.n2vec2))/(24*np.pi*aux**2))

        self.dPhidn3 = fftn(self.n0/(aux) +(self.n1*self.n2-(self.n1vec0*self.n2vec0+self.n1vec1*self.n2vec1+self.n1vec2*self.n2vec2))/((aux)*(aux)) + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec0*self.n2vec0+self.n2vec1*self.n2vec1+self.n2vec2*self.n2vec2))/(12*np.pi*(aux)*(aux)*(aux)))

        self.dPhidn1vec0 = (-1)*fftn(self.n2vec0/aux)
        self.dPhidn1vec1 = (-1)*fftn(self.n2vec1/aux)
        self.dPhidn1vec2 = (-1)*fftn(self.n2vec2/aux)
        self.dPhidn2vec0 = fftn( -self.n1vec0/aux - self.n2*self.n2vec0/(4*np.pi*aux**2))
        self.dPhidn2vec1 = fftn(-self.n1vec1/aux- self.n2*self.n2vec1/(4*np.pi*aux**2))
        self.dPhidn2vec2 = fftn(-self.n1vec2/aux- self.n2*self.n2vec2/(4*np.pi*aux**2))

        dPhidn = ifftn((self.dPhidn2 + self.dPhidn1/twopi + self.dPhidn0/np.pi)*self.w2_hat)

        dPhidn += ifftn(self.dPhidn3*self.w3_hat)

        dPhidn -= ifftn((self.dPhidn2vec0+self.dPhidn1vec0/twopi)*self.w2vec0_hat +(self.dPhidn2vec1+self.dPhidn1vec1/twopi)*self.w2vec1_hat + (self.dPhidn2vec2+self.dPhidn1vec2/twopi)*self.w2vec2_hat)

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
        self.w2vec0_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)
        self.w2vec1_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)
        self.w2vec2_hat = np.empty((self.N,self.N,self.N),dtype=np.complex64)

        self.n3 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2vec0 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2vec1 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.n2vec2 = np.empty((self.N,self.N,self.N),dtype=np.float32)
        self.ghs =  np.empty((self.N,self.N,self.N),dtype=np.float32)
        
        norm = (2*np.pi/self.L)
        M = N//2
        kcut = M*norm
        kx = np.fft.fftfreq(self.N, d=self.delta)*twopi
        ky = np.fft.fftfreq(self.N, d=self.delta)*twopi
        kz = np.fft.fftfreq(self.N, d=self.delta)*twopi
        Kx,Ky,Kz = np.meshgrid(kx,ky,kz)

        self.w3_hat = self.sigma**2*w3FT(Kx,Ky,Kz)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)
        self.w3_hat[0,0,0] = np.pi*self.sigma**3/6

        self.w2_hat = self.sigma**2*w2FT(Kx,Ky,Kz)*sigmaLancsozFT(Kx,Ky,Kz,kcut)*translationFT(Kx,Ky,Kz,0.5*self.L)

        self.w2vec0_hat = -1.0j*Kx*self.w3_hat
        self.w2vec1_hat = -1.0j*Ky*self.w3_hat
        self.w2vec2_hat = -1.0j*Kz*self.w3_hat

        # # # the GPU Fourier Transform
        # # x_gpu = gpuarray.to_gpu(np.zeros((self.N,self.N,self.N)))
        # # self.plan = Plan(x_gpu.shape,np.float32,np.complex64)
        # # self.inverse_plan = Plan(x.shape, in_dtype=np.complex64, out_dtype=np.float32)

    def weighted_densities(self,n_hat):
        self.n3 = ifftn(n_hat*self.w3_hat)
        self.n2 = ifftn(n_hat*self.w2_hat)
        self.n2vec0 = ifftn(n_hat*self.w2vec0_hat)
        self.n2vec1 = ifftn(n_hat*self.w2vec1_hat)
        self.n2vec2 = ifftn(n_hat*self.w2vec2_hat)
        self.n0 = self.n2/(np.pi*self.sigma**2)
        self.n1 = self.n2/(twopi*self.sigma)
        self.n1vec0 = self.n2vec0/(twopi*self.sigma)
        self.n1vec1 = self.n2vec1/(twopi*self.sigma)
        self.n1vec2 = self.n2vec2/(twopi*self.sigma)
        self.oneminusn3 = 1-self.n3

        self.phi2 = (2*self.n3-self.n3*self.n3+2*self.oneminusn3*np.log(self.oneminusn3))/self.n3

        self.phi3 = (2*self.n3-3*self.n3*self.n3+2*self.n3*self.n3*self.n3+2*np.log(self.oneminusn3)*self.oneminusn3**2)/(self.n3*self.n3)

        self.dphi2dn3 = -(self.n3*(2+self.n3)+2*np.log(self.oneminusn3))/(self.n3*self.n3)
        self.dphi3dn3 = 2*self.oneminusn3*self.dphi2dn3/self.n3

        xi = 1- (self.n2vec0*self.n2vec0+self.n2vec1*self.n2vec1+self.n2vec2*self.n2vec2)/self.n2**2
        self.ghs = 1/self.oneminusn3+self.n2*xi/(4*self.oneminusn3**2)+self.n2**2*xi/(72*self.oneminusn3**3)

    def Phi(self,n_hat):
        self.weighted_densities(n_hat)
        return (-self.n0*np.log(self.oneminusn3)+(self.n1*self.n2-(self.n1vec0*self.n2vec0+self.n1vec1*self.n2vec1+self.n1vec2*self.n2vec2))*(1+self.phi2/3.0)/self.oneminusn3 +(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec0*self.n2vec0+self.n2vec1*self.n2vec1+self.n2vec2*self.n2vec2))*(1-self.phi3/3.0)/(24*np.pi*self.oneminusn3**2)).real

    def dPhidn(self,n_hat):
        self.weighted_densities(n_hat)

        self.dPhidn0 = fftn(-np.log(self.oneminusn3 ))
        self.dPhidn1 = fftn(self.n2*(1+self.phi2/3.0)/self.oneminusn3 )
        self.dPhidn2 = fftn(self.n1*(1+self.phi2/3.0)/self.oneminusn3  + (3*self.n2*self.n2-3*(self.n2vec0*self.n2vec0+self.n2vec1*self.n2vec1+self.n2vec2*self.n2vec2))*(1-self.phi3/3.0)/(24*np.pi*self.oneminusn3**2))

        self.dPhidn3 = fftn(self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec0*self.n2vec0+self.n1vec1*self.n2vec1+self.n1vec2*self.n2vec2))*(self.dphi2dn3*self.oneminusn3/3+(1+self.phi2/3.0))/(self.oneminusn3**2) + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec0*self.n2vec0+self.n2vec1*self.n2vec1+self.n2vec2*self.n2vec2))*(-self.dphi3dn3*self.oneminusn3/3.0+2*(1-self.phi3/3.0))/(24*np.pi*self.oneminusn3**3))

        self.dPhidn1vec0 = fftn( -self.n2vec0*(1+self.phi2/3.0)/self.oneminusn3 )
        self.dPhidn1vec1 = fftn( -self.n2vec1*(1+self.phi2/3.0)/self.oneminusn3 )
        self.dPhidn1vec2 = fftn( -self.n2vec2*(1+self.phi2/3.0)/self.oneminusn3 )
        self.dPhidn2vec0 = fftn( -self.n1vec0*(1+self.phi2/3.0)/self.oneminusn3  - 6*self.n2*self.n2vec0*(1-self.phi3/3.0)/(24*np.pi*self.oneminusn3**2))
        self.dPhidn2vec1 = fftn(-self.n1vec1*(1+self.phi2/3.0)/self.oneminusn3 - 6*self.n2*self.n2vec1*(1-self.phi3/3.0)/(24*np.pi*self.oneminusn3**2))
        self.dPhidn2vec2 = fftn(-self.n1vec2*(1+self.phi2/3.0)/self.oneminusn3 - 6*self.n2*self.n2vec2*(1-self.phi3/3.0)/(24*np.pi*self.oneminusn3**2))

        self.dPhidn2conv = ifftn((self.dPhidn2 + self.dPhidn1/(twopi*self.sigma) + self.dPhidn0/(np.pi*self.sigma**2))*self.w2_hat)

        self.dPhidn3conv = ifftn(self.dPhidn3*self.w3_hat)

        self.dPhidn2vecconv = ifftn((self.dPhidn2vec0+self.dPhidn1vec0/(twopi*self.sigma))*self.w2vec0_hat +(self.dPhidn2vec1+self.dPhidn1vec1/(twopi*self.sigma))*self.w2vec1_hat + (self.dPhidn2vec2+self.dPhidn1vec2/(twopi*self.sigma))*self.w2vec2_hat)

        del self.dPhidn0,self.dPhidn1,self.dPhidn2,self.dPhidn3,self.dPhidn1vec0,self.dPhidn1vec1,self.dPhidn1vec2,self.dPhidn2vec0,self.dPhidn2vec1,self.dPhidn2vec2
        
        return (self.dPhidn2conv+self.dPhidn3conv-self.dPhidn2vecconv).real



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
        N = 128
        delta = L/N
        fmt = RFFFT(N,delta)
        eta = 0.42
        rhob = eta/(np.pi/6.0)

        nsig = int(0.5/delta)

        n0 = rhob*np.ones((N,N,N),dtype=np.float32)
        n0[:nsig,:,:] = 1.0e-12
        n0[N-nsig:,:,:] = 1.0e-12

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
            return Omegak.sum()*delta**3/L**3

        def dOmegadnR(lnn,mu):
            n = np.exp(lnn)
            n_hat = fftn(n)
            dphidn = fmt.dPhidn(n_hat)
            del n_hat
            return n*(lnn + dphidn - mu)*delta**3/L**3

        print("Doing the N=%d"% N) 
        mu = np.log(rhob) + (8*eta - 9*eta*eta + 3*eta*eta*eta)/np.power(1-eta,3)
        
        [nsol,Omegasol,Niter] = optimize_fire2(lnn,Omega,dOmegadnR,mu,1.0e-20,10.0,True)

        n = np.exp(nsol)
        Nint = n.sum()*delta**3
        print('rhob=',rhob,'\t F/N =',(Omegasol*L**3)/N+mu)

        np.save('fmt-rf-slitpore-eta'+str(eta)+'-N-'+str(N)+'.npy',[z,n[:,N//2,N//2]/rhob])

        [zRF,rhoRF] = np.load('fmt-wbii-slitpore-eta'+str(eta)+'-N-'+str(N)+'.npy') 

        plt.plot(z,n[:,N//2,N//2]/rhob,label='RF')
        plt.plot(zRF,rhoRF/rhob,label='WBII')
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

        # plt.imshow(n0[:,:,N//2].real, cmap='Greys_r')
        # plt.colorbar(label='$\\rho(x,y,0)/\\rho_b$')
        # plt.xlabel('$x$')
        # plt.ylabel('$y$')
        # plt.show()

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