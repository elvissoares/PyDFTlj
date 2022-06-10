import numpy as np
from scipy.optimize import minimize
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-03

" ==== FIRE: Fast Inertial Relaxation Engine ===== "

" References: "
"- Bitzek, E., Koskinen, P., Gähler, F., Moseler, M., & Gumbsch, P. (2006). Structural relaxation made simple. Physical Review Letters, 97(17), 1–4. https://doi.org/10.1103/PhysRevLett.97.170201"
"- Guénolé, J., Nöhring, W. G., Vaid, A., Houllé, F., Xie, Z., Prakash, A., & Bitzek, E. (2020). Assessment and optimization of the fast inertial relaxation engine (FIRE) for energy minimization in atomistic simulations and its implementation in LAMMPS. Computational Materials Science, 175. https://doi.org/10.1016/j.commatsci.2020.109584"


" Global variables for the FIRE algorithm"
Ndelay = 20
Nmax = 10000
finc = 1.1
fdec = 0.5
fa = 0.99
Nnegmax = 2000

def optimize_fire(x0,f,df,params,alpha0=0.62,rtol=1e-6,dt=40.0,logoutput=False):
    error = 10*rtol 
    dtmax = 10*dt
    dtmin = 0.02*dt
    alpha = alpha0
    Npos = 0

    x = x0.copy()
    V = np.zeros(x.shape)
    F = -df(x,params)

    error0 = max(np.abs(F.min()),F.max())

    for i in range(Nmax):

        P = (F*V).sum() # dissipated power
        
        if (P>0):
            Npos = Npos + 1
            if Npos>Ndelay:
                dt = min(dt*finc,dtmax)
                alpha = alpha*fa
        else:
            Npos = 0
            dt = max(dt*fdec,dtmin)
            alpha = alpha0
            V = np.zeros(x.shape)

        V = V + 0.5*dt*F
        V = (1-alpha)*V + alpha*F*np.linalg.norm(V)/np.linalg.norm(F)
        x = x + dt*V
        F = -df(x,params)
        V = V + 0.5*dt*F

        error = max(np.abs(F.min()),F.max())
        if error/error0 < rtol: break

        if logoutput: print(i,f(x,params),error/error0)

    del V, F  
    return [x,f(x,params),i]

def optimize_fire2(x0,f,df,params,alpha0=0.2,rtol=1e-4,dt=40.0,logoutput=False):
    error = 10*rtol 
    dtmax = 2*dt
    dtmin = 0.02*dt
    alpha = alpha0
    Npos = 0
    Nneg = 0

    x = x0.copy()
    V = np.zeros_like(x)
    F = -df(x,params)

    error0 = max(np.abs(F.min()),F.max())

    for i in range(Nmax):

        P = np.sum(F*V) # dissipated power
        
        if (P>0):
            Npos = Npos + 1
            Nneg = 0
            if Npos>Ndelay:
                dt = min(dt*finc,dtmax)
                alpha = alpha*fa
        else:
            Npos = 0
            Nneg = Nneg + 1
            if Nneg > Nnegmax: break
            if i> Ndelay:
                dt = max(dt*fdec,dtmin)
                alpha = alpha0
            x = x - 0.5*dt*V
            V = np.zeros_like(x)
            
        V = V + 0.5*dt*F
        V = (1-alpha)*V + alpha*F*np.linalg.norm(V)/np.linalg.norm(F)
        x = x + dt*V
        F = -df(x,params)
        V = V + 0.5*dt*F

        error = max(np.abs(F.min()),F.max())
        if error/error0 < rtol: break

        if logoutput: print(i,f(x,params),error/error0)

    del V, F  
    return [x,f(x,params),i]


##### Take a example using Rosenbrock function ######
if __name__ == "__main__":

    # The Rosenbrock function
    def gradf(x,params):
        [a,b] = params
        return np.array([-2*(a-x[0])-4*b*(x[1]-x[0]*x[0])*x[0],2*b*(x[1]-x[0]*x[0])])

    def f(x,params):
        [a,b] = params
        return (np.power((a-x[0]),2)+b*np.power((x[1]-x[0]*x[0]),2))

    p = [1,100]
    x0 = np.array([3.0,4.0])

    [xmin,fmin,Niter] = optimize_fire(x0,f,gradf,p,rtol=1e-6)

    print("xmin = ", xmin)
    print("fmin = ", fmin)
    print("Iterations = ",Niter)

    [xmin,fmin,Niter] = optimize_fire2(x0,f,gradf,p,rtol=1e-6)

    print("xmin = ", xmin)
    print("fmin = ", fmin)
    print("Iterations = ",Niter)
