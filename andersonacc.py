import numpy as np
from scipy.optimize import minimize
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2021-12-01

" ==== Anderson Acceleration Mixing Method ===== "

" References: "
"- Mairhofer, J., & Gross, J. (2017). Numerical aspects of classical density functional theory for one-dimensional vapor-liquid interfaces. Fluid Phase Equilibria, 444, 1-12. https://doi.org/10.1016/j.fluid.2017.03.023"
"- Walker, H. F., & Ni, P. (2011). Anderson acceleration for fixed-point iterations. SIAM Journal on Numerical Analysis, 49(4), 1715-1735. https://doi.org/10.1137/10078356X"


" Global variables for the Anderson algorithm"
Nmax = 10000

def optimize_anderson(x0,xmodel,f,df,params,beta=0.19,atol=1e-7,logoutput=False):
    m = 5
    alpha0 = (1.0/m)*np.ones(m)
    alpha = alpha0.copy()
    Niter = 0

    x = x0.copy()

    xstr = [np.zeros_like(x)]*m
    ustr = [np.zeros_like(x)]*m
    Fstr = [np.zeros_like(x)]*m

    flast = f(x0,params)

    while Niter < Nmax:

        F = df(x,params)
        u = xmodel(x,params)
        
        if Niter < m:
            xstr[Niter] = x
            Fstr[Niter] = F
            ustr[Niter] = u
        else:
            xstr[:m-1] = xstr[1:m]
            xstr[m-1] = x
            Fstr[:m-1] = Fstr[1:m]
            Fstr[m-1] = F
            ustr[:m-1] = ustr[1:m]
            ustr[m-1] = u

            def objective(alp):
                fobj = 0.0
                for l in range(m):
                    fobj += (alp[l]**2)*Fstr[l]
                return np.linalg.norm(fobj)/np.linalg.norm(alp)

            res = minimize(objective, np.sqrt(alpha), method='Nelder-Mead', tol=1e-6)
            alpha[:] = res.x**2
            alpha[:] = alpha/alpha.sum()

            x = np.zeros_like(x0)
            for l in range(m):
                x += (1-beta)*alpha[l]*xstr[l] + beta*alpha[l]*ustr[l]

            fnew = f(x,params)
            error = np.abs(fnew-flast)
            flast = fnew
            
            if logoutput: print(Niter,fnew,error)
            if error < atol: break
        Niter += 1

    del F, xstr, Fstr 
    return [x,f(x,params),Niter]