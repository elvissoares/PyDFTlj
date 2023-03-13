# PyDFTlj
An python library for calculations using the classical Density Functional Theory (cDFT) for Lennard-Jones fluids in 1D and 3D geometries.

## Dependencies

* [NumPy](https://numpy.org) is the fundamental package for scientific computing with Python.
* [SciPy](https://scipy.org/) is a collection of fundamental algorithms for scientific computing in Python.
* [PyFFTW](https://pyfftw.readthedocs.io/en/latest/) is a pythonic wrapper around FFTW, the speedy FFT library. 
* [Matplotlib](https://matplotlib.org/stable/index.html) is a comprehensive library for creating static, animated, and interactive visualizations in Python.
* *Optional*: [SciencePlots](https://github.com/garrettj403/SciencePlots) is a Matplotlib styles for scientific figures

## Installation

### Option 1: Using `setup.py`

Clone `PyDFTlj` repository if you haven't done it yet.

Go to `PyDFTlj`'s root folder, there you will find `setup.py` file, and run the command below:

```Shell
python setup.py install
```

### Option 2: Using pip to install directly from the GitHub repo

You can run

```Shell
pip install git+https://github.com/elvissoares/PyDFTlj
```

and then you will be prompted to enter your GitHub username and password/access token.

## cDFT basics

The cDFT is the extension of the equation of state to treat inhomogeneous fluids. For a fluid with temperature T, total volume V, and chemical potential $\mu$ specified, the grand potential, $\Omega$, is written as

$$\Omega[\rho(\boldsymbol{r})] = F[\rho (\boldsymbol{r})] +  \int_{V} [ V^{(\text{ext})}(\boldsymbol{r}) - \mu ]\rho(\boldsymbol{r}) d\boldsymbol{r}$$

where $F[\rho (\boldsymbol{r})] $ is the free-energy functional, $V^{(\text{ext})} $ is the external potential, and $\mu $ is the chemical potential. The free-energy functional  can be written as a sum $ F = F^\text{id} + F^\text{exc} $, where $F^\text{id} $ is the ideal gas contribution and $F^\text{exc}$ is the excess contribution.

The ideal-gas contribution $F^\text{id} $ is given by the exact expression

$$ F^{\text{id}}[\rho (\boldsymbol{r})] = k_B T\int_{V} \rho(\boldsymbol{r})[\ln(\rho (\boldsymbol{r})\Lambda^3)-1] d\boldsymbol{r}$$

where $k_B $ is the Boltzmann constant, and $\Lambda $ is the well-known thermal de Broglie wavelength.

The excess Helmholtz free-energy, $F^{\text{exc} }$, is the free-energy functional due to particle-particle interactions and can be splitted in the form

$$ F^{\text{exc}}[\rho (\boldsymbol{r})] = F^{\text{hs}}[\rho (\boldsymbol{r})] + F^{\text{att}}[\rho (\boldsymbol{r})] $$
where $F^{\text{hs}} $ is the hard-sphere repulsive interaction excess contribution and $F^{\text{att}} $ is the attractive interaction excess contribution. 

The hard-sphere contribution, $F^{\text{hs}} $, represents the hard-sphere exclusion volume correlation and it can be described using different formulations of the fundamental measure theory (FMT) as

- [x] **R**osenfeld **F**unctional (**RF**) - [Rosenfeld, Y., Phys. Rev. Lett. 63, 980–983 (1989)](https://link.aps.org/doi/10.1103/PhysRevLett.63.980)
- [x] **W**hite **B**ear version **I** (**WBI**) - [Yu, Y.-X. & Wu, J., J. Chem. Phys. 117, 10156–10164 (2002)](http://aip.scitation.org/doi/10.1063/1.1520530); [Roth, R., Evans, R., Lang, A. & Kahl, G., J. Phys. Condens. Matter 14, 12063–12078 (2002)](https://iopscience.iop.org/article/10.1088/0953-8984/14/46/313)
- [x] **W**hite **B**ear version **II** (**WBII**) - [Hansen-Goos, H. & Roth, R. J., Phys. Condens. Matter 18, 8413–8425 (2006)](https://iopscience.iop.org/article/10.1088/0953-8984/18/37/002)

The attractive contribution, $F^\text{att}$, of the Lennard-Jones potential can be described by several formulations as listed below:

- [x] **M**ean **F**ield **A**pproximation (**MFA**) - 
- [x] **B**ulk **D**ensity **A**pproximation (**BFD**) - 
- [x] **W**eighted **D**ensity **A**pproximation (**WDA**) - [Shen, G., Ji, X., & Lu, X. (2013). The Journal of Chemical Physics, 138(22), 224706.](http://aip.scitation.org/doi/10.1063/1.4808160)
- [x] **M**odified **M**ean-**F**ield **A**pproximation (**MMFA**) - [Soares, E. do A., Barreto, A. G., & Tavares, F. W. (2021). Fluid Phase Equilibria, 542–543, 113095.](https://doi.org/10.1016/j.fluid.2021.113095)
<!-- - [ ] **f**unctionalized **M**ean **S**pherical **A**pproximation (**fMSA**) - [Roth, R., & Gillespie, D. (2016). Journal of Physics Condensed Matter, 28(24), 244006.](http://dx.doi.org/10.1088/0953-8984/28/24/244006) -->

where [x] represents the implemented functionals.

The thermodynamic equilibrium is given by the functional derivative of the grand potential in the form 

$$ \frac{\delta \Omega}{\delta \rho(\boldsymbol{r})} = k_B T \ln(\rho(\boldsymbol{r}) \Lambda^3) + \frac{\delta F^\text{exc}[\rho]}{\delta \rho(\boldsymbol{r})}  +V^{(\text{ext})}(\boldsymbol{r})-\mu = 0$$


When necessary, we use the MBWR[^1] equation of state for Lennard-Jones Fluids. We also describe the direct correlation function using the double Yukawa potential from the FMSA[^2]. 

## Usage example

You can access our [examples](https://github.com/elvissoares/PyDFTlj/tree/master/examples) folder and you can find different applications of the PyDFTlj. 

### Lennard-Jones equation of State (lj_phasediagram.py)

|![Figure1](https://github.com/elvissoares/PyDFTlj/blob/master/figures/phasediagram_lennardjones.png)|![Figure2](https://github.com/elvissoares/PyDFTlj/blob/master/figures/pressure_lennardjones.png)|
|:--:|:--:|
| <b>Fig.1 - The phase diagram of the LJ fluid. The curve represents the MBWR EoS[^1]. </b>| <b>Fig.2 - The saturation pressure as a function of the inverse of the temperature. </b>|

### LJ fluid near a hardwall (lj1d_hardwall.py)

|![Figure3](https://github.com/elvissoares/PyDFTlj/blob/master/figures/lj1d-hardwall-rhob=0.5-T=1.35.png)|![Figure4](https://github.com/elvissoares/PyDFTlj/blob/master/figures/lj1d-hardwall-rhob=0.82-T=1.35.png)|
|:--:|:--:|
| <b>Fig.3 - The density profiles of LJ fluid near a hardwall with reduce temperature T*=1.35 and reduced density of ρ*=0.5. Symbols: MC  data. Lines: Different DFT formulations. </b>| <b>Fig.4 - The density profiles of LJ fluid near a hardwall with reduce temperature T*=1.35 and reduced density of ρ*=0.82. Symbols: MC  data. Lines: Different DFT formulations. </b>|

### LJ fluid inside a Slit-pore like (lj1d_slitpore.py)

|![Figure5](https://github.com/elvissoares/PyDFTlj/blob/master/figures/lj1d-slitpore-steele-T1.2-rhob0.5925-H7.5.png)|![Figure6](https://github.com/elvissoares/PyDFTlj/blob/master/figures/lj1d-slitpore-steele-T1.2-rhob0.5925-H3.0and2.0.png)|
|:--:|:--:|
| <b>Fig.5 - The density profiles of LJ fluid confined in slit-like pores at reduced density of ρ*=0.5925 and reduced temperature of T*=1.2 for H = 7.5. Symbols: MC data. Lines: Different DFT formulations.  </b>| <b>Fig.6 - The density profiles of LJ fluid confined in slit-like pores at reduced density of ρ*=0.5925 and reduced temperature of T*=1.2 for two pore widths. For clarity, the density profiles for H=2.0 are shifted left by 0.5. Symbols: MC data. Lines: Different DFT formulations. </b>|

### LJ fluid Radial Distribution Function (lj3d_rdf.py)

|![Figure7](https://github.com/elvissoares/PyDFTlj/blob/master/figures/lj3d-radialdistribution-rhob=0.84-T=0.71.png)|
|:--:|
|<b>Fig.7 - The radial distribution function of LJ fluid at reduced density of ρ*=0.84 and reduced temperature of T*=0.71. Symbols: MC data. Lines: Different DFT formulations.  </b>|

----
## References

[^1]: [Johnson, J. K., Zollweg, J. A., & Gubbins, K. E. (1993). The Lennard-Jones equation of state revisited. Molecular Physics, 78(3), 591–618.](https://www.tandfonline.com/doi/full/10.1080/00268979300100411)

[^2]: [Tang, Y., & Lu, B. C. Y. (2001). On the mean spherical approximation for the Lennard-Jones fluid. Fluid Phase Equilibria, 190(1–2), 149–158.](https://linkinghub.elsevier.com/retrieve/pii/S0378381201006008)