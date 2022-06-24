# PyDFT3D
An python script implementation of classical Density Functional Theory.

The grand potential is written as 
$$ \Omega[\{\rho_i (\boldsymbol{r})\}] = F[\{\rho_i (\boldsymbol{r})\}] + \sum_i \int \text{d}\boldsymbol{r}\ \left[ V^\text{ext}_i(\boldsymbol{r}) - \mu_i \right]\rho_i(\boldsymbol{r})$$

where $F $ is the free-energy functional, $V^{\text{ext}}_i $ is the external potential, and $\mu_i $ is the chemical potential of species $i $. The free-energy functional  can be written as a sum 
$$F = F^\text{id} + F^\text{exc}$$
where $F^\text{id} $ is the ideal gas contribution and $F^\text{exc} $ is the excess contribution.

The ideal-gas contribution $F^\text{id} $ is given by the exact expression
$$F^\text{id}[\{\rho_i (\boldsymbol{r})\}] = k_B T\sum_i \int_{V} d\boldsymbol{r}\ \rho_i(\boldsymbol{r})[\ln(\rho_i (\boldsymbol{r})\Lambda_i^3)-1]$$

where $k_B $ is the Boltzmann constant, $T $ is the absolute temperature, and $\Lambda_i$ is the well-known thermal de Broglie wavelength of each ion.

The excess Helmholtz free-energy, $F^\text{exc} $, is the free-energy functional due to particle-particle interactions splitted in the form

$$
F^\text{exc}[\{\rho_i(\boldsymbol{r})\}] = F^\text{hs}[\{\rho_i(\boldsymbol{r})\}] + F^\text{att}[\{\rho_i(\boldsymbol{r})\}]
$$

where $F^{\text{hs}} $ is the hard-sphere repulsive interaction excess contribution and $F^{\text{att}}$ is the attractive interaction excess contribution. 

The hard-sphere contribution, $F^{\text{hs}} $, represents the hard-sphere exclusion volume correlation and it can be described using different formulations of the fundamental measure theory (FMT) as

- [x] **R**osenfeld **F**unctional (**RF**) - [Rosenfeld, Y., Phys. Rev. Lett. 63, 980–983 (1989)](https://link.aps.org/doi/10.1103/PhysRevLett.63.980)
- [x] **W**hite **B**ear version **I** (**WBI**) - [Yu, Y.-X. & Wu, J., J. Chem. Phys. 117, 10156–10164 (2002)](http://aip.scitation.org/doi/10.1063/1.1520530); [Roth, R., Evans, R., Lang, A. & Kahl, G., J. Phys. Condens. Matter 14, 12063–12078 (2002)](https://iopscience.iop.org/article/10.1088/0953-8984/14/46/313)
- [x] **W**hite **B**ear version **II** (**WBII**) - [Hansen-Goos, H. & Roth, R. J., Phys. Condens. Matter 18, 8413–8425 (2006)](https://iopscience.iop.org/article/10.1088/0953-8984/18/37/002)