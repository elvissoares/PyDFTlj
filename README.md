# PyDFT3D
An python script implementation of classical Density Functional Theory.

The grand potential is written as 
$$ \Omega[\rho(\boldsymbol{r})] = F[\rho(\boldsymbol{r})] + \int \text{d}\boldsymbol{r}\ \left[ V_{ext}(\boldsymbol{r}) - \mu \right]\rho(\boldsymbol{r})$$
where $F$ is the free-energy functional, $V_{ext}$ is the external potential, and $\mu$ is the chemical potential. 