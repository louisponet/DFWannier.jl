# Exchanges
```@meta
CurrentModule = DFWannier
```
Using the [`TBHamiltonian`](@ref TBOperator) and [`calc_exchanges`](@ref), it is possible to calculate the magnetic exchange parameters $J_{ij}$ between atoms $i$ and $j$ for the isotropic Heisenberg model:\\
$E = \sum_{i,j} J_{ij} \overrightarrow{S}_i \cdot \overrightarrow{S}_j$

This involves calculating the Green's functions $G$ and on-site magnetic field matrices $\Delta$, which then determine $J$ as
$J_{ij} = \frac{1}{2\pi} \int_{-\infty, E_f} d\varepsilon \Delta_i G_{ij}^{\downarrow}(\varepsilon) \Delta_j G_{ji}^{\uparrow}(\varepsilon).
See [Weak ferromagnetism in antiferromagnets: Fe$_2$O$_3$ and La$_2$CuO$_4$](https://elar.urfu.ru/bitstream/10995/111495/1/2-s2.0-33644554849.pdf).

```@docs
Exchange2ndOrder
Exchange4thOrder
calc_exchanges
```
