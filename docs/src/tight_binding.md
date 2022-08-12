# Tight Binding
Various tight binding operators can be generated using the outputs of Wannier90.
```@meta
CurrentModule = DFWannier
```

```@docs
TBBlock
TBOperator
```
The following functions can be used to generate specific operators.

```@docs
read_hamiltonian
read_spin
read_r
```

## Reciprocal
The tight binding operators can be used to interpolate properties in reciprocal space.

```@docs
Hk
HamiltonianKGrid
WannierBand
wannierbands
``` 
