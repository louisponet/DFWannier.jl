# DFWannier Documentation
This package is meant to be used as a post processing tool for Wannier90. It relies heavily on
[`DFControl`](https://louisponet.github.io/DFControl.jl/stable).
The main capabilities are
- Generating the Tight-Binding Hamiltonian, Spin and dipole operators
- Interpolating bands and other properties in _k_-space using said operators
- Generating the real space Wannier functions
- Calculate various Berry and geometric properties
- Calculate the magnetic exchange parameters for the Heisenberg model 

