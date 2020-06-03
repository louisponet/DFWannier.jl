#!/bin/bash
#SBATCH -J NiO_afm 
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --time=128:00:00
#SBATCH -p amd-hp
source /etc/profile.d/modules.sh
module load qe/qe-6.4
#mpirun -np 64 /home/ponet/Software/qe_build/lin_mixing/bin/pw.x -nk 8 -ndiag 8 < scf.in > scf.out
#mpirun -np 64 /home/ponet/Software/qe_build/lin_mixing/bin/pw.x -nk 8 -ndiag 8 < bands.in > bands.out
#mpirun -np 64 /home/ponet/Software/qe_build/lin_mixing/bin/pw.x -nk 8 -ndiag 8 < nscf.in > nscf.out
#mpirun -np 64 /home/ponet/Software/qe_build/lin_mixing/bin/projwfc.x < projwfc.in > projwfc.out
#/home/ponet/Software/qe_build/lin_mixing/bin/wannier90.x -pp wanup.win > wanup.wout
#mpirun -np 64 /home/ponet/Software/qe_build/lin_mixing/bin/pw2wannier90.x < pw2wan_wanup.in > pw2wan_wanup.out
wannier90.x wanup.win > wanup.wout
#/home/ponet/Software/qe_build/lin_mixing/bin/wannier90.x -pp wandn.win > wandn.wout
#mpirun -np 64 /home/ponet/Software/qe_build/lin_mixing/bin/pw2wannier90.x < pw2wan_wandn.in > pw2wan_wandn.out
wannier90.x wandn.win > wandn.wout
