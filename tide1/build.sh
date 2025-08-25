#!/bin/tcsh


module purge
module load intel-oneapi-compilers/2022.0.1-gcc-11.2.0
module load openmpi/4.1.2-intel-2021.5.0
module load netcdf-c/4.8.1-openmpi-4.1.2-intel-2021.5.0
module load netcdf-fortran/4.5.3-openmpi-4.1.2-intel-2021.5.0
spack load /iynhcfb

make setup




