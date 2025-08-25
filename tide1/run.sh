#!/bin/tcsh
#SBATCH --job-name=job1
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=uo0780
#SBATCH --output=log.out
#SBATCH --error=log.err

limit stacksize 4g

setenv OMPI_MCA_pml "ucx"
setenv OMPI_MCA_btl self
setenv OMPI_MCA_osc "pt2pt"
setenv UCX_IB_ADDR_TYPE ib_global
# for most runs one may or may not want to disable HCOLL
setenv OMPI_MCA_coll "^ml,hcoll"
setenv OMPI_MCA_coll_hcoll_enable "0"
setenv HCOLL_ENABLE_MCAST_ALL "0"
setenv HCOLL_MAIN_IB mlx5_0:1
setenv UCX_NET_DEVICES mlx5_0:1
setenv UCX_TLS mm,knem,cma,dc_mlx5,dc_x,self
setenv UCX_UNIFIED_MODE y
setenv HDF5_USE_FILE_LOCKING FALSE
setenv OMPI_MCA_io "romio321"
setenv UCX_HANDLE_ERRORS bt


module purge
module load intel-oneapi-compilers/2022.0.1-gcc-11.2.0
module load openmpi/4.1.2-intel-2021.5.0
module load netcdf-c/4.8.1-openmpi-4.1.2-intel-2021.5.0
module load netcdf-fortran/4.5.3-openmpi-4.1.2-intel-2021.5.0
spack load /iynhcfb

srun -l --cpu_bind=verbose --hint=nomultithread --distribution=block:cyclic ./setup.x 16 8 


