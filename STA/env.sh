# Source before running RELION on the Jupyter desktop (node agq004, cascadelake).
#   source env.sh
source ~/.bashrc 2>/dev/null || true
module load gcc/13.2.0 openmpi/5.0.7 cuda/12.6 2>/dev/null || true
export PATH=/user/muth9/u12095/software/relion/install/bin:$PATH
# Fallback: the exact OpenMPI 5.0.7 relion_refine_mpi was built against (cascadelake)
if ! command -v mpirun >/dev/null 2>&1; then
  export PATH=/sw/rev/25.04/cascadelake_opx_cuda70_rocky8/linux-rocky8-cascadelake/gcc-13.2.0/openmpi-5.0.7-sz7qoklrajggjuqugukwhvhybcdlzowh/bin:$PATH
fi
