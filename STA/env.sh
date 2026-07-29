# Source before running RELION (on the Jupyter desktop or a GPU compute node):
#   source env.sh
#
# Uses the cluster's GPU-enabled RELION module. (The old self-built
# ~/software/relion install has been removed -- it was CPU-only and broken:
# configured CUDA=ON but without nvcc, so 0 GPU kernels were compiled.)
source ~/.bashrc 2>/dev/null || true
module load gcc/13.2.0 openmpi/5.0.7 relion/4.0.1 2>/dev/null || \
  echo "WARN: could not load relion/4.0.1 (try: module load gcc/13.2.0 openmpi/5.0.7 relion/4.0.1)"
# -> provides GPU relion_refine / relion_refine_mpi AND a matching mpirun.

# Fallback for mpirun only if the module did not put one on PATH:
command -v mpirun >/dev/null 2>&1 || export PATH=/sw/rev/25.04/cascadelake_opx_cuda70_rocky8/linux-rocky8-cascadelake/gcc-13.2.0/openmpi-5.0.7-sz7qoklrajggjuqugukwhvhybcdlzowh/bin:$PATH
