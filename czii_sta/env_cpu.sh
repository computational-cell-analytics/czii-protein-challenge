# RELION 4.0.1 on the CPU nodes (sapphirerapids). Different arch => different
# compiler/MPI from the GPU nodes; the -nvptx sequence fails here.
source /usr/share/lmod/lmod/init/bash
module load gcc/11.5.0
module load openmpi/4.1.7
module load relion/4.0.1
