# RELION 4.0.1 on the grete GPU nodes (zen2). Source directly -- piping it runs a
# subshell and the modules vanish.
source /usr/share/lmod/lmod/init/bash
module load gcc/13.2.0
module load openmpi/5.0.7
module load gcc/13.2.0-nvptx      # without this relion silently stays off PATH
module load openmpi/5.0.7
module load relion/4.0.1
