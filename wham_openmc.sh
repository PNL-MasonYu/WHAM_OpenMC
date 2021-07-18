#!/bin/sh
#This file is called submit-script.sh
#SBATCH --partition=univ2       # default "univ2", if not specified
#SBATCH --time=2-00:00:00       # run time in days-hh:mm:ss
#SBATCH --nodes=4               # require 2 nodes
#SBATCH --ntasks-per-node=20    # cpus per node (by default, "ntasks"="cpus")
#SBATCH --mem=128000             # RAM per node
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
# Make sure to change the above two lines to reflect your appropriate
# file locations for standard error and output

# Now list your executable command (or a string of them).
# Example for code compiled with a software module:
module load openmpi
mpirun -n 80 /software/openmc/build/bin/openmc