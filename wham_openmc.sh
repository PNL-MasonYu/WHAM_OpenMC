#!/bin/sh
#This file is called submit-script.sh
#SBATCH --partition=pre       # default "univ2", if not specified
#SBATCH --time=1-00:00:00       # run time in days-hh:mm:ss
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=20    # cpus per node (by default, "ntasks"="cpus")
#SBATCH --mem=64000             # RAM per node
#SBATCH --error=job.err
#SBATCH --output=job.out
# Make sure to change the above two lines to reflect your appropriate
# file locations for standard error and output

# Now list your executable command (or a string of them).
# Example for code compiled with a software module:
module load openmpi hdf5
mpirun -n 4 /software/myu233/openmc/build/bin/openmc -s 20
