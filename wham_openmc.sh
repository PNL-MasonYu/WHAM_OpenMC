#!/bin/sh
#This file is called submit-script.sh
#SBATCH --partition=univ2       # default "univ2", if not specified
#SBATCH --time=5-00:00:00       # run time in days-hh:mm:ss
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20    # cpus per node (by default, "ntasks"="cpus")
#SBATCH --mem=128000             # RAM per node
#SBATCH --error=univ2job.err
#SBATCH --output=univ2job.out
# Make sure to change the above two lines to reflect your appropriate
# file locations for standard error and output

# Now list your executable command (or a string of them).
# Example for code compiled with a software module:
module load openmpi hdf5
#/software/myu233/openmc/build/bin/openmc -s 20
mpiexec -n 5 /software/myu233/openmc/build/bin/openmc -s 20
