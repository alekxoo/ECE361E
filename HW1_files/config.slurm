#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J HW1_gr0                        # Job name
#SBATCH -o HW1_gr0.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e HW1_gr0.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100-small                 # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=<UT_MAIL_INSERT_HERE>@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A CCR24007                       # Allocation name

# Other commands must follow all #SBATCH directives...
module load python3

# Environment code
source $WORK/HW1_virtualenv/bin/activate

# Launch code...
python -u $WORK/HW1_files/starter.py --batch_size=128 --epochs=25 --lr=0.01 > $WORK/HW1_files/out

# ---------------------------------------------------

