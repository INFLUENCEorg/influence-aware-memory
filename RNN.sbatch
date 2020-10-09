#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

#The directory
#SBATCH --chdir=/tudelft.net/staff-umbrella/influence/slurm

# The default partition is the 'general' partition
#SBATCH --partition=influence,general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=long

# The default run (wall-clock) time is 1 minute
#SBATCH --time=168:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# The default number of CPUs per task is 1, however CPUs are always allocated per 2, so for a single task you should use 2
#SBATCH --cpus-per-task=8

# The default memory per node is 1024 megabytes (1GB)
#SBATCH --mem=10024

# Set mail type to 'END' to receive a mail when the job finishes (with usage statistics)
#SBATCH --mail-type=END

#SBATCH -x, --exclude=influ[5,6]

# Uncomment these lines when your job requires this software
module use /opt/insy/modulefiles
#module load matlab/R2018b


# Complex or heavy commands should be started with 'srun' (see 'man srun' for more information)
# (This is just an example, srun is of course not necessary for this command.)
export INFL=/tudelft.net/staff-umbrella/influence
export SUMO_HOME=/tudelft.net/staff-umbrella/influence/sumo
export PYTHONPATH=/tudelft.net/staff-umbrella/influence/sumo/tools
export IAM_HOME=/tudelft.net/staff-umbrella/influence/influence-aware-memory
echo "$CONFIG $SCENE $FLICKER $SLURM_JOB_ID" >> $IAM_HOME/job_log/job_log.txt
srun /bin/sh -c 'cd $IAM_HOME && singularity exec influence-aware-memory.img python3 ./experimentor.py $CONFIG $SCENE $FLICKER'