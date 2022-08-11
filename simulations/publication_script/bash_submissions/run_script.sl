#!/bin/bash
#SBATCH --job-name="ns_1-randMSL"
#SBATCH --time=10:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --output=%j   # output file name
#SBATCH --account=   # account name
#SBATCH --mail-user=   # email address
#SBATCH --mail-type=FAIL
#SBATCH --no-requeue   # do not requeue when preempted and on node failure
#SBATCH --signal=23@60  # send signal to job at [seconds] before end

# LOAD MODULEFILES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# Define simulations variables


state_type="xy_T_0.05"
ns=3
s_lidx=0
s_uidx=24
R_choices=3
max_perturbations=0
opt_tol=1e-8
diff_tol=1e-8
init_beta=1e-3
data_dir="pca_data/"
trials=1

#eval "$(conda shell.bash hook)"
#conda activate qmsc

python ../script_publication_ccps.py $state_type $ns $s_lidx $s_uidx $R_choices $max_perturbations $opt_tol $diff_tol $init_beta $data_dir $trials
