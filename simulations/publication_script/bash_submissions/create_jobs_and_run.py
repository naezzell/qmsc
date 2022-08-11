#!/usr/bin/env python
import os
import subprocess
import sys
import numpy as np


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
        
data_dir = f"{os.getcwd()}/pub_data"
job_directory = f"{os.getcwd()}/.job"
out_directory = f"{os.getcwd()}/.job/.out"
#data_dir = os.path.join(scratch, 'pub_data')
mkdir_p(data_dir)
mkdir_p(job_directory)
mkdir_p(out_directory)

def main(test, ansatz, input_list):
    """
    Submits many jobs.
    """
    
    for inp in input_list:
        time = inp[-2]
        job_name = inp[-1]
        inp = inp[0:-2]
        inp.append(data_dir)
        inp.append(1)
        inp = [str(x) for x in inp]
        job_file = os.path.join(job_directory, f"{job_name}.sh")
        with open(job_file, 'w') as fh:
            # standard SLURM script header
            if test == 0:
                fh.writelines("#!/bin/bash\n")
            else:
                fh.writelines("#!/usr/bin/env bash\n")
            fh.writelines(f"#SBATCH --job-name={job_name}\n")
            fh.writelines(f"#SBATCH --output=.out/{job_name}.out\n")
            fh.writelines(f"#SBATCH --error=.out/{job_name}.err\n")
            fh.writelines("#SBATCH --nodes=1\n")
            fh.writelines("#SBATCH --ntasks=8\n")
            fh.writelines(f"#SBATCH --time={time}\n")
            fh.writelines(f"#SBATCH --account=\n")
            fh.writelines("#SBATCH --mail-type=FAIL\n")
            fh.writelines("#SBATCH --mail-user=\n")
            fh.writelines("#SBATCH --no-requeue\n")
            fh.writelines("#SBATCH --signal=23@60\n")
            # actual script for running python code
            if test == 0:
                fh.writelines("eval $(conda shell.bash hook)\n")
                fh.writelines("conda activate qmsc\n")
            #print(f"python {os.getcwd()}/../script_publication_vspa.py '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s'" % tuple(inp[0:11]))
            if ansatz == "vspa":
                fh.write(f"python {os.getcwd()}/../script_publication_vspa.py '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s'" % tuple(inp[0:11]))
            elif ansatz == "vcaa":
                fh.write(f"python {os.getcwd()}/../script_publication_ccps.py '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s'" % tuple(inp[0:11]))
            #fh.write(f"python {os.getcwd()}/../script_publication_vspa.py '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s'" % tuple(inp))

            if test == 1:
                print(f"Trying to submit: {job_file}")
                subprocess.Popen(["bash", job_file])
                print("Success?")
            else:
                print(f"Trying to submit: {job_file}")
                subprocess.Popen(["sbatch", job_file])
                print("Success?")





init_beta = 1e-3

# small system sizes xy model inputs
small_xy_input_list = []
na_choices = 0
max_perturbations = 0
opt_tol = 1e-16
diff_tol = 1e-8
s_lidx = 0
s_uidx = 24
state_types = ['xy_T_0.05', 'xy_T_0.5', 'xy_T_5.0']
for ns in range(1, 3):
    time = "1:00:00"
    job_name = f"xy_{ns}_T_{T}"
    for st in state_types:
        inp = [st, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
        small_xy_input_list.append(inp)
    

medium_xy_input_list = []
max_perturbations = 2
for st in state_types:
    ns = 3
    job_name = f"xy_{ns}"
    time = "5:00:00"
    inp = [st, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
    medium_xy_input_list.append(inp)
    ns = 4
    job_name = f"xy_{ns}"
    time = "10:00:00"
    inp = [st, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
    medium_xy_input_list.append(inp)
    ns = 5
    job_name = f"xy_{ns}"
    time = "15:00:00"
    inp = [st, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
    medium_xy_input_list.append(inp)
    ns = 6
    job_name = f"xy_{ns}"
    time = "20:00:00"
    inp = [st, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
    medium_xy_input_list.append(inp)
    

large_xy_input_list = []
max_perturbations = 3
opt_tol = 1e-16
diff_tol = 1e-8
s_lidx = 0
s_uidx = 12
time = "24:00:00"
for st in state_types:
    for ns in range(7, 10):
        job_name = f"xy_{ns}"
        inp = [st, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
        large_xy_input_list.append(inp)
max_perturbations = 3
opt_tol = 1e-16
diff_tol = 1e-8
s_lidx = 13
s_uidx = 24
time = "24:00:00"
for st in state_types:
    for ns in range(7, 10):
        job_name = f"xy_{ns}"
        inp = [st, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
        large_xy_input_list.append(inp)
        

bures_list = []
state_type = "bures"
na_choices = 0
max_perturbations = 0
opt_tol = 1e-16
diff_tol = 1e-8
s_lidx = 0
s_uidx = 24

ns = 1
time = "2:00:00"
job_name = f"bures_{ns}"
inp = [state_type, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
ns = 2
time = "8:00:00"
job_name = f"bures_{ns}"
inp = [state_type, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
ns = 3
time = "24:00:00"
job_name = f"bures_{ns}"
max_perturbations = 2
inp = [state_type, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
ns = 4
time = "24:00:00"
job_name = f"bures_{ns}"
max_perturbations = 2
inp = [state_type, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, time, job_name]
    

                
if __name__ == "__main__":
    # parse inputs
    test = sys.argv[1]
    test = int(test)
    # bures
    #main(test, "vspa", bures_list)
    #main(test, "vcca", bures_list)
    # large xy
    #main(test, "vspa", large_xy_input_list)
    #main(test, "vcca", large_xy_input_list)
    # medium xy
    #main(test, "vspa", medium_xy_input_list)
    #main(test, "vcca", medium_xy_input_list)
    # small xy
    #main(test, "vspa", small_xy_input_list)
    main(test, "vcca", small_xy_input_list)
