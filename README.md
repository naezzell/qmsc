# QMSC
This package contains code related to the Quantum Mixed State Compiling (QMSC) algorithm presented in the preprint "Quantum Mixed State Compiling" by Nic Ezzell, Elliott M. Ball, Aliza U. Siddiqui, Mark M. Wilde, Andrew T. Sornborger, Patrick J. Coles, and Zoë Holmes. The goal of qmsc is to compile a potentially unknown mixed state coherently (i.e. on the same device). One can also do so incoherently (uploading from one device to another), but our code does not attempt to provide an implementation for incoherent uploading at the moment.

In this README, we shall discuss (i) how to download and install qmsc (ii) the package structure and useage and (iii) a dicussion of which parts of the repo are relevant to our paper and how to obtain/ analyze our data.

## (i) How to download and install qmsc
Simply clone the repo in your preferred directory and run `pip install -e .` within your desired python virtual environment/ conda environment while located in the edd directory with the setup.py file. If this doesn't make sense to you, please refer to the more detailed instructions in "setup_commands.txt." 

## (ii) Package structure and usage
We modeled our package structure after `numpy`, but by no means did we try to make things "production" quality. In other words, don't be surprised to find defunct functions or weird conventions. We also make no promises that our code will be maintained as Qiskit changes.

### Summary of structure

All the source code is located in `src/qmsc`. This main directory has several sub-directories
- `ansatz` -- contains the `FlexibleAnsatz` class to create variational quantum circuit ansatze
- `backend` -- contains the `IBMQBackend` class for sending/receiving jobs from IBM
- `circuit` -- contains the `AnsatzCirc` class (inhereits from qiskit QuantumCircuit) which allows more interactive circuit ansatz building; one can also convert a `FlexibleAnsatz` to a `AnsatzCirc` at any time
- `hamiltonian` -- contains utils related to building Hamiltonians and their associated thermal states
- `qstate` -- contains utils related to manipulating and processing a quantum state (i.e. building projectors, turning a state from a vector to a dictionary mapping computational basis elements to probabiltiies, etc...)
- `cost` -- contains `Cost` class for evaluating various costs given a circuit (not used in the end, but still contains some useful stuff/ ideas so we included it)

### Usage
The basic idea is to
1. Generate an ansatz for a mixed state using `FlexibleAnsatz` class (in `ansatz` module)
2. Convert the ansatz into a `AnsatzCirc` (in `circuit` module)
3. Evaluate costs on locally or on hardware using `IBMQBackend` (in `backend` module)
There are also several utilties to take advantage of
1. `qstate` contains utilities to build quantum states (i.e. Bures, random XY chains, etc...) and manipulate them
2. `hamiltonian` contains utilites to generate Hamiltonians that can be used for `qstate` or just to analyze the spectrum

A better understanding of the usage can be gleaned from `tutorials` folder which contains a walk-through of the major methods of the above classes and utilties. 

## (iii) Relevance to paper and data availibility
We include the files necessary to run our numerical simulations as well as our hardware implementations. Further, we provide the data and analysis/plotting of it within this repo. In particular,

1. `data_and_plotting` -- contains raw data and analysis/plotting
2. `simulations` -- contains files necessary to run our numerical simulations on HPC (or locally) with which we generated much of the data in data_and_plotting
3. `implementations` -- contains hardware implementation script examples with which we generated the hardware data in data_and_plotting

### simulations folder explanation
Within the simulations folder, we have the following files

1. `interactive_x_script.ipynb` -- to run Jupyter notebook version of the ccps/vspa qmsc code (x stands for ccps/vpsa here)
2. `qmsc_x_script.py` -- to run a command line (CLI) version of script more useful for batching many jobs
3. `generate_target_states.ipynb` -- used to generate random target states of various classes (our actual states used are in bash_submission)
4. `bash_submissions/run_script.sl` -- a bash/slurm script to run scripts from CLI or on HPC
5. `bash_submissions/create_jobs_and_run.py` -- a python script to generate many such "run_scripts" with different parameters and run locally or on HPC
6. `bash_submissions/random_x.pkl` -- contains the random states we input to our algorithm as pickled python objects