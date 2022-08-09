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
- `cost` -- contains `Cost` class for evaluating various costs given a circuit (mostly not used in the end, but still contains some useful stuff/ ideas)
