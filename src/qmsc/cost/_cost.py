import re
import numpy as np
import warnings
from qmsc.backend import IBMQBackend
import qmsc.qstate as qstate

class Cost():
    """
    Cost class is initialized with a circuit and methods compute
    different relevant cost funtions over the circuit.
    """

    def __init__(self, circ, reverse=True, simulate=True, backend_name=None,
                 slow_but_gen=False):
        self.circ = circ
        if simulate:
            if backend_name is None:
                if slow_but_gen is True:
                    self.s = circ.get_statevector_slow_but_general(reverse=reverse)
                else:
                    self.s = circ.get_statevector(reverse=reverse)
        else:
            self.backend = backend
            circ.link_with_backend(backend)
            t_circ = circ.get_transpiled_circ()
            job = backend.submit_job(t_circ)
            result = job.result()
            self.counts = result.get_counts()

    def set_statevector(self):
        """
        Sets statevector from self.circ.
        """
        self.s = self.circ.get_statevector()
        return

    def set_circ(self, circ):
        """
        Set self.circ to [circ].
        """
        self.circ = circ
        return

    ############################################################
    # Noiseless cost funcs
    ############################################################
    ##################################################
    # (Noiseless) Statevector cost funcs
    ##################################################
    def noiseless_svec_prob0_cost(self, qubits='all'):
        """
        Computes probability of all 0 bitstring over
        [qubits] by computing final statevector.
        Defaults to using full statevector if
        [qubits] == 'all'.
        """
        # get state computed at end of circ and num qubits
        n = self.circ.num_qubits
        # if using all qubits, just compute projection
        if qubits == 'all':
            # construct all 0 vector in 2**n space
            #all0 = np.zeros(2**n)
            #all0[0] = 1
            # compute inner product < 0 | \psi >
            amp = self.s[0]
            #amp = np.dot(all0, self.s)
            # return cost as 1 - prob0
            prob0 = np.abs(amp)**2
        # if subset, make projection operator and Tr[P \rho]
        else:
            # construct pure state density operator
            rho = qstate.utils.make_density_operator([self.s], [1])
            # construct project onto |0><0| on [qubits]
            proj = qstate.utils.make_zero_projector(qubits, n)
            # compute prob as Tr[ proj * \rho]
            prob0 = np.trace(np.matmul(proj, rho))

        # compute cost and cast as real just for type check
        cost = np.real(1 - prob0)

        return cost

    def noiseless_svec_innerprod_cost(self, answer, qubits='all'):
        """
        Computes probability of all 0 bitstring over
        [qubits] by computing final statevector.
        Defaults to using full statevector if
        [qubits] == 'all'.
        """
        # get state computed at end of circ and num qubits
        n = self.circ.num_qubits
        # if using all qubits, just compute projection
        if qubits == 'all':
            # compute inner product < answer | self.s >
            amp = np.dot(answer.conj(), self.s)
            # get fidelity overlap
            fid = np.abs(amp)**2
        # if subset, make projection operator and Tr[P \rho]
        else:
            n = int(np.log2(len(self.s)))
            # construct \rho (learned density op) over qubits
            rho_idxs = list(range(n))
            rho = qstate.utils.make_projector(self.s, rho_idxs, n)
            # construct \sigma desired density op over qubits
            sigma = qstate.utils.make_projector(answer, rho_idxs, n)
            # get \sig_j over just [qubits]
            sig_j = sigma.ptrace(qubits)
            # make overall projector to answer on [qubits]
            proj = qstate.utils.make_op_projector(sig_j, qubits, n)
            # compute fidelity
            fid = (proj * rho).tr()

        # compute cost and cast as real just for type check
        cost = np.real(1 - fid)

        return cost

    ##################################################
    # (Noiseless) Shot based cost funcs
    ##################################################
    def noiseless_shots_prob0_cost(self, qubits='all', shots=8000):
        """
        Computes probability of all 0 bitstring over
        [qubits] by "sampling" 8000 shots from state-vec.
        Defaults to using full statevector if
        [qubits] == 'all'.
        """
        # step 0 get probability of success
        p0 = 1 - self.noiseless_svec_prob0_cost(qubits)
        # generate sample
        sample = [np.random.choice([1, 0], p=[p0, 1-p0]) for _ in range(shots)]
        # get p0 estimate
        p0_hat = sum(sample) / shots
        # return cost
        cost = 1 - p0_hat

        return cost
    
    ############################################################
    # Quantum Hardware based cost funcs
    ############################################################
    def hardware_shots_prob0_cost(self, cbits='all'):
        """
        Computes probability of all 0 bitstring over
        [qubits] from running on actual hardware.
        Defaults to using full statevector if
        [qubits] == 'all'.
        """
        # get total number of measured bits
        keys_list = list(self.counts.keys())
        num_bits = len(keys_list[0])
        if cbits == 'all':
            cbits = list(range(num_bits))
        # sum up counts that have 0...0 on cbits
        reg_str = ""
        for bit in range(num_bits):
            if bit in cbits:
                reg_str += "[0]"
            else:
                reg_str += "."
        check_str = re.compile(reg_str)
        success_count = 0
        tot_count = 0
        for key in self.counts:
            if check_str.fullmatch(key) is not None:
                success_count += self.counts[key]
                tot_count += self.counts[key]
            else:
                tot_count += self.counts[key]

        p0 = success_count / tot_count
        cost = 1 - p0

        return cost

def check_contiguous(array):
    # sort array
    array.sort()

    # check if array[i] - array[i-1] != 1
    for i in range(1, len(array)):
        if array[i] - array[i - 1] != 1:
            return False

    return True
