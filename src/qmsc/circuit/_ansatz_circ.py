import numpy as np
import itertools
import qiskit
from qiskit import (QuantumCircuit, IBMQ, execute, transpile,
                    schedule as build_schedule, Aer)
from qiskit.converters import circuit_to_dag
import qmsc.circuit._va_circ_utils as utils
from qiskit.providers.aer import AerSimulator


class AnsatzCirc(QuantumCircuit):
    """
    AnsatzCirc is class for variational quantum circuits.
    Inherits from Qiskit QuantumCircuit.
    """

    def __init__(self, *args, name=None, ibmq_backend=None, mode='circ'):
        super(AnsatzCirc, self).__init__(*args, name=name)
        self.ibmq_backend = ibmq_backend
        self.gate_dict = None

    ##################################################
    # General Useful Utility Methods
    ##################################################
    def link_with_backend(self, ibmq_backend):
        """
        Links self with [ibmq_backend] which changes the default
        behavior of adding gates by converting all gates to
        native gates for [backend] before adding from then on.
        Also allows for get_transpiled_circ method to work.
        """
        self.ibmq_backend = ibmq_backend
        return

    def get_transpiled_circ(self):
        """ Returns transpiled circ provided linked with backend """
        try:
            return transpile(self, self.ibmq_backend.backend)
        except AttributeError:
            e = "ibmq_backend not defined. Try self.link_with_backend(backend)."
            raise AttributeError(e)

    def get_gate_count(self, trans=False, ibmq_backend=None):
        """Returns dictionary with which gates are in circuit and
        how many of them there are."""
        circ = self.copy()
        if trans:
            circ = transpile(circ, ibmq_backend.backend)
        circ_dag = circuit_to_dag(circ)
        gate_count = circ_dag.count_ops()

        return gate_count

    def get_phys_time(self, ibmq_backend='linked', sub=None):
        """Given an IBMQBackend called [backend], returns the total
        run_time of this circuit in ns. Substracts any gates in [sub]
        along with their count, i.e. if state prep is 2 u3s, then
        sub = {'u3': 2} and this would reduce phys_time by the time of
        2 u3 gates.

        NOTE: This assumes circuit is written in terms of native gates, but
        the methods we've developed do this automatically.
        """
        if ibmq_backend == 'linked':
            ibmq_backend = self.ibmq_backend
        time = 0
        gate_count = self.get_gate_count(True, ibmq_backend)
        gate_times = ibmq_backend.get_gate_times()
        native_gates = ibmq_backend.get_native_gates()
        for gate, count in gate_count.items():
            if gate in native_gates:
                time += gate_times[gate] * count

        if sub is not None:
            for gate, count in sub.items():
                if gate in native_gates:
                    time -= gate_times[gate] * count

        return time

    def add_all_measurements(self):
        """
        Adds self.measure(j, j) for all qubits in circuit.
        """
        for j in range(self.num_qubits):
            self.measure(j, j)
        return

    def remove_measurements(self):
        """
        Removes all measurements in circuit.
        """
        def ismeasurement(gate):
            return isinstance(gate, qiskit.circuit.measure.Measure)
        self.data = [data for data in self.data if not ismeasurement(data[0])]
        return

    def get_statevector(self, reverse=False, as_dict=False):
        """
        Returns statevector obtained by running the circuit assuming no
        noise. Removes measurements if present to get correct answer.
        """
        # check if circuit contains measurement
        gate_dict = self.get_gate_count()
        # remove measurements
        if 'measure' in gate_dict:
            self.remove_measurements()
        # add save state if not there
        if 'save_state' not in gate_dict:
            self.save_state()

        # get statevector
        backend = AerSimulator(method='statevector')
        if reverse:
            job = backend.run(self.reverse_bits(), shots=1)
        else:
            job = backend.run(self, shots=1)
        result = job.result()
        state = result.get_statevector(self)
        if as_dict:
            state = utils.get_dict_rep_of_vec(state)

        return state

    def get_unitary(self, reverse=False):
        """
        Gets unitary equivalent of running the circuit with no noise.
        Removes measurements if present to get correct answer.
        """
        # check if circuit contains measurement
        gate_dict = self.get_gate_count()
        if 'measure' in gate_dict:
            # remove measurements
            self.remove_measurements()
        # get unitary
        if reverse:
            job = execute(self.reverse_bits(), Aer.get_backend('unitary_simulator'))
        else:
            job = execute(self, Aer.get_backend('unitary_simulator'))
        unitary_results = job.result()
        return unitary_results.results[0].data.unitary

    def get_ideal_counts(self, shots=8000, reverse=True):
        """
        Returns counts for circuit when no noise is present. If
        no measurements added, employs self.measure_all().
        """
        # check if circuit contains measurement
        gate_dict = self.get_gate_count()
        if 'measure' not in gate_dict:
            # add all possible measurements
            self.add_all_measurements()
        # get statevector
        if reverse:
            job = execute(self.reverse_bits(), Aer.get_backend('aer_simulator'), shots=shots)
        else:
            job = execute(self, Aer.get_backend('aer_simulator'), shots=shots)
        result = job.result()
        return result.get_counts()

    def cast_as_gate(self, label=None):
        """
        Turns entire circuit into one black-box gate.
        Note: Assumes no measurements.
        """
        unitary = self.get_unitary()
        gate = qiskit.extensions.UnitaryGate(unitary, label)
        return gate

    ##################################################
    # Useful encoding (i.e. Choi state)
    ##################################################
    def encode_choi(self, qlist1, qlist2):
        """
        Prepares Choi state that pairs as
        (qlist1[j] <--> qlist2[j]).
        """
        # dimension checks
        if len(qlist1) != len(qlist2):
            e = "qlist1 and 2 must be same len"
            raise ValueError(e)
        if len(set(qlist1) & set(qlist2)):
            e = "qlist1 and 2 must be disjoint"
            raise ValueError(e)

        all_q = qlist1 + qlist2
        cq_list1 = list(range(len(qlist1)))
        cq_list2 = np.array(list(range(len(qlist2))))
        cq_list2 = list(cq_list2 + max(cq_list1) + 1)
        # create choi encoding circ then append to self
        choi_circ = AnsatzCirc(len(all_q))
        choi_circ.h(cq_list1)
        for j in range(len(qlist1)):
            choi_circ.cnot(cq_list1[j], cq_list2[j])
        # turn into gate and append to self
        self.compose(choi_circ, all_q, inplace=True)

        return choi_circ

    def decode_choi(self, qlist1, qlist2):
        """
        Decodes Choi state that pairs as
        (qlist1[j] <--> qlist2[j]).
        """
        if len(qlist1) != len(qlist2):
            e = "qlist1 and 2 must be same len"
            raise ValueError(e)
        if len(set(qlist1) & set(qlist2)):
            e = "qlist1 and 2 must be disjoint"
            raise ValueError(e)

        all_q = qlist1 + qlist2
        cq_list1 = list(range(len(qlist1)))
        cq_list2 = np.array(list(range(len(qlist2))))
        cq_list2 = list(cq_list2 + max(cq_list1) + 1)
        # create choi encoding circ then append to self
        choi_circ = AnsatzCirc(len(all_q))
        for j in reversed(range(len(qlist1))):
            choi_circ.cnot(cq_list1[j], cq_list2[j])
        choi_circ.h(cq_list1)
        # turn into gate and append to self
        self.compose(choi_circ, all_q, inplace=True)

        return choi_circ

    ##################################################
    # Add arbitrary 1 and 2 qubit gates
    ##################################################
    def add_arb_1q_gate(self, pvec, qubit):
        """
        Appends arb sing qubit gate onto paramed by
        [pvec] on [qubit]. In particular,
        pvec = [gamma, beta, alpha], and we append
        rz(\gamma)ry(\beta)rz(\alpha).
        """
        # check that dim(pvec) is legit
        n = 1
        dim = 3
        if np.shape(pvec) != (dim, ):
            e = f"pvec has shape: {np.shape(pvec)} instead of ({dim},)"
            raise ValueError(e)

        # create 1q gate and append to main circ
        one_q_circ = AnsatzCirc(n)
        one_q_circ.rz(pvec[0], 0)
        one_q_circ.ry(pvec[1], 0)
        one_q_circ.rz(pvec[2], 0)
        self.compose(one_q_circ, [qubit], inplace=True)

        return one_q_circ

    def add_arb_2q_gate(self, pvec, qubits):
        """
        Appends arb 2 qubit gate paramed by
        [pvec] onto [qubits]. pvec is a 15x1
        vector. Arb decomp uses 3 CNOTs...
        """
        # check that dim(pvec) is legit
        n = 2
        dim = 15
        if np.shape(pvec) != (dim, ):
            e = f"pvec has shape: {np.shape(pvec)} instead of ({dim},)"
            raise ValueError(e)

        # create 2q gate and append to main circ
        two_q_circ = AnsatzCirc(n)
        # arb 1q layer
        ps = pvec[0:3]
        two_q_circ.add_arb_1q_gate(ps, 0)
        ps = pvec[3:6]
        two_q_circ.add_arb_1q_gate(ps, 1)
        # cnot separator
        two_q_circ.cnot(1, 0)
        # rotation layer
        ps = pvec[6]
        two_q_circ.rz(ps, 0)
        ps = pvec[7]
        two_q_circ.ry(ps, 1)
        # cnot separator
        two_q_circ.cnot(0, 1)
        # rotation layer
        ps = pvec[8]
        two_q_circ.ry(ps, 1)
        # cnot layer
        two_q_circ.cnot(1, 0)
        # final arb 1q layer
        ps = pvec[9:12]
        two_q_circ.add_arb_1q_gate(ps, 0)
        ps = pvec[12: ]
        two_q_circ.add_arb_1q_gate(ps, 1)
        # cast as gate
        self.compose(two_q_circ, qubits, inplace=True)

        return two_q_circ


    ##################################################
    # Add variational layers
    ##################################################
    def add_1q_layer(self, pvec, qubits):
        """
        Adds parameterized (by [pvec]) 1q gate on [qubits].
        """
        # check that dim(pvec) is legit
        n = len(qubits)
        dim = 3 * n
        if np.shape(pvec) != (dim, ):
            e = f"pvec has shape: {np.shape(pvec)} instead of ({dim},)"
            raise ValueError(e)

        # create 1q layer circuit and add to main circ
        layer_circ = AnsatzCirc(n)
        for i in range(len(qubits)):
            layer_circ.add_arb_1q_gate(pvec[3 * i: 3 * (i + 1)], i)
        self.compose(layer_circ, qubits, inplace=True)

        return layer_circ


    def add_2q_layer(self, pvec, qubits):
        """
        Adds all possible 2 qubit gates to
        [qubits]. If n = len(qubits), then
        [pvec] is 15(n)(n-1)/2 x 1 vector.
        """
        # check that dim(pvec) is legit
        n = len(qubits)
        dim = int((15 * n * (n-1)) / 2)
        if np.shape(pvec) != (dim, ):
            e = f"pvec has shape: {np.shape(pvec)} instead of ({dim},)"
            raise ValueError(e)

        # make 2q layer circuit and add to main circ
        layer_circ = AnsatzCirc(n)
        for i, pair in enumerate(itertools.combinations(range(n), 2)):
            pvec_slice = pvec[15 * i: 15 * (i+1)]
            layer_circ.add_arb_2q_gate(pvec_slice, pair)
        self.compose(layer_circ, qubits, inplace=True)

        return layer_circ


    def add_l_local_z(self, pvec, qubits, l=1):
        """
        Applies layer of all possible l-local
        e^{i \gamma_{j1j2..jl}Z_j1Z_j2...Z_jl}
        diagonal operators.
        """
        # check that dim(pvec) is legit
        n = len(qubits)
        lq_list = list(range(n))
        combos = list(itertools.combinations(lq_list, l))
        dim = len(combos)
        if np.shape(pvec) != (dim, ):
            e = f"Given {len(pvec)} angles, but need {dim}."
            raise ValueError(e)

        # make l_local_z layer and add to main circ
        layer_circ = AnsatzCirc(n)
        # l = 1 is special case
        if l == 1:
            for i in lq_list:
                layer_circ.rz(pvec[i], i)
        # l = 2 special case
        elif l == 2:
            for i, pair in enumerate(combos):
                q1, q2 = pair
                layer_circ.rzz(pvec[i], q1, q2)
        # l > 2 requires transpile breakdown
        else:
            print("Not implemented right now.")
        self.compose(layer_circ, qubits, inplace=True)

        return layer_circ


    def add_ry_layer(self, pvec, qubits):
        """
        Applies layer of RY(\theta) over
        all [qubits].
        """
        # check that dim(pvec) is legit
        n = len(qubits)
        dim = n
        if np.shape(pvec) != (dim, ):
            e = f"pvec has len {len(pvec)} but should be {len(qubits)}"
            raise ValueError(e)

        # make ry layer and add to main circ
        layer_circ = AnsatzCirc(n)
        for i in range(n):
            layer_circ.ry(pvec[i], i)
        self.compose(layer_circ, qubits, inplace=True)

        return layer_circ


    def add_bravo_layer(self, pvec, qlist1, qlist2):
        """
        Appends all possible CZ_{ij} gates
        where i in qlist1 and j in qlist2.
        """
        # check that dim(pvec) is legit
        n1 = len(qlist1)
        n2 = len(qlist2)
        n = n1 + n2
        dim = 2 * n + n1
        if np.shape(pvec) != (dim, ):
            e = f"len pvec is {len(pvec)} should be {dim}"
            raise ValueError(e)

        # make Bravo layer and add to main circ
        layer_circ = AnsatzCirc(n)
        lq_list1 = list(range(n1))
        lq_list2 = list(np.array(list(range(n2))) + (max(lq_list1) + 1))
        # init RY layer
        pvec1 = pvec[0: n1]
        layer_circ.add_ry_layer(pvec1, lq_list1)
        pvec2 = pvec[n1: (n1+n2)]
        layer_circ.add_ry_layer(pvec2, lq_list2)
        # CZ layer with RY in middle
        combos = list(itertools.product(lq_list1, lq_list2))
        mid_loc = int(np.floor(len(combos) / 2)) - 1
        for i, pidx in enumerate(utils.out_to_in_idxs(len(combos))):
            pair = combos[pidx]
            layer_circ.cz(pair[0], pair[1])
            # RY in middle
            if i == mid_loc or mid_loc == -1:
                pvec1 = pvec[n: (2*n1) + n2]
                layer_circ.add_ry_layer(pvec1, lq_list1)
                pvec2 = pvec[(2*n1) + n2: 2*n]
                layer_circ.add_ry_layer(pvec2, lq_list2)
        # final RY
        pvec1 = pvec[2*n: (2*n) + n1]
        layer_circ.add_ry_layer(pvec1, lq_list1)
        # cast as gate
        all_qlist = qlist1
        all_qlist.extend(qlist2)
        self.compose(layer_circ, all_qlist, inplace=True)

        return layer_circ


    def add_2q_hef_layer(self, pvec, qubits, r=1):
        """
        Adds QAQC 2q layer blocks. (see
        Fig. 2 in variational quantum state
        diagonlization with Ryan LaRose.).
        Assumes [qubits] contiguous list.
        """
        # check that dim(pvec) is legit
        n = len(qubits)
        if n == 2:
            dim = 15 * r
        elif (n % 2 == 0) and (n > 2):
            dim = (15 * n) * r
        else:
            dim = (15 * (n - 1)) * r
        if np.shape(pvec) != (dim, ):
            e = f"pvec has shape: {np.shape(pvec)} instead of ({dim},)"
            raise ValueError(e)

        # make 2q block layer and append to main circ
        layer_circ = AnsatzCirc(n)
        lq_list = list(range(n))
        if n == 2:
            i2 = 0
            for _ in range(r):
                i1 = i2
                i2 = i1 + 15
                layer_circ.add_arb_2q_gate(pvec[i1: i2], [0, 1])
        else:
            g_idx = 0
            for j in range(r):
                if n % 2 == 0:
                    qf = n
                else:
                    qf = n - 2
                # add "even" to "odd" layer
                for q in range(0, qf, 2):
                    pvec_slice = pvec[15 * g_idx: 15 * (g_idx + 1)]
                    layer_circ.add_arb_2q_gate(pvec_slice, [q, q+1])
                    g_idx += 1
                # add "odd" to "even" layer
                qf = n
                for q in range(1, qf, 2):
                    pvec_slice = pvec[15 * g_idx: 15 * (g_idx + 1)]
                    layer_circ.add_arb_2q_gate(pvec_slice, [q, (q + 1) % n])
                    g_idx += 1
        self.compose(layer_circ, qubits, inplace=True)

        return layer_circ


    def add_qaqc_layer(self, pvec, qubits, r=1):
        """
        Adds a QAQC layer that consists of 2q
        even/odd odd/even layer sandwiched between
        two single qubit layers.
        """
        # check that dim(pvec) is legit
        n = len(qubits)
        if n == 2:
            dim = (15 + (3 * n)) * r + (3 * n)
        elif (n > 2) and (n % 2 == 0):
            dim = (15 * n + 3 * n) * r + (3 * n)
        else:
            dim = (15 * (n - 1) + 3 * n) * r + (3 * n)
        if np.shape(pvec) != (dim, ):
            e = f"pvec has shape: {np.shape(pvec)} instead of ({dim},)"
            raise ValueError(e)

        # make qaqc layer and append to main circ
        layer_circ = AnsatzCirc(n)
        lq_list = list(range(n))
        # 1q layer at beginning
        i1 = 0
        i2 = 3 * n
        layer_circ.add_1q_layer(pvec[i1: i2], lq_list)
        # r reps of hardware efficient 2q layer + 1q layer
        for j in range(r):
            i1 = i2
            if n == 2:
                i2 = i1 + 15
            elif (n % 2) == 0:
                i2 = i1 + (15 * n)
            else:
                i2 = i1 + (15 * (n - 1))
            layer_circ.add_2q_hef_layer(pvec[i1: i2], lq_list)
            # 1q layer
            i1= i2
            i2 = i1 + (3 * n)
            layer_circ.add_1q_layer(pvec[i1: i2], lq_list)
        # add to main circ
        self.compose(layer_circ, qubits, inplace=True)

        return layer_circ


    def add_vff_layer(self, pvec, qubits, r=1, l=1):
        """
        Adds a VFF layer that consists of
        2q-hef - Diag_Z -- 2q-hef^{\dag}.
        """
        # check that dim(pvec) is legit
        n = len(qubits)
        lq_list = list(range(n))
        combos = list(itertools.combinations(lq_list, l))
        z_dim = len(combos)
        if n == 2:
            dim = 15 * r + z_dim
        elif (n % 2 == 0) and (n > 2):
            dim = (15 * n) * r + z_dim
        else:
            dim = (15 * (n - 1)) * r + z_dim
        if np.shape(pvec) != (dim, ):
            e = f"pvec has shape: {np.shape(pvec)} instead of ({dim},)"
            raise ValueError(e)

        # make vff layer and append to main circ
        w_circ = AnsatzCirc(n)
        w_pvec = pvec[0: (dim - z_dim)]
        w_circ.add_2q_hef_layer(w_pvec, lq_list, r)
        z_circ = AnsatzCirc(n)
        z_pvec = pvec[(dim - z_dim): dim]
        z_circ.add_l_local_z(z_pvec, lq_list, l)

        # append to main circ
        self.compose(w_circ, qubits, inplace=True)
        self.compose(z_circ, qubits, inplace=True)
        self.compose(w_circ.inverse(), qubits, inplace=True)

        return w_circ, z_circ

    ##################################################
    # Implement various Trotterized Hamiltonians
    ##################################################
    def apply_trot1_xy(self, t, r, qpairs, Jx=1, Jy=1):
        """
        Applies [RXX(tJ/r)-RYY(tJ/r)]^r,
        i.e. Trotterized XX + YY Hamiltonian
        evolution.
        """
        # cast Jx/Jy as lists if number for convenience
        if isinstance(Jx, (int, float)):
            Jx = [Jx]
        if isinstance(Jy, (int, float)):
            Jy = [Jy]
        # check that len(qpairs), Jx, Jy agree on dimension
        num_p = len(qpairs)
        if len(Jx) != (num_p):
            e = f"Jx should have length {num_p}."
            raise ValueError(e)
        if len(Jx) != len(Jy):
            e = "Jx and Jy should have same length."
            raise ValueError(e)

        # get number of qubits
        seen = []
        for q in qpairs:
            if q[0] not in seen:
                seen.append(q[0])
            if q[1] not in seen:
                seen.append(q[1])
        n = len(seen)

        # create XY Trotterization and append to main circ
        trot_circ = AnsatzCirc(n)
        for _ in range(r):
            for i, qp in enumerate(qpairs):
                q1, q2 = qp
                trot_circ.rxx((2 * Jx[i] * t) / r, q1, q2)
                trot_circ.ryy((2 * Jy[i] * t) / r, q1, q2)
        self.compose(trot_circ, seen, inplace=True)

        return trot_circ
