import numpy as np
import itertools
from qmsc.circuit import AnsatzCirc

# inherited from object for backwards compatibility
class FlexibleAnsatz(object):
    """
    Flexible Ansatz contains data and update moves used to
    generate new ansatzs in circuit/ qasm form.
    """

    def __init__(self, n, cwa=None):
        """
        Instantiate [self] with [n] qubits. Non-default
        choices for [ansatz] are
        1. HEF_2-D
        """
        self.n = n
        self.cwa = []
        self.pnum = 0
        # insert ansatz which populates self.cwa/self.pnum
        if cwa == 'hef2d':
            qubits = list(range(n))
            self.insert_hef2d(qubits)
        elif cwa == '1local_z':
            qubits = list(range(n))
            self.insert_1local_z(qubits)
        elif cwa == 'arb1q':
            qubits = list(range(n))
            for q in qubits:
                self.insert_arb1q(q)


        return


    def __str__(self):
        ansatz_str = ""
        for tup in self.cwa:
            inst, params, qubits = tup
            ansatz_str += f"{inst}({params})_({qubits})\n"

        return ansatz_str


    def get_num_instructions(self):
        """
        Get number of instructions in [cwa].
        Roughly, this corresponds to # of
        parametrs necessary to specify it.
        """
        return len(self.cwa)


    def get_num_parameters(self):
        """
        Gets number of parameters that define
        this ansatz.
        """
        return self.pnum


    def get_insertion_locations(self):
        """
        Gets indices where a new circuit element
        can be inserted.
        """
        num_inst = self.get_num_instructions()
        locs = list(range(num_inst + 1))
        return locs


    def get_instructions(self):
        """
        Returns list of instructions in the order
        in which they are added to Circuit.
        """
        instructions = []
        for j in range(len(self.cwa)):
            instructions.append(self.cwa[j][0])

        return instructions


    def get_parameters(self):
        """
        Get flat list of parameters in the order
        they appear in the circuit.
        """
        flat_param_list = []
        for tup in self.cwa:
            params = tup[1]
            if params is not None:
                for p in params:
                    flat_param_list.append(p)

        return flat_param_list


    def resample_parameters(self, idres_to_idres=True):
        """
        Updates parameters with new random parameters.
        If [idres_to_idres] == True, doesn't update
        parameters which are 0. That is, identity
        resolutions stay identity.
        """
        new_params = []
        for j in range(len(self.cwa)):
            params = self.cwa[j][1]
            if params is not None:
                for k in range(len(params)):
                    if (params[k] == 0) and (idres_to_idres is True):
                        p = 0
                    else:
                        p = 2 * np.pi * np.random.random()
                    p = 2 * np.pi * np.random.random()
                    self.cwa[j][1][k] = p
                    new_params.append(p)

        return new_params


    def update_parameters(self, new_params):
        """
        Updates parameters with new [new_params].
        """
        g_idx = 0
        for j in range(len(self.cwa)):
            params = self.cwa[j][1]
            if params is not None:
                for k in range(len(params)):
                    self.cwa[j][1][k] = new_params[g_idx]
                    g_idx += 1

        return new_params

    def remove_identity_elements(self, atol=1e-2):
        """
        Remove single qubit gates and CNOTs that multiply contiguously to
        an identity.
        """
        new_inst = []
        # find instructions to remove
        for i, inst in enumerate(self.cwa):
            # if rotation, check if angle is 0
            if inst[0] in ['rx', 'ry', 'rz']:
                ang = inst[1][0]
                if np.abs(ang) > 2 * np.pi:
                    ang = (ang % (2 * np.pi))
                    inst[1][0] = ang
                is_zero = (np.abs(ang) < atol)
                if not is_zero:
                    new_inst.append(inst)
            # if CNOT, check if keeping or not
            if inst[0] == "cnot":
                if new_inst == []:
                    new_inst.append(inst)
                else:
                    if inst == new_inst[-1]:
                        new_inst.pop()
                    else:
                        new_inst.append(inst)
            elif inst[0] == "h":
                if new_inst == []:
                    new_inst.append(inst)
                else:
                    if inst == new_inst[-1]:
                        new_inst.pop()
                    else:
                        new_inst.append(inst)
            elif inst[0] == "crx":
                ang = inst[1][0]
                if np.abs(ang) > 2 * np.pi:
                    ang = (ang % (2 * np.pi))
                    inst[1][0] = ang
                is_zero = (np.abs(ang) < atol)
                if not is_zero:
                    new_inst.append(inst)
            elif inst[0] == "G":
                ang = inst[1][0]
                if np.abs(ang) > 2 * np.pi:
                    ang = (ang % (2 * np.pi))
                    inst[1][0] = ang
                is_zero = (np.abs(ang) < atol)
                if not is_zero:
                    new_inst.append(inst)
            elif inst[0] == "G2":
                ang = inst[1][0]
                if np.abs(ang) > 2 * np.pi:
                    ang = (ang % (2 * np.pi))
                    inst[1][0] = ang
                is_zero = (np.abs(ang) < atol)
                if not is_zero:
                    new_inst.append(inst)
        self.cwa = new_inst

        return new_inst


    def build_circ(self, circ_n, qubit_mapping='1-1'):
        """
        Build AnsatzCirc (inhereits from
        Qiskit QuantumCircuit) over [n] qubits
        that implements current workin ansatz,
        [self.cwa].
        [params] specify angles that paramaterize
        ansatz and are filled in the same order
        as [self.cwa].
        """
        circ = AnsatzCirc(circ_n)
        if qubit_mapping == '1-1':
            qubit_mapping = {}
            for q in range(self.n):
                qubit_mapping[q] = q
        for gate, params, qubits in self.cwa:
            if gate.lower() == 'rz':
                angle = params[0]
                mapped_qubits = [qubit_mapping[q] for q in qubits]
                circ.rz(angle, mapped_qubits)
            elif gate.lower() == 'ry':
                angle = params[0]
                mapped_qubits = [qubit_mapping[q] for q in qubits]
                circ.ry(angle, mapped_qubits)
            elif gate.lower() == 'rx':
                angle = params[0]
                mapped_qubits = [qubit_mapping[q] for q in qubits]
                circ.rx(angle, mapped_qubits)
            elif gate.lower() == 'cnot':
                q0, q1 = qubits
                mapped_q0 = qubit_mapping[q0]
                mapped_q1 = qubit_mapping[q1]
                circ.cnot(mapped_q0, mapped_q1)
            elif gate.lower() == 'h':
                mapped_qubits = [qubit_mapping[q] for q in qubits]
                circ.h(mapped_qubits)
            elif gate.lower() == 'crx':
                angle = params[0]
                q0, q1 = qubits
                mapped_q0 = qubit_mapping[q0]
                mapped_q1 = qubit_mapping[q1]
                circ.crx(angle, mapped_q0, mapped_q1)
            elif gate.lower() == "g":
                angle = params[0]
                q0, q1 = qubits
                mapped_q0 = qubit_mapping[q0]
                mapped_q1 = qubit_mapping[q1]
                g = givens_rotation(angle)
                circ.unitary(g, [mapped_q0, mapped_q1])
            elif gate.lower() == "g2":
                angle = params[0]
                q0, q1 = qubits
                mapped_q0 = qubit_mapping[q0]
                mapped_q1 = qubit_mapping[q1]
                g2 = givens_rotation2(angle)
                circ.unitary(g2, [mapped_q0, mapped_q1])
            else:
                e = f"Instruction {gate} not castable to AnsatzCirc."
                raise ValueError(e)

        return circ


    def build_qasm(self, circ_n, qubit_mapping='1-1'):
        """
        Build QASM string for ansatz for an
        [n] qubit "circuit."
        """
        circ = self.build_circ(circ_n, qubit_mapping)

        return circ.qasm()


    def insert_from_qasm(self, qasm):
        """
        Builds instructions from QASM string.
        """
        added = []
        ignore = ["OPENQASM", "include", "qreg", "creg", "barrier"]
        qasm_list = qasm.split("\n")
        for line in qasm_list:
            if sum([ignore[j] in line for j in range(len(ignore))]) != 0:
                continue
            elif line == "":
                continue
            else:
                parsed = line.split(' ')
                qubits = parsed[1].strip(';')
                qubits = [int(x[2:-1]) for x in qubits.split(',')]
                if parsed[0] == 'cx':
                    gate = 'cnot'
                    params = None
                elif parsed[0] == 'h':
                    gate = "h"
                    params = None
                else:
                    gate = parsed[0][0:2]
                    params = [float(parsed[0][3:-1])]
                inst = (gate, params, qubits)
                added.append(inst)

        self.cwa += added


    def insert_custom(self, inst_tup, loc='end'):
        """
        Insert a custom instruction tuple.
        """
        if loc == 'end':
            loc = self.get_num_instructions()
        self.cwa[loc:loc] = [inst_tup]
        if inst_tup[1] is not None:
            pnum = len(inst_tup[1])
        else:
            pnum = 0

        self.pnum += pnum

        return inst_tup, pnum


    def _build_arb1q(self, q):
        """
        Builds arbitrary 1 qubit gate acting
        on [q].

        Returns
        -----------------------------------
        * gate [list] -- contains arbitrary 1q
        gate as list of tuples (rotation, qubit).
        * pnum [int] -- # parameters to speicfy
        [gate]
        """
        gate = []
        pnum = 0
        phi = np.random.normal() % (2 * np.pi)
        gate.append(('rz', [phi], [q]))
        phi = np.random.normal() % (2 * np.pi)
        gate.append(('ry', [phi], [q]))
        phi = np.random.normal() % (2 * np.pi)
        gate.append(('rz', [phi], [q]))
        pnum = 3

        return gate, pnum


    def insert_arb1q(self, q, loc='end'):
        """
        Inserts arbitrary 1 qubit gate
        at location [loc] acting on [qubits].
        """
        if q > self.n - 1:
            e = f"Cannot use qubit {q} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()
        arb1q, arb1q_pnum = self._build_arb1q(q)
        self.cwa[loc:loc] = arb1q
        self.pnum += arb1q_pnum

        return arb1q, arb1q_pnum


    def insert_arb1q_layer(self, qubits, loc='end'):
        """
        Inserts arbitrary 1 qubit gate at location [loc]
        acting on [qubits].
        """
        if max(qubits) > self.n - 1:
            e = f"Cannot use qubit {max(qubits)} in {self.n} qubit ansatz!"
            raise ValueError(e)

        if loc == 'end':
            loc = self.get_num_instructions()

        gate = []
        tot_pnum = 0
        for q in qubits:
            arb1q, arb1q_pnum = self._build_arb1q(q)
            gate.extend(arb1q)
            tot_pnum += arb1q_pnum
            self.cwa[loc:loc] = arb1q
            self.pnum += arb1q_pnum
            loc += 3

        return gate, tot_pnum


    def insert_h(self, q, loc='end'):
        """
        Inserts Hadamard on [q] at [loc].
        """
        if q > self.n - 1:
            e = f"Cannot use qubit {q} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()
        hadamard = [('h', None, [q])]
        self.cwa[loc:loc] = hadamard

        return hadamard, 0


    def _build_arb2q(self, q0, q1):
        """
        Builds arbitrary 2 qubit gate acting
        on [q0] and [q1].

        Returns
        -----------------------------------
        * gate [list] -- contains arbitrary 1q
        gate as list of tuples (rotation, qubit).
        * pnum [int] -- # parameters to speicfy
        [gate]        """
        gate = []
        pnum = 0
        # add 1q layer
        gate_1q_q0, pnum_1q = self._build_arb1q(q0)
        gate_1q_q1, _ = self._build_arb1q(q1)
        gate.extend(gate_1q_q0)
        pnum += pnum_1q
        gate.extend(gate_1q_q1)
        pnum += pnum_1q
        # cnot separator
        cnot_q1_q0 = ('cnot', None, [q1, q0])
        gate.append(cnot_q1_q0)
        # add rotation layer
        phi = np.random.normal() % (2 * np.pi)
        rz_q0 = ('rz', [phi], [q0])
        gate.append(rz_q0)
        pnum += 1
        phi = np.random.normal() % (2 * np.pi)
        ry_q1 = ('ry', [phi], [q1])
        gate.append(ry_q1)
        pnum += 1
        # cnot separator
        cnot_q0_q1 = ('cnot', None, [q0, q1])
        gate.append(cnot_q0_q1)
        # rotation layer
        phi = np.random.normal() % (2 * np.pi)
        ry_q1 = ('ry', [phi], [q1])
        gate.append(ry_q1)
        pnum += 1
        # cnot layer
        cnot_q1_q0 = ('cnot', None, [q1, q0])
        gate.append(cnot_q1_q0)
        # final 1q layer
        gate_1q_q0, pnum_1q = self._build_arb1q(q0)
        gate_1q_q1, _ = self._build_arb1q(q1)
        gate.extend(gate_1q_q0)
        pnum += pnum_1q
        gate.extend(gate_1q_q1)
        pnum += pnum_1q

        return gate, pnum


    def _build_hef2d(self, qubits, r=1):
        """
        Builds generic hardware efficient 2-design
        asatz list acting on [qubits].

        Returns
        -----------------------------------
        * gate [list] -- contains arbitrary 1q
        gate as list of tuples (rotation, qubit).
        * pnum [int] -- # parameters to speicfy
        [gate]
        """
        # build up gate
        gate = []
        pnum = 0
        k = len(qubits)
        if k == 2:
            i2 = 0
            for _ in range(r):
                temp_gate, pnum_2q = self._build_arb2q(qubits[0], qubits[1])
                i1 = i2
                i2 = i1 + 15
                gate.extend(temp_gate)
                pnum += pnum_2q
        else:
            for j in range(r):
                if k % 2 == 0:
                    qf = k
                else:
                    qf = k - 2
                # add "even" to "odd" layer
                for q_idx in range(0, qf, 2):
                    q0 = qubits[q_idx]
                    q1 = qubits[q_idx + 1]
                    temp_gate, pnum_2q = self._build_arb2q(q0, q1)
                    gate.extend(temp_gate)
                    pnum += pnum_2q
                # add "odd" to "even" layer
                qf = k
                for q_idx in range(1, qf, 2):
                    q0 = qubits[q_idx]
                    q1 = qubits[(q_idx + 1) % k]
                    temp_gate, pnum_2q = self._build_arb2q(q0, q1)
                    gate.extend(temp_gate)
                    pnum += pnum_2q

        return gate, pnum


    def insert_cnot(self, q0, q1, loc='end'):
        """
        Inserts CNOT acting on [q0] and [q1].
        """
        if max([q0, q1]) > self.n - 1:
            e = f"Cannot use qubit {max([q0, q1])} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()
        cnot = [('cnot', None, [q0, q1])]
        self.cwa[loc:loc] = cnot

        return cnot, 0


    def insert_ghz_prep(self, qubits, loc='end'):
        """
        Prepares GHZ state on [qubits] assuming we start
        in the |00...0> state.
        """
        if max(qubits) > self.n - 1:
            e = f"Cannot use qubit {max([q0, q1])} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()

        ghz_prep = []
        hadamard = self.insert_h(qubits[0])
        ghz_prep.append(hadamard)
        for idx in range(len(qubits[1:])):
            q0 = qubits[idx]
            q1 = qubits[idx + 1]
            cnot = self.insert_cnot(q0, q1)
            ghz_prep.append(cnot)

        return ghz_prep, 0


    def insert_qgan_ghz_ansatz(self, qubits, loc='end'):
        """
        Prepares GHZ ansatz for QGANs as in Kiani, Llyod,
        Earth Mover's distance https://arxiv.org/abs/2101.03037.
        """
        if max(qubits) > self.n - 1:
            e = f"Cannot use qubit {max([q0, q1])} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()

        ghz_ansatz = []
        num_params = 0
        # build initial qubit rotation
        q0 = qubits[0]
        phi = np.random.normal() % (2 * np.pi)
        rx = ('rx', [phi], [q0])
        phi = np.random.normal() % (2 * np.pi)
        ry = ('ry', [phi], [q0])
        phi = np.random.normal() % (2 * np.pi)
        rz = ('rz', [phi], [q0])
        ghz_ansatz.append(rx)
        ghz_ansatz.append(ry)
        ghz_ansatz.append(rz)
        num_params += 3
        # build controlled RX cascade
        for q in qubits[1:]:
            phi = np.random.normal() % (2 * np.pi)
            crx = ('crx', [phi], [q-1, q])
            ghz_ansatz.append(crx)
            num_params += 1

        # insert ansatz actually
        self.cwa[loc:loc] = ghz_ansatz
        self.pnum += num_params

        return ghz_ansatz, num_params


    def insert_hef2d(self, qubits, loc='end'):
        """
        Inserts hardware efficient 2 design
        ansatz at location [loc] acting
        on [qubits].
        """
        if max(qubits) > self.n - 1:
            e = f"Cannot use qubit {max(qubits)} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()
        hef2d, hef_pnum = self._build_hef2d(qubits)
        self.cwa[loc:loc] = hef2d
        self.pnum += hef_pnum

        return hef2d, hef_pnum


    def _build_evff_arb2q(self, q0, q1):
        """
        Builds EVFF (arbitrary + extra) 2 qubit gate acting
        on [q0] and [q1].

        Returns
        -----------------------------------
        * gate [list] -- contains arbitrary 1q
        gate as list of tuples (rotation, qubit).
        * pnum [int] -- # parameters to speicfy
        [gate]        """
        gate = []
        pnum = 0
        # add 1q layer
        gate_1q_q0, pnum_1q = self._build_arb1q(q0)
        gate_1q_q1, _ = self._build_arb1q(q1)
        gate.extend(gate_1q_q0)
        pnum += pnum_1q
        gate.extend(gate_1q_q1)
        pnum += pnum_1q
        # cnot separator
        cnot_q1_q0 = ('cnot', None, [q1, q0])
        gate.append(cnot_q1_q0)
        # add rotation layer
        phi = np.random.normal() % (2 * np.pi)
        rz_q0 = ('rz', [phi], [q0])
        gate.append(rz_q0)
        pnum += 1
        phi = np.random.normal() % (2 * np.pi)
        ry_q1 = ('ry', [phi], [q1])
        gate.append(ry_q1)
        pnum += 1
        # cnot separator
        cnot_q0_q1 = ('cnot', None, [q0, q1])
        gate.append(cnot_q0_q1)
        # ADDED Rotation Layer
        gate_1q_q0, pnum_1q = self._build_arb1q(q0)
        gate_1q_q1, _ = self._build_arb1q(q1)
        gate.extend(gate_1q_q0)
        pnum += pnum_1q
        gate.extend(gate_1q_q1)
        pnum += pnum_1q
        # ADDED CNOT to sum to identity
        cnot_q0_q1 = ('cnot', None, [q0, q1])
        gate.append(cnot_q0_q1)
        # rotation layer
        phi = np.random.normal() % (2 * np.pi)
        ry_q1 = ('ry', [phi], [q1])
        gate.append(ry_q1)
        pnum += 1
        # cnot layer
        cnot_q1_q0 = ('cnot', None, [q1, q0])
        gate.append(cnot_q1_q0)
        # final 1q layer
        gate_1q_q0, pnum_1q = self._build_arb1q(q0)
        gate_1q_q1, _ = self._build_arb1q(q1)
        gate.extend(gate_1q_q0)
        pnum += pnum_1q
        gate.extend(gate_1q_q1)
        pnum += pnum_1q

        return gate, pnum


    def _build_evff_hef2d(self, qubits, r=1):
        """
        Builds augmented EVFF hardware efficient 2-design
        ansatz list acting on [qubits].

        Returns
        -----------------------------------
        * gate [list] -- contains arbitrary 1q
        gate as list of tuples (rotation, qubit).
        * pnum [int] -- # parameters to speicfy
        [gate]
        """
        # build up gate
        gate = []
        pnum = 0
        k = len(qubits)
        if k == 2:
            i2 = 0
            for _ in range(r):
                temp_gate, pnum_2q = self._build_evff_arb2q(qubits[0], qubits[1])
                i1 = i2
                i2 = i1 + 15
                gate.extend(temp_gate)
                pnum += pnum_2q
        else:
            for j in range(r):
                if k % 2 == 0:
                    qf = k
                else:
                    qf = k - 2
                # add "even" to "odd" layer
                for q_idx in range(0, qf, 2):
                    q0 = qubits[q_idx]
                    q1 = qubits[q_idx + 1]
                    temp_gate, pnum_2q = self._build_evff_arb2q(q0, q1)
                    gate.extend(temp_gate)
                    pnum += pnum_2q
                # add "odd" to "even" layer
                qf = k
                for q_idx in range(1, qf, 2):
                    q0 = qubits[q_idx]
                    q1 = qubits[(q_idx + 1) % k]
                    temp_gate, pnum_2q = self._build_evff_arb2q(q0, q1)
                    gate.extend(temp_gate)
                    pnum += pnum_2q

        return gate, pnum


    def insert_evff_hef2d(self, qubits, loc='end'):
        """
        Inserts hardware efficient 2 design
        ansatz at location [loc] acting
        on [qubits].
        """
        if max(qubits) > self.n - 1:
            e = f"Cannot use qubit {max(qubits)} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()
        evff_hef2d, evff_hef_pnum = self._build_evff_hef2d(qubits)
        self.cwa[loc:loc] = evff_hef2d
        self.pnum += evff_hef_pnum

        return evff_hef2d, evff_hef_pnum


    def _build_1local_z(self, qubits):
        """
        Builds 1 local Z gate on [qubits].

        Returns
        -----------------------------------
        * gate [list] -- contains arbitrary 1q
        gate as list of tuples (rotation, qubit).
        * pnum [int] -- # parameters to speicfy
        [gate]
        """
        gate = []
        pnum = 0
        for q in qubits:
            phi = np.random.normal() % (2 * np.pi)
            z_op = ('rz', [phi], [q])
            gate.append(z_op)
            pnum += 1

        return gate, pnum


    def insert_1local_z(self, qubits, loc='end'):
        """
        Inserts 1 local Z operation on [qubits].
        """
        if max(qubits) > self.n - 1:
            e = "Cannot use {max(qubits)} in {self.n} qubit ansatz"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()
        local_z, z_pnum = self._build_1local_z(qubits)
        self.cwa[loc:loc] = local_z
        self.pnum += z_pnum

        return local_z, z_pnum

    def insert_msl_entangling_ansatz(self, qalist, qblist, loc='end'):
        """
        Entangles qubits from [qalist] with those in [qblist]
        in such a way that resulting mixed state over [qalist]
        is rank(\sigma) = 2^nb for nb = len(qblist).
        """
        qubits = qalist + qblist
        if max(qubits) > self.n - 1:
            e = f"Cannot use qubit {max([q0, q1])} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()

        entangling_ansatz = []
        num_params = 0
        # insert layer of arbitrary 1 qubit gates
        for qa in qalist:
            sing_op, added_params = self.insert_arb1q(qa)
            entangling_ansatz.append(sing_op)
            num_params += added_params
        # insert CNOT cascade
        for i, qb in enumerate(qblist):
            qa = qalist[i]
            cnot = self.insert_cnot(qa, qb)
            entangling_ansatz.append(cnot)

        return entangling_ansatz, num_params


    def _build_givens(self, q0, q1):
        """
        Builds a standard Givens rotation.
        """
        gate = [('G', [0], [q0, q1])]
        pnum = 1

        return gate, pnum
    
    
    def _build_givens_gate(self, q0, q1):
        """
        Builds G-G2-G gives gate with [angles]
        acting on [q0] and [q1].
        """
        gate = []
        gate.append(('G', [0], [q0, q1]))
        gate.append(('G2', [0], [q0, q1]))
        gate.append(('G', [0], [q0, q1]))
        pnum = 3
        
        return gate, pnum

    
    def _build_givens_layer(self, qubits):
        """
        Builds an odd-even tiling of Givens
        gates. 
        """
        # build up gate
        gate = []
        pnum = 0
        k = len(qubits)
        if k == 2:
            temp_gate, pnum_2q = self._build_givens_gate(qubits[0], qubits[1])
            gate.extend(temp_gate)
            pnum += pnum_2q
        else:
            if k % 2 == 0:
                qf = k
            else:
                qf = k - 2
            # add "even" to "odd" layer
            for q_idx in range(0, qf, 2):
                q0 = qubits[q_idx]
                q1 = qubits[q_idx + 1]
                temp_gate, pnum_2q = self._build_givens_gate(q0, q1)
                gate.extend(temp_gate)
                pnum += pnum_2q
            # add "odd" to "even" layer
            qf = k
            for q_idx in range(1, qf - 1, 2):
                q0 = qubits[q_idx]
                q1 = qubits[(q_idx + 1) % k]
                temp_gate, pnum_2q = self._build_givens_gate(q0, q1)
                gate.extend(temp_gate)
                pnum += pnum_2q

        return gate, pnum


    def _build_basic_givens_layer(self, qubits):
        """
        Builds an odd-even tiling of Givens
        rotations.
        """
        # build up gate
        gate = []
        pnum = 0
        k = len(qubits)
        if k == 2:
            temp_gate, pnum_2q = self._build_givens(qubits[0], qubits[1])
            gate.extend(temp_gate)
            pnum += pnum_2q
        else:
            if k % 2 == 0:
                qf = k
            else:
                qf = k - 2
            # add "even" to "odd" layer
            for q_idx in range(0, qf, 2):
                q0 = qubits[q_idx]
                q1 = qubits[q_idx + 1]
                temp_gate, pnum_2q = self._build_givens(q0, q1)
                gate.extend(temp_gate)
                pnum += pnum_2q
            # add "odd" to "even" layer
            qf = k
            for q_idx in range(1, qf - 1, 2):
                q0 = qubits[q_idx]
                q1 = qubits[(q_idx + 1) % k]
                temp_gate, pnum_2q = self._build_givens(q0, q1)
                gate.extend(temp_gate)
                pnum += pnum_2q

        return gate, pnum
    
    
    def insert_givens_layers(self, qubits, loc='end'):
        """
        The US we haven't explored.
        """
        if max(qubits) > self.n - 1:
            e = f"Cannot use qubit {max(qubits)} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()
        givens_layers, givens_pnum = self._build_givens_layer(qubits)
        self.cwa[loc:loc] = givens_layers
        self.pnum += givens_pnum

        return givens_layers, givens_pnum


    def insert_basic_givens_layers(self, qubits, loc='end'):
        """
        The US we haven't explored.
        """
        if max(qubits) > self.n - 1:
            e = f"Cannot use qubit {max(qubits)} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()
        givens_layers, givens_pnum = self._build_basic_givens_layer(qubits)
        self.cwa[loc:loc] = givens_layers
        self.pnum += givens_pnum

        return givens_layers, givens_pnum

    def _build_idres_arb1q(self, q):
        """
        Builds arbitrary 1 qubit gate acting
        like identity on [q].

        Returns
        -----------------------------------
        * gate [list] -- contains arbitrary 1q
        gate as list of tuples (rotation, qubit).
        * pnum [int] -- # parameters to speicfy
        [gate]
        """
        gate = []
        pnum = 0
        phi = 0
        gate.append(('rz', [phi], [q]))
        gate.append(('ry', [phi], [q]))
        gate.append(('rz', [phi], [q]))
        pnum = 3

        return gate, pnum


    def insert_idres0(self, q0, loc="end"):
        """
        Add [idres0] to circuit onto qubits [q0]
        and [q1] at location [loc].
        """
        if max([q0]) > self.n - 1:
            e = f"Cannot use qubit {max([q0, q1])} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()
        idres, id_pnum = self._build_idres_arb1q(q0)
        self.cwa[loc:loc] = idres
        self.pnum += id_pnum

        return idres, id_pnum


    def _build_idres1(self, q0, q1):
        """
        Builds resolution to identity.
        """
        gate = []
        pnum = 0
        # instructions used here
        # init params to 0 for identity resolution
        # add initial CNOT
        cnot = ('cnot', None, [q0, q1])
        gate.append(cnot)
        # add 1q layer
        rz_q0 = ('rz', [0], [q0])
        gate.append(rz_q0)
        pnum += 1
        rx_q0 = ('rx', [0], [q0])
        gate.append(rx_q0)
        pnum += 1
        rz_q0 = ('rz', [0], [q0])
        gate.append(rz_q0)
        pnum += 1
        rx_q1 = ('rx', [0], [q1])
        gate.append(rx_q1)
        pnum += 1
        rz_q1 = ('rz', [0], [q1])
        gate.append(rz_q1)
        pnum += 1
        rx_q1 = ('rx', [0], [q1])
        gate.append(rx_q1)
        pnum += 1
        # add final CNOT
        cnot = ('cnot', None, [q0, q1])
        gate.append(cnot)

        return gate, pnum


    def insert_idres1(self, q0, q1, loc='end'):
        """
        Add [idres1] to circuit onto qubits [q0]
        and [q1] at location [loc].
        """
        if max([q0, q1]) > self.n - 1:
            e = f"Cannot use qubit {max([q0, q1])} in {self.n} qubit ansatz!"
            raise ValueError(e)
        if loc == 'end':
            loc = self.get_num_instructions()
        idres1, id1_pnum = self._build_idres1(q0, q1)
        self.cwa[loc:loc] = idres1
        self.pnum += id1_pnum

        return idres1, id1_pnum


    def insert_random_idres(self, qsup="all", loc='random'):
        """
        Picks random identity resolution acting on
        qubits in [qsup] (default to all) and places
        it at [loc] (defaults to random).
        """
        # first pick random idres
        id_idx = np.random.choice([0, 1])
        # next pick random qubits
        if qsup == "all":
            qubits = list(range(self.n))
            q0 = np.random.choice(qubits)
            qubits.remove(q0)
            q1 = np.random.choice(qubits)
        else:
            q0 = np.random.choice(qsup[0])
            if q0 in qsup[1]:
                qsup[1].remove(q0)
            if qsup[1] != []:
                q1 = np.random.choice(qsup[1])
            else:
                q1 = None
        # pick random location if applicable
        if loc == 'random':
            n_inst = self.get_num_instructions()
            options = list(range(n_inst))
            loc = np.random.choice(options)

        # finally, actually add idres
        if q1 is None:
            self.insert_idres0(q0, loc)
        else:
            if id_idx == 0:
                self.insert_idres0(q0, loc)
                self.insert_idres0(q1, loc)
            if id_idx == 1:
                self.insert_idres1(q0, q1, loc)

        return id_idx, q0, q1, loc


    def insert_random_idres_after_loc(self, qsup="all", after_loc=0):
        """
        Picks random identity resolution acting on
        qubits in [qsup] (default to all) and places
        it at [loc] (defaults to random).
        """
        # first pick random idres, favoring more complicated one
        id_idx = np.random.choice([0, 1], p=[0.2, 0.8])
        # next pick random qubits
        if qsup == "all":
            qubits = list(range(self.n))
            q0 = np.random.choice(qubits)
            qubits.remove(q0)
            q1 = np.random.choice(qubits)
        else:
            q0 = np.random.choice(qsup[0])
            if q0 in qsup[1]:
                qsup[1].remove(q0)
            if qsup[1] != []:
                q1 = np.random.choice(qsup[1])
            else:
                q1 = None
        # pick random location AFTER after_loc
        n_inst = self.get_num_instructions()
        options = list(range(after_loc, n_inst))
        loc = np.random.choice(options)

        # finally, actually add idres
        if q1 is None:
            self.insert_idres0(q0, loc)
        else:
            if id_idx == 0:
                self.insert_idres0(q0, loc)
                self.insert_idres0(q1, loc)
            if id_idx == 1:
                self.insert_idres1(q0, q1, loc)

        return id_idx, q0, q1, loc

    def insert_random_idres_before_loc(self, qsup="all", before_loc=0):
        """
        Picks random identity resolution acting on
        qubits in [qsup] (default to all) and places
        it at [loc] (defaults to random).
        """
        # first pick random idres
        id_idx = np.random.choice([0, 1])
        # next pick random qubits
        if qsup == "all":
            qubits = list(range(self.n))
            q0 = np.random.choice(qubits)
            qubits.remove(q0)
            q1 = np.random.choice(qubits)
        else:
            q0 = np.random.choice(qsup)
            qsup.remove(q0)
            q1 = np.random.choice(qsup)
        # pick random location BEFORE before_loc
        options = list(range(0, before_loc))
        loc = np.random.choice(options)

        # finally, actually add idres
        if q1 is None:
            self.insert_idres0(q0, loc)
        else:
            if id_idx == 0:
                self.insert_idres0(q0, loc)
                self.insert_idres0(q1, loc)
            if id_idx == 1:
                self.insert_idres1(q0, q1, loc)

        return id_idx, q0, q1, loc

    def insert_random_idres_in_pca_ansatz(self, na, nb):
        """
        Adds random (identity) perturbation in PCA ansatz.
        In particular, adds either to [a_qubits] or [b_qubits]
        provided na > 2 or nb > 2. Ensures addition is
        BEFORE the CNOT cascade.
        """
        # compute # of CNOTs and qubit partitions
        num_cnots = nb
        a_qubits = list(range(na))
        b_qubits = list(range(na, na + nb))
        # decide which partition of qubits to add idres to
        if na < 3 and nb < 3:
            q_sup = None
        elif na < 3:
            q_sup = b_qubits
        elif nb < 3:
            q_sup = a_qubits
        else:
            q_sup = np.random.choice([0, 1])
            if q_sup == 0:
                q_sup = a_qubits
            else:
                q_sup = b_qubits
        # invoke above function
        if q_sup is None:
            result = None, None, None, None
        else:
            print(q_sup)
            result = self.insert_random_idres_before_loc(q_sup, num_cnots)

        return result


def givens_rotation(theta):
   """
   Forms Givens rotation matrix
   with angle [theta].
   """
   c = np.cos(theta / 2)
   s = np.sin(theta / 2)
   g = np.array([[1, 0, 0, 0],
                   [0, c, -s, 0],
                   [0, s, c, 0],
                   [0, 0, 0, 1]])

   return g


def givens_rotation2(theta):
   """
   Forms Givens rotation matrix
   with angle [theta].
   """
   c = np.cos(theta / 2)
   s = np.sin(theta / 2)
   g2 = np.array([[0, 1, 0, 0],
                   [c, 0, 0, s],
                   [-s, 0, 0, c],
                   [0, 0, 1, 0]])

   return g2
