import itertools
import numpy as np
import qiskit

def gen_haar_random_gate(n):
    """
    Generates a Haar random unitary gate
    on n qubits.
    """
    u = qiskit.quantum_info.random_unitary(2**n)
    gate = qiskit.extensions.UnitaryGate(u, label='haar-U')
    return gate

def gen_cliff_random_gate(n):
    """
    Generates a random Clifford gate.
    """
    cliff_mat = qiskit.quantum_info.random_clifford(n).to_matrix()
    gate = qiskit.extensions.UnitaryGate(cliff_mat, label='rand-Cliff')
    return gate

def gen_k_rand_basis_state(k, n, replace=False):
    """
    Given an [n] qubit system, generate
    k random basis elements with/without
    replacement.
    """
    state_list = []
    dim = 2**n
    idx_list = np.random.choice(range(dim), k, replace)
    for i in idx_list:
        state = np.zeros(dim)
        state[i] = 1
        state_list.append(state)

    return state_list


def get_harr_random_u3_params():
    """
    Generates the theta, phi, and lambda parameters for a harr-random
    unitary in u3(theta, phi, lambda) form
    """
    decomp = qiskit.quantum_info.synthesis.OneQubitEulerDecomposer(basis='U3')
    haar_random = qiskit.quantum_info.random_unitary(2).data

    return decomp.angles(haar_random)

def get_dict_rep_of_vec(vec):
    """
    Returns a dict mapping of
    {(i, j, k,...): amp[i,j,k...]}
    which maps basis element to amplitude
    of statevector.
    """
    dict_vec = {}
    n = int(np.log2(len(vec)))
    for i, b_tup in enumerate(itertools.product([0,1], repeat=n)):
        if not np.isclose(vec[i], 0.):
            dict_vec[b_tup] = vec[i]
    return dict_vec

def guess(n):
    return 2*np.pi*np.random.rand(n)

def get_pvec_size_for_layer(lname, qlist1, r=1, qlist2=None, l=1):
    """
    Given a [layer], returns how large the
    parameter vector should be to add
    a layer of given type to circuit.
    """
    # determine if sing qlist layer or not
    if qlist2 is None:
        n = len(qlist1)
    else:
        n1 = len(qlist1)
        n2 = len(qlist2)
        n = n1 + n2

    # find necessary dimensions
    if lname == '1q':
        dim = 3 * n
    elif lname == '2q':
        dim = int((15 * n * (n-1)) / 2)
    elif lname == 'l_loc_z':
        lq_list = list(range(n))
        combos = list(itertools.combinations(lq_list, l))
        dim = len(combos)
    elif lname == 'ry':
        dim = n
    elif lname == 'bravo':
        dim = 2 * n + n1
    elif lname == '2q_eff':
        if n == 2:
            dim = 15 * r
        elif (n % 2 == 0) and (n > 2):
            dim = (15 * n) * r
        else:
            dim = (15 * (n - 1)) * r
    elif lname == 'qaqc':
        if n == 2:
            dim = (15 + (3 * n)) * r + (3 * n)
        elif (n > 2) and (n % 2 == 0):
            dim = (15 * n + 3 * n) * r + (3 * n)
        else:
            dim = (15 * (n - 1) + 3 * n) * r + (3 * n)
    elif lname == 'vff':
        lq_list = list(range(n))
        combos = list(itertools.combinations(lq_list, l))
        z_dim = len(combos)
        if n == 2:
            dim = (15 + (3 * n)) * r + (3 * n) + z_dim
        elif (n > 2) and (n % 2 == 0):
            dim = (15 * n + 3 * n) * r + (3 * n) + z_dim
        else:
            dim = (15 * (n - 1) + 3 * n) * r + (3 * n) + z_dim
    elif lname == 'vff2':
        lq_list = list(range(n))
        combos = list(itertools.combinations(lq_list, l))
        z_dim = len(combos)
        if n == 2:
            dim = 15 * r + z_dim
        elif (n % 2 == 0) and (n > 2):
            dim = (15 * n) * r + z_dim
        else:
            dim = (15 * (n - 1)) * r + z_dim

    return dim

def out_to_in_idxs(n):
    """
    Returns list indices used to traverse
    a list from the 'outside' to 'inside.'
    """
    n -= 1
    idxs = []
    for j in range(int(n / 2) + 1):
        if j == n-j:
            idxs.append(j)
        else:
            idxs.append(j)
            idxs.append(n - j)

    return idxs

def build_pad0_state(s, s_qubits, a_qubits):
    """
    Given [s] over [s_qubits], builds full
    state over [a_qubits] where q \in a but
    not \in s are put in |0> state.
    """
    u = np.array([1, 0])
    d = np.array([0, 1])
    dict_s = utils.get_dict_rep_of_vec(s)
    for key in dict_s.keys():
        basis_rep = key
    sidx = 0
    full_s = 1
    for q in a_qubits:
        if q in s_qubits:
            if basis_rep[sidx] == 0:
                full_s = np.kron(full_s, u)
            else:
                full_s = np.kron(full_s, d)
            sidx += 1
        else:
            full_s = np.kron(full_s, u)

    return full_s
