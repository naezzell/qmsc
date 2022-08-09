import itertools
import numpy as np
import qutip as qt
import qmsc.circuit as qmscc
from qiskit import QuantumRegister, QuantumCircuit, BasicAer, execute

def make_density_operator(state_list, prob_list):
    """
    Create a density operator given an ensemble
    of states in [state_list] with probabilities
    [prob_list].
    """
    if len(state_list) != len(prob_list):
        e = f"Ensemble has {len(statelen)} state but {len(prob_list)} probs."
        raise ValueError(e)

    dim = len(state_list[0])
    tot_rho = np.zeros((dim, dim)) + 0.0j
    tot_prob = 0.0
    for j in range(len(state_list)):
        s = state_list[j]
        p = prob_list[j]
        rho = np.kron(s.conj(), s).reshape(dim, dim)
        tot_rho += (p * rho)
        tot_prob += p
        if tot_prob > 1:
            w = f"Warning: \sum p_i = {tot_prob}"
            warnings.warn(w)

    return tot_rho

def make_zero_projector(index_list, n):
    """
    Create a projector where qubits in
    [index_list] (of len k) are in 0
    state and other n - k qubits free.

    For index_list = 1 and n = 3,
    gives P = I (x) |0><0| (x) I,
    where I is 2 x 2 identity.
    """
    # assumes ordered list, so just sort it just in case
    index_list.sort()
    if index_list[-1] > (n-1):
        e = "Largest idx of index_list is {index_list[-1]} but only {n} qubits"
        raise ValueError(e)
    prev_idx = -1
    # |0><0| projector
    zero_proj = np.array([[1, 0], [0, 0]])
    # set running projector to identity
    proj = 1
    for i in index_list:
        idpad = 1
        # make identity padding if needed
        pad = i - prev_idx
        if pad != 1:
            dim = int(2**(pad - 1))
            idpad = np.identity(dim)
        proj = np.kron(proj, idpad)
        # add |0><0| projector
        proj = np.kron(proj, zero_proj)
        prev_idx = i

    # add final identity padding if necessary
    pad = n - index_list[-1]
    if pad > 1:
        dim = int(2**(pad - 1))
        idpad = np.identity(dim)
        proj = np.kron(proj, idpad)

    return proj

def par_trace(rho, dim1, dim2, sys):
    """
    Traces out sys
    """
    res_rho = rho.reshape(dim1, dim2, dim1, dim2)
    if sys in [1, 'A', 'a']:
        sub_rho = np.trace(res_rho, axis1=0, axis2=2)
    elif sys in [2, 'B', 'b']:
        sub_rho = np.trace(res_rho, axis1=1, axis2=3)

    return sub_rho


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

def np_state_to_qt_state(np_state):
    """
    Converts a numpy array representation of state to
    qutip state which is easier to manipulate a la partial
    trace and whatnot.
    """
    dict_state = get_dict_rep_of_vec(np_state)
    qt_state = None
    for key, value in dict_state.items():
        if qt_state is None:
            qt_state = value * qt.states.ket(key)
        else:
            qt_state += value * qt.states.ket(key)

    return qt_state

def check_contiguous(array):
    # sort array
    array.sort()

    # check if array[i] - array[i-1] != 1
    for i in range(1, len(array)):
        if array[i] - array[i - 1] != 1:
            return False

    return True

def make_projector(state, s_idxs, n):
    """
    Make Projector onto |state><state|_{s_idxs}
    2^[s_idxs] dim subspace of [n] qubit Hilbert
    space.
    """
    if not check_contiguous(s_idxs):
        e = f"State idx list, s_idxs={s_idxs} which is not contiguous."
        raise ValueError(e)

    # make projector in s_idxs space
    if type(state) == np.ndarray:
        qt_s = np_state_to_qt_state(state)
    else:
        qt_s = state
    s_op = qt.ket2dm(qt_s)
    # make full projector
    op_list = []
    added = False
    for j in range(n):
        if j in s_idxs:
            if added is False:
                op_list.append(s_op)
                added = True
        else:
            op_list.append(qt.identity(2))

    return qt.tensor(*op_list)

def make_op_projector(s_op, s_idxs, n):
    """
    Make Projector onto |state><state|_{s_idxs}
    2^[s_idxs] dim subspace of [n] qubit Hilbert
    space.
    """
    if not check_contiguous(s_idxs):
        e = f"State idx list, s_idxs={s_idxs} which is not contiguous."
        raise ValueError(e)

    # make full projector
    op_list = []
    added = False
    for j in range(n):
        if j in s_idxs:
            if added is False:
                op_list.append(s_op)
                added = True
        else:
            op_list.append(qt.identity(2))

    return qt.tensor(*op_list)

def gen_haar_random_unitary(n):
    """
    Generates Haar random unitary on [n]
    qubits.
    """
    u = qmscc.utils.gen_haar_random_gate(n).to_matrix()
    return u

def gen_cliff_random_unitary(n):
    """
    Generates Clifford random unitary on
    [n] qubits.
    """
    u = qmscc.utils.gen_cliff_random_gate(n).to_matrix()
    return u

def generate_random_prob_vec(N):
    """
    Generate random probability vector
    over N outcomes.
    Samples uniformly over probability simplex.
    """
    # first, generate N uniform random numbers
    x = np.random.random(int(N))
    # generate exp distribution from unif
    exp_dist = -np.log(x)
    # get random uniform prob vec
    prob_vec = exp_dist / sum(exp_dist)

    return sorted(prob_vec, reverse=True)

def gen_haar_rand_density_op(n, rank):
    """
    Generate a random density matrix over
    [n] qubits with given [rank].
    """
    # step 0: check n, rank match
    if rank > 2**n:
        e = f"Rank cannot be greater than 2**{n}."
        raise ValueError(e)

    # step 1: generate eigenvals obeying rank and probability
    vals = generate_random_prob_vec(rank)
    # pad zeros to get full dimensions
    evals = list(vals) + [0 for _ in range(int(2**n - rank))]
    # generate random unitary and create density op as U D U^{\dag}
    u = gen_haar_random_unitary(n)
    d = np.diag(evals)
    udg = u.conj().transpose()
    rho = np.matmul(np.matmul(u, d), udg)

    return rho

def gen_cliff_rand_density_op(n, rank):
    """
    Generate a random density matrix over
    [n] qubits with given [rank].
    """
    # step 0: check n, rank match
    if rank > 2**n:
        e = f"Rank cannot be greater than 2**{n}."
        raise ValueError(e)

    # step 1: generate eigenvals obeying rank and probability
    vals = generate_random_prob_vec(rank)
    # pad zeros to get full dimensions
    evals = list(vals) + [0 for _ in range(int(2**n - rank))]
    # generate cliff random unitary and create density op as U D U^{\dag}
    u = gen_cliff_random_unitary(n)
    d = np.diag(evals)
    udg = u.conj().transpose()
    rho = np.matmul(np.matmul(u, d), udg)

    return rho

def gen_purification_from_rho(rho):
    """
    Given [rho] on n qubits, finds appropriate purification
    on 2 * n qubits, i.e. Sqrt{rho}.
    """
    n = int(np.log2(rho.shape[0]))
    # get qutip version of rho to take sqrt
    dim = [2 for _ in range(n)]
    qt_rho = qt.Qobj(rho, dims = [dim, dim])
    sqrt_rho = qt_rho.sqrtm()
    # get state purification in right shape
    state = sqrt_rho.full().reshape(2**(2 * n))

    return state

def convert_to_qutip_vec(np_array):
    """
    Converts a numpy array to a Qutip object.
    """
    shape = np_array.shape[0]
    ket_dim = [2 for _ in range(int(np.log2(shape)))]
    bra_dim = [1 for _ in range(int(np.log2(shape)))]
    dims = [ket_dim, bra_dim]

    return qt.Qobj(np_array, dims)

def generate_kunal_state(n, na):
    """
    Generates states used in VQSE paper.
    """
    q1 = QuantumRegister(n, 'q1')
    anc = QuantumRegister(na, 'ancilla')
    qc = QuantumCircuit(q1, anc)

    qc.h(anc[0])
    qc.h(anc[1])

    #new--
    qc.ry(0.8,anc[2])
    qc.ry(0.35,anc[3])
    #---

    qc.ry(0.3,q1[0])
    qc.ry(0.4,q1[1])
    qc.ry(0.5,q1[2])
    qc.h(q1[3])
    qc.ry(0.1,anc[0])
    qc.ry(0.6,anc[1])
    qc.cx(q1[0],q1[1])
    qc.cx(q1[2],q1[3])
    qc.cx(anc[0],q1[0])
    qc.cx(q1[3],anc[1])
    qc.h(q1[1])
    qc.h(q1[2])

    #new---
    qc.cx(q1[1],anc[2])
    qc.cx(anc[3],q1[2])
    qc.ry(.43,q1[1])
    qc.cx(anc[3],anc[2])
    qc.ry(.7,q1[2])

    backend = BasicAer.get_backend('statevector_simulator')

    # Create a Quantum Program for execution
    job = execute(qc, backend)

    result = job.result()

    outputstate = result.get_statevector(qc, decimals=11)


    psi = convert_to_qutip_vec(outputstate)
    rho = psi.ptrace(list(range(n)))

    return rho
