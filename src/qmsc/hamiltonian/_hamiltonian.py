import numpy as np
import scipy
import qiskit

def ham_to_uni(h, t):
    """
    Creates unitary corresponding to [h]
    acting on system for [t] ns.
    """
    u = scipy.linalg.expm(-1j * t * h)
    return u

def mat_to_gate(mat, qubits, label=None):
    """
    Converts numpy matrix to qiskit gate.
    """
    gate = qiskit.extensions.UnitaryGate(mat, label)
    return gate

def make_xy_hamiltonian(qpairs=[(0,1)], Jx=1, Jy=1, n=None):
    """
    Creates H = \sum_{<i,j>} (Jx.X_iX_j + Jy.Y_iY_j)
    as a matrix.
    """
    # set default number of qubits if not specified
    if n is None:
        n = get_num_qubits_from_pairs(qpairs)
    # cast Jx/Jy as lists if number for convenience
    if isinstance(Jx, (int, float)):
        Jx = [Jx]
    if isinstance(Jy, (int, float)):
        Jy = [Jy]
    # check that n, Jx, Jy agree on dimension
    num_p = len(qpairs)
    if len(Jx) != (num_p):
        e = f"Jx should have length {num_p}."
        raise ValueError(e)
    if len(Jx) != len(Jy):
        e = "Jx and Jy should have same length."
        raise ValueError(e)
    # create Hamiltonian
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    q1, q2 = qpairs[0]
    xop = Jx[0]*make_two_local_op(x, q1, x, q2, n)
    yop = Jy[0]*make_two_local_op(y, q1, y, q2, n)
    h = xop + yop
    for i, pair in enumerate(qpairs[1:]):
        q1, q2 = pair
        xop = Jx[i+1]*make_two_local_op(x, q1, x, q2, n)
        yop = Jy[i+1]*make_two_local_op(y, q1, y, q2, n)
        h += xop + yop

    return h

def make_xy_chain_hamiltonian(n, rand=0):
    """
    Makes [n] qubit XY model.
    """
    qpairs = [(q, q + 1) for q in range(n - 1)]
    if rand == 0:
        Jx = [1 for _ in range(n - 1)]
        Jy = Jx
    else:
        Jx = np.random.normal(size = n - 1)
        Jy = np.random.normal(size = n - 1)
    return make_xy_hamiltonian(qpairs, Jx, Jy, n)

def zzmat(J, t):
    """
    Create the unitary matrix associated with a ZZ interaction
    of coupling strength J acting for time t.
    """
    n_exp = np.exp(-1j * t * J)
    p_exp = np.exp(1j * t * J)
    mat = np.array([[n_exp, 0, 0, 0], [0, p_exp, 0, 0],
                    [0, 0, p_exp, 0], [0, 0, 0, n_exp]])
    return mat


def make_two_local_op(op1, i, op2, j, n):
    """
    Computes Id2(x)...(x)op1(x)Id2(x)...
    op2(x)...(x)Id2 where op1 at loc i
    and op2 and loc j for n tot qubits.
    """
    if j <= i:
        e = f"j should be greater than i"
        return ValueError(e)
    id_pad1 = np.identity(2**i)
    mat = np.kron(id_pad1, op1)
    id_pad2 = np.identity(2**(j-i-1))
    mat = np.kron(mat, id_pad2)
    mat = np.kron(mat, op2)
    id_pad3 = np.identity(2**(n-j-1))
    mat = np.kron(mat, id_pad3)
    return mat

def get_num_qubits_from_pairs(pair_list):
    """
    Extracts number of distinct elements
    used in list of pairs.
    """
    q_seen = []
    for p in pair_list:
        if p[0] not in q_seen:
            q_seen.append(p[0])
        if p[1] not in q_seen:
            q_seen.append(p[1])
    return len(q_seen)

def gen_rand_xy_ham(n = 3):
    """
    Generates an ensemble of XY
    models with [n] qubits.
    """
    qubits = list(range(n))
    qpairs = [(q, q+1) for q in qubits[0:n-1]]
    Jx = np.random.normal(size=len(qpairs))
    Jy = np.random.normal(size=len(qpairs))

    return make_xy_hamiltonian(qpairs, Jx, Jy, n)

def get_sorted_eig_decomp(a):
    """
    Gives eigendecomposition of a
    with sorted e-vals/ e-vecs.
    """
    e_vals, e_vecs = np.linalg.eig(a)
    sort_idxs = np.argsort(e_vals)
    e_vals = e_vals[sort_idxs]
    e_vecs = e_vecs[:, sort_idxs]

    return e_vals, e_vecs


def get_one_exc_subspace(ham):
    """
    Given a Hamiltonian [ham], returns
    the subspace consisting of the
    ground + 1st excited space.
    """
    e_vals, e_vecs = get_sorted_eig_decomp(ham)
    # extract indices of two smallest e-vals
    gs_space = np.where(np.isclose(e_vals, e_vals[0]))[0]
    exc1_idx = gs_space[-1] + 1
    first_space = np.where(np.isclose(e_vals, e_vals[exc1_idx]))[0]
    low_space = np.concatenate((gs_space, first_space))

    # get low-lying eigenvectors
    low_vecs = []
    for i in low_space:
        low_vecs.append(e_vecs[:, i])

    return low_vecs

def get_ground_subspace(ham):
    """
    Given a Hamiltonian [ham], returns
    the subspace consisting of the
    ground-states.
    """
    e_vals, e_vecs = get_sorted_eig_decomp(ham)
    # extract indices of lowest evals
    gs_space = np.where(np.isclose(e_vals, e_vals[0]))[0]
    # extract gs vectors
    gs_vecs = []
    for i in gs_space:
        gs_vecs.append(e_vecs[:, i])

    return gs_vecs

def get_first_exc_subspace(ham):
    """
    Given a Hamiltonian [ham], returns
    the subspace consisting of the
    1st excited states.
    """
    e_vals, e_vecs = get_sorted_eig_decomp(ham)
    # extract indices of two smallest e-vals
    gs_space = np.where(np.isclose(e_vals, e_vals[0]))[0]
    exc1_idx = gs_space[-1] + 1
    first_space = np.where(np.isclose(e_vals, e_vals[exc1_idx]))[0]

    # get low-lying eigenvectors
    first_vecs = []
    for i in first_space:
        first_vecs.append(e_vecs[:, i])

    return first_vecs

def gen_one_exc_xy_ensemble(s, n = 3):
    """
    Finds one excitation of subspace of [s] XY
    spin-chains with [n] qubits and returns
    associated ensemble of states.
    """
    ensemble = []
    for _ in range(s):
        ham = gen_rand_xy_ham(n)
        low_vecs = get_one_exc_subspace(ham)
        ensemble.extend(low_vecs)

    return ensemble

def bin_vectors(vectors):
    """
    Finds counts of different vectors
    to form empirical pmf.
    """
    tag_to_vector = {'s1': vectors[0]}
    tag_to_count = {'s1': 1}

    for state in vectors[1:]:
        broken = False
        for key, value in tag_to_vector.items():
            if np.isclose(state, value).all():
                tag_to_count[key] += 1
                broken = True
                break
            else:
                continue
        if broken == False:
            new_key = f"s{int(key[1:])+1}"
            tag_to_vector[new_key] = state
            tag_to_count[new_key] = 1

    # normalize counts
    norm = sum(tag_to_count.values())
    for key in tag_to_count:
        tag_to_count[key] /= norm

    return tag_to_vector, tag_to_count
