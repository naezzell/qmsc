import time
import sys
import copy
import numpy as np
import qmsc.hamiltonian as ham
from qmsc.circuit import VaCirc, AnsatzCirc
from qmsc.backend import IBMQBackend
from qmsc.ansatz import FlexibleAnsatz
import qmsc.qstate as qstate
from qiskit.circuit.random import random_circuit
import qiskit
import qutip
import scipy
import time
import pickle
import matplotlib.pyplot as plt

def main(state_type, ns, s_lidx, s_uidx, na_choices=0, max_perturbations=2, opt_tol=1e-16, diff_tol=1e-16, init_beta=1e-3, data_dir="."):
    """
    VSPA MSL script.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fname_start = f"{data_dir}/VPSA_{state_type}_ns_{ns}_datetime_{timestr}"
    csv_fname = f"{fname_start}.csv"
    state_fname = f"{fname_start}.pkl"
    with open(csv_fname, "w") as f:
        line = "ns,sidx,T,re,na,numerical_DHS,opt_DHS,num_iterations,num_func_evals"
        f.write(line)
        
    data_dict = {}
    for sidx in range(s_lidx, s_uidx + 1):
        if state_type == "bures":
            with open("random_bures_states.pkl", 'rb') as f:
                state_dict = pickle.load(f)
                rho, re = state_dict[(ns, sidx)]
                T = -1
        elif state_type == "hilbertschmidt":
            with open("random_hilbertschmidt_states.pkl", 'rb') as f:
                state_dict = pickle.load(f)
                rho, re = state_dict[(ns, sidx)]
                T = -1
        elif state_type == "xy_T_0.05":
            with open("random_xy_T_0.05_states.pkl", 'rb') as f:
                state_dict = pickle.load(f)
                T = 0.05
                rho, re = state_dict[(ns, T, sidx)]
        elif state_type == "xy_T_0.5":
            with open("random_xy_T_0.5_states.pkl", 'rb') as f:
                state_dict = pickle.load(f)
                T = 0.5
                rho, re = state_dict[(ns, T, sidx)]
        elif state_type == "xy_T_5.0":
            with open("random_xy_T_5.0_states.pkl", 'rb') as f:
                state_dict = pickle.load(f)
                T = 5.0
                rho, re = state_dict[(ns, T, sidx)]
                
        # use epsilon rank to determine the number of ancilla necessary 
        na = int(np.ceil(np.log2(re)))
        if na == 0:
            na = 1
        if na_choices == 0:
            if na == 1:
                na_vals = np.array([1])
            else:
                na_vals = np.array([1, na])
        elif na_choices == 1:
            na_vals = np.array([1])
        else:
            na_vals = np.array([na])
        # compute purity
        rp = np.real(rho.purity())
        # iterate over na vals and perform optimization
        for na in na_vals:
            print(f"ns={ns},T={T}, re={re}, na={na}")
            # compute best possible cost
            R = 2**na
            opt_dhs = compute_optimal_DHS(rho, R)
            print(f"Theoretically Optimal DHS: {opt_dhs}")

            # make ansatz and optimize
            ansatz = init_ansatz(ns, na, state_type)
            x0 = np.random.random(ansatz.get_num_parameters())
            cost_history = []
            cost = lambda angles: compute_msl_cost(angles, ansatz, ns, na, rho, rp, cost_history)
            result = scipy.optimize.minimize(cost, x0, tol=diff_tol)
            # update best parameters after setting angles back to [0, 2\pi] range
            angles_star = result.x % (2 * np.pi)
            ansatz.update_parameters(angles_star)
            dhs_star = float(result.fun)
            nfev = result.nfev
            nit = result.nit
            print(f"Numerically Optimal DHS: {dhs_star}")

            # if solution not satisfactory, try some perturbations
            num_pert = 0
            if num_pert < max_perturbations:
                stop_condition_met = False
            else:
                stop_condition_met = True
            beta = init_beta
            diff = np.abs(opt_dhs - dhs_star)
            print(f"Diff: {diff}.")
            if diff < diff_tol:
                print(f"Close enough with given diff--we stop.")
                stop_condition_met = True

            while stop_condition_met is False:
                old_ansatz = copy.deepcopy(ansatz)
                pert = insert_ansatz_layer(ansatz, state_type)
                x0 = ansatz.get_parameters()
                pert_history = []
                cost = lambda angles: compute_msl_cost(angles, ansatz, ns, na, rho, rp, pert_history)
                pert_result = scipy.optimize.minimize(cost, x0, method=opt_method)
                pert_dhs = float(pert_result.fun)
                print(f"Perturbed DHS: {pert_dhs}")
                if pert_dhs >= dhs_star:
                    weight = np.exp(-beta * (pert_dhs - dhs_star))
                    accept = np.random.choice([True, False], p=[weight, 1 - weight])
                else:
                    accept = True
                if accept is True:
                    print("Perturbation accepted.")
                    cost_history += pert_history
                    dhs_star = pert_dhs
                    angles_star = pert_result.x % (2 * np.pi)
                    ansatz.update_parameters(angles_star)
                    nfev += pert_result.nfev
                    nit += pert_result.nit
                    diff = np.abs(opt_dhs - dhs_star)
                else:
                    print("Perturbation rejected.")
                    ansatz = old_ansatz
                if diff < diff_tol:
                    print(f"Close enough with given diff--we stop.")
                    break
                num_pert += 1
                print(f"Num pert: {num_pert}\n")
                if num_pert >= max_perturbations:
                    break
                beta *= 10

            # save summary data to csv file
            with open(csv_fname, "a+") as f:
                line = f"\n{ns},{sidx},{T},{re},{na},{dhs_star},{opt_dhs},{nit},{nfev},{num_pert}"
                f.write(line)

            # save target state and final optimized state
            data_dict[f"rho_ns_{ns}_T_{T}_re_{re}_na_{na}_sidx_{sidx}"] = rho
            data_dict[f"costs_ns_{ns}_T_{T}_re_{re}_na_{na}_sidx_{sidx}"] = cost_history
            data_dict[f"optDHS_ns_{ns}_T_{T}_re_{re}_na_{na}_sidx_{sidx}"] = opt_dhs
            data_dict[f"nit_ns_{ns}_T_{T}_re_{re}_na_{na}_sidx_{sidx}"] = nit
            ansatz.update_parameters(angles_star)
            sigma = compute_sigma(ansatz, ns, na)
            data_dict[f"sigma_ns_{ns}_T_{T}_re_{re}_na_{na}_sidx_{sidx}"] = sigma
            with open(state_fname, "wb") as f:
                pickle.dump(data_dict, f)


def convert_to_qutip_vec(np_array):
    """
    Converts a numpy array to a Qutip object.
    """
    shape = np_array.shape[0]
    ket_dim = [2 for _ in range(int(np.log2(shape)))]
    bra_dim = [1 for _ in range(int(np.log2(shape)))]
    dims = [ket_dim, bra_dim]
    
    return qutip.Qobj(np_array, dims) 

def init_ansatz(ns, na, state_type):
    """
    Init ansatz appropriate for state type.
    """
    if "xy" in state_type:
        l = int(np.ceil(np.log2(ns)))
        ansatz = generate_xy_sigma_ansatz(ns, na, l)
    else:
        l = ns
        ansatz = generate_hef2d_sigma_ansatz(ns, na, l)
        
    return ansatz

def generate_xy_sigma_ansatz(ns, na, layers=1):
    """
    Prepares diagonal density matrix.
    """
    ansatz = FlexibleAnsatz(ns + na)
    qubits = list(range(ns + na))
    for l in range(layers):
        ansatz.insert_givens_layers(qubits)
        
    return ansatz

def generate_hef2d_sigma_ansatz(ns, na, layers=1):
    """
    Prepares diagonal density matrix.
    """
    ansatz = FlexibleAnsatz(ns + na)
    qubits = list(range(ns + na))
    for l in range(layers):
        ansatz.insert_hef2d(qubits)
        
    return ansatz

def compute_sigma(ansatz, ns, na):
    """
    Computes sigma from given [ansatz]
    where sigma is [ns] qubit density matrix.
    """
    circ = ansatz.build_circ(ns + na)
    psi = np.asarray(circ.get_statevector(True))
    psi = convert_to_qutip_vec(psi)
    sigma = psi.ptrace(range(ns))
    
    return sigma

def compute_optimal_DHS(rho, R):
    """
    Computes D^*_HS when learning [rho]
    with a rank [R] approximation.
    """
    # compute \lambda's and sort them
    lam = np.sort(np.real(np.linalg.eigvals(rho)))
    d = len(lam)
    # compute lowest d - R lowest and R highest
    low_lam = np.array(lam[0:d-R])
    high_lam = np.array(lam[d-R:])
    N = (1 - np.sum(high_lam)) / R
    # compute cost
    opt_DHS = np.sum(low_lam**2) +  R * N**2
        
    return opt_DHS

def compute_DHS(rho, rp, sigma):
    """
    Computes Hilbert-Schmidt distance
    between [rho] and [sigma].
    """
    purity_terms = rho.purity() + sigma.purity()
    cross_term = 2 * np.real((rho * sigma).tr())
    
    return purity_terms - cross_term

def compute_msl_cost(angles, ansatz, ns, na, rho, rp, cost_history=None):
    """
    Constructs sigma and then evaluates DHS.
    """
    # update angles of ansatz
    ansatz.update_parameters(angles)
    # construct sigma
    sigma = compute_sigma(ansatz, ns, na)
    # compute DHS
    dhs = compute_DHS(rho, rp, sigma)
    if cost_history is not None:
        cost_history.append(dhs)
    
    return dhs

def insert_ansatz_layer(ansatz, state_type):
    """
    Adds an addition layer to the ansatz.
    """
    qubits = list(range(ansatz.n))
    if "xy" in state_type:
        ansatz.insert_givens_layers(qubits)
    else:
        ansatz.insert_hef2d(qubits)
        
    return


if __name__ == "__main__":
    # parse inputs
    state_type, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol, diff_tol, init_beta, data_dir, trials = sys.argv[1:]
    ns = int(ns)
    s_lidx = int(s_lidx)
    s_uidx = int(s_uidx)
    na_choices = int(na_choices)
    max_perturbations = int(max_perturbations)
    opt_tol = float(opt_tol)
    diff_tol = float(diff_tol)
    init_beta = float(init_beta)
    trials = int(trials)
    # run main
    for t in range(trials):
        main(state_type, ns, s_lidx, s_uidx, na_choices, max_perturbations, opt_tol,
             diff_tol, init_beta, data_dir)
