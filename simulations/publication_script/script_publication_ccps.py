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

def main(state_type, ns, s_lidx, s_uidx, R_choices=0, max_perturbations=2, opt_tol=1e-16,
         diff_tol=1e-16, init_beta=1e-3, data_dir="."):
    """
    CCPS MSL script.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fname_start = f"{data_dir}/CCPS_{state_type}_ns_{ns}_datetime_{timestr}"
    csv_fname = f"{fname_start}.csv"
    state_fname = f"{fname_start}.pkl"
    with open(csv_fname, "w") as f:
        line = "ns,sidx,T,re,R,numerical_DHS,opt_DHS,num_iterations,num_func_evals"
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

        # convert \rho to a qiskit operator
        rho = qiskit.quantum_info.DensityMatrix(rho.data.toarray())
        # use epsilon rank to determine the number of ancilla necessary
        na = int(np.ceil(np.log2(re)))
        if na == 0:
            na += 1
        R = 2**na
        if R_choices == 0:
            if R == 2:
                R_vals = np.array([2])
            else:
                R_vals = np.array([2, R])
        elif R_choices == 1:
            R_vals = np.array([2])
        elif R_choices == 2:
            R_vals = np.array([R])
        else:
            R_vals = np.array([8])
        # compute purity
        rp = np.real(rho.purity())
        # perform optimization over R values of interest
        for R in R_vals:
            print(f"ns={ns}, re={re}, R={R}")
            # compute best possible cost
            opt_dhs = compute_optimal_DHS(rho, R)
            print(f"Theoretically Optimal DHS: {opt_dhs}")
            # make ansatz and optimize
            ansatz = init_ansatz(ns, state_type)
            # perform first optimization
            p = np.random.random(size = R - 1)
            angles = np.random.random(ansatz.get_num_parameters())
            x0 = np.concatenate((p, angles))
            cost_history = []
            cost = lambda angles: compute_msl_cost(angles, R, ansatz, rho, rp, cost_history)
            result = scipy.optimize.minimize(cost, x0)
            # update best parameters after setting angles back to [0, 2\pi] range
            x0 = result.x % (2 * np.pi)
            p_star = result.x[0:R - 1]
            angles_star = result.x[R-1:]
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
                # make perturbation and re-optimize
                old_ansatz = copy.deepcopy(ansatz)
                pert = insert_ansatz_layer(ansatz, state_type)
                cost = lambda x: compute_msl_cost(x, R, ansatz, rho, rp)
                x0 = np.concatenate((p_star, ansatz.get_parameters()))
                pert_history = []
                cost = lambda angles: compute_msl_cost(angles, R, ansatz, rho, rp, pert_history)
                pert_result = scipy.optimize.minimize(cost, x0)
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
                    x0 = pert_result.x % (2 * np.pi)
                    p_star = pert_result.x[0:R - 1]
                    angles_star = pert_result.x[R-1:]
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
                line = f"\n{ns},{sidx},{T},{re},{R},{dhs_star},{opt_dhs},{nit},{nfev},{num_pert}"
                f.write(line)

            # save target state and final optimized state
            data_dict[f"rho_ns_{ns}_T_{T}_re_{re}_R_{R}_sidx_{sidx}"] = rho
            data_dict[f"costs_ns_{ns}_T_{T}_re_{re}_R_{R}_sidx_{sidx}"] = cost_history
            data_dict[f"optDHS_ns_{ns}_T_{T}_re_{re}_R_{R}_sidx_{sidx}"] = opt_dhs
            data_dict[f"nit_ns_{ns}_T_{T}_re_{re}_R_{R}_sidx_{sidx}"] = nit
            ansatz.update_parameters(angles_star)
            psi_list = generate_state_ensemble(R, ansatz)
            for i in range(R):
                if i < R - 1:
                    data_dict[f"p_ns_{ns}_T_{T}_re_{re}_R_{R}_sidx_{sidx}_{i}"] = p_star[i]
                else:
                    data_dict[f"p_ns_{ns}_T_{T}_re_{re}_R_{R}_sidx_{sidx}_{i}"] = 1 - sum(p_star)
                data_dict[f"psi_ns_{ns}_T_{T}_re_{re}_R_{R}_sidx_{sidx}_{i}"] = psi_list[i]
            with open(state_fname, "wb") as f:
                pickle.dump(data_dict, f)
                
                

def init_ansatz(ns, state_type):
    """
    Init ansatz appropriate for state type.
    """
    if "xy" in state_type:
        l = int(np.ceil(np.log2(ns)))
        ansatz = generate_xy_sigma_ansatz(ns, l)
    else:
        l = ns
        ansatz = generate_hef2d_sigma_ansatz(ns, l)
        
    return ansatz

def generate_xy_sigma_ansatz(ns, layers=1):
    """
    Prepares diagonal density matrix.
    """
    ansatz = FlexibleAnsatz(ns)
    qubits = list(range(ns))
    for l in range(layers):
        ansatz.insert_givens_layers(qubits)
        
    return ansatz

def generate_hef2d_sigma_ansatz(ns, layers=1):
    """
    Prepares diagonal density matrix.
    """
    ansatz = FlexibleAnsatz(ns)
    qubits = list(range(ns))
    for l in range(layers):
        if ns == 1:
            ansatz.insert_arb1q_layer(qubits)
        else:
            ansatz.insert_hef2d(qubits)
        
    return ansatz

def generate_qubit_basis_states(r, n):
    """
    Generates [r] orthogonal computational
    basis states over [n] qubits.
    """
    basis_states = []
    for k in range(r):
        state = np.zeros(2**n)
        state[k] = 1
        state = qiskit.quantum_info.Statevector(state)
        basis_states.append(state)
        
    return basis_states

def generate_state_ensemble(num_states, ansatz):
    """
    Generates the |psi_i> that make up
    \sigma = \sum_i p_i | \psi_i > < \psi_i|
    given an ansatz.
    """
    # compute U from ansatz
    circ = ansatz.build_circ(ansatz.n)
    u = circ.get_unitary(reverse=True)
    # compute states
    basis_states = generate_qubit_basis_states(num_states, ansatz.n)
    # transform them under same U
    psi_list = []
    for state in basis_states:
        psi = qiskit.quantum_info.Statevector(np.matmul(u, state))
        psi_list.append(psi)
        
    return psi_list

def compute_DHS(p_vec, psi_list, rho, rp):
    """
    Computes Hilbert-Schmidt distance between
    rho and sigma given
    \sigma = \sum_i p_vec[i] |psi_list[i]><psi_list[i]|
    and rp = Tr[\rho^2]. 
    """
    purity_term = rp + np.sum(np.dot(p_vec, p_vec))
    cross_term = 0.0
    for i, psi in enumerate(psi_list):
        pi = p_vec[i]
        exp_val = np.real(psi.expectation_value(rho))
        cross_term += pi * exp_val
    
    return purity_term - 2 * cross_term

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

def compute_msl_cost(parameters, num_states, ansatz, rho, rp, history=None):
    """
    Computes mixed-state learning cost given
    * parameters = [p_1, .., p_R, \theta_1, \theta_2, ... \theta_n]
    * num_states = determines rank of generates \sigma
    * ansatz used to generate U
    * rho = the target state
    * rp = the target state purity
    """
    # extract probs
    p_vec = parameters[0:num_states - 1]
    p_vec = np.concatenate((p_vec, [1 - sum(p_vec)]))
    # update angles in ansatz with remaining parameters
    thetas = parameters[num_states-1:]
    ansatz.update_parameters(thetas)
    psi_list = generate_state_ensemble(num_states, ansatz)
    # compute DHS
    dhs = compute_DHS(p_vec, psi_list, rho, rp)
    if history is not None:
        history.append(dhs)
    
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
    state_type, ns, s_lidx, s_uidx, R_choices, max_perturbations, opt_tol, diff_tol, init_beta, data_dir, trials = sys.argv[1:]
    ns = int(ns)
    s_lidx = int(s_lidx)
    s_uidx = int(s_uidx)
    R_choices = int(R_choices)
    max_perturbations = int(max_perturbations)
    opt_tol = float(opt_tol)
    diff_tol = float(diff_tol)
    init_beta = float(init_beta)
    trials = int(trials)
    # run main
    for t in range(trials):
        main(state_type, ns, s_lidx, s_uidx, R_choices, max_perturbations, opt_tol,
             diff_tol, init_beta, data_dir)
