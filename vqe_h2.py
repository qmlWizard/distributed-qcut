# h2_vqe_cut.py

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from circuit_cut.distribute import RayAgent
from circuit_cut.cutting import CircuitCutting


# ============================================================
# H2 Hamiltonian
# H2 at approximately 0.74 A, Jordan-Wigner, 4 qubits.
# H = sum_i c_i P_i
# ============================================================

H2 = SparsePauliOp.from_list([
    ("IIII", -0.0420789854),
    ("ZIII",   0.1777128750),
    ("IZII",   0.1777128750),
    ("IIZI",  -0.2427428005),
    ("IIIZ",  -0.2427428005),

    ("ZZII",   0.1705973837),
    ("ZIZI",   0.1229330505),
    ("ZIIZ",   0.1676831943),
    ("IZZI",   0.1676831943),
    ("IZIZ",   0.1229330505),
    ("IIZZ",   0.1762764072),

    ("XXYY",  -0.0447501439),
    ("YYXX",  -0.0447501439),
    ("XYYX",   0.0447501439),
    ("YXXY",   0.0447501439),
])


# ============================================================
# VQE ansatz
# ============================================================

def h2_ansatz(theta):
    qc = QuantumCircuit(4)

    # Hartree-Fock |1100>
    qc.x(0)
    qc.x(1)

    # Variational layer
    for i in range(4):
        qc.ry(theta[i], i)

    # Entanglement
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    # Second variational layer
    for i in range(4):
        qc.ry(theta[4 + i], i)

    return qc


# ============================================================
# Helpers
# ============================================================

def label_to_qubit_ops(label):
    """
    Qiskit Pauli labels are little-endian: the RIGHTMOST character acts
    on qubit 0. Return a list of (qubit, pauli_char) so that qubit
    indices match what Statevector / SparsePauliOp use.
    """
    n = len(label)
    return [(q, label[n - 1 - q]) for q in range(n)]


# ============================================================
# Exact (no cutting) evaluation
# ============================================================

def expectation_exact(theta, pauli):
    sv = Statevector(h2_ansatz(theta))
    return float(sv.expectation_value(SparsePauliOp(pauli)).real)


def energy_exact(theta):
    sv = Statevector(h2_ansatz(theta))
    return float(sv.expectation_value(H2).real)


# ============================================================
# Evaluate one Pauli term using circuit cutting
# ============================================================

def expectation_cut(theta, pauli, agent):
    qc = h2_ansatz(theta)
    z_qubits = []

    for q, p in label_to_qubit_ops(pauli):
        if p == "X":
            qc.h(q)
            z_qubits.append(q)
        elif p == "Y":
            qc.sdg(q)
            qc.h(q)
            z_qubits.append(q)
        elif p == "Z":
            z_qubits.append(q)

    if len(z_qubits) == 0:
        return 1.0

    cc = CircuitCutting(
        qc,
        agent=agent,
        observable=z_qubits,
        qubits_per_subcircuit=2,
        partitioner="gurobi",
        shots= 2 ** 14,
        samples=10000,
        run_actual=True,
        optimization_level=1,
        save=False,
        verbose=False,
    )

    metrics, artifacts = cc.run()
    return metrics["cutting_expval"]


def energy_cut(theta, agent, verbose=False):
    E = 0.0
    for pauli, coeff in zip(H2.paulis.to_labels(), H2.coeffs.real):
        expval = expectation_cut(theta, pauli, agent)
        contribution = coeff * expval
        E += contribution
        if verbose:
            print(
                f"{pauli:4s} "
                f"coeff={coeff:+.8f} "
                f"<P>={expval:+.6f} "
                f"contribution={contribution:+.6f}"
            )
    return E


# =================================================================================
# VQE runner that records the energy history
# =================================================================================

def run_vqe(energy_fn, theta0, maxiter, label):
    history = []
    thetas = []

    def wrapped(theta):
        E = energy_fn(theta)
        history.append(E)
        thetas.append(np.array(theta, copy=True))
        print(f"[{label}] eval {len(history):4d}  E = {E:.8f} Ha")
        return E

    result = minimize(wrapped, theta0, method="COBYLA", options={"maxiter": maxiter})
    return result, np.array(history), thetas    


# =================================================================================
# Plotting
# =================================================================================

def make_plots(hist_exact, hist_cut, hist_cut_true, E_ref,
               terms, per_term_exact, per_term_cut, outfile):

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    # --- Panel 1: energy convergence ---------------------------------
    ax = axes[0]
    ax.plot(hist_exact, label="No cutting (exact statevector)", lw=2)
    ax.plot(hist_cut, label="Circuit cutting (sampled)", lw=1.5, alpha=0.85)
    ax.plot(hist_cut_true, "--", label="Exact energy at cut-VQE parameters", lw=1.5)
    ax.axhline(E_ref, color="k", ls=":", label=f"Exact diagonalisation ({E_ref:.5f} Ha)")
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("H2 VQE convergence")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Panel 2: error vs reference ---------------------------------
    ax = axes[1]
    ax.semilogy(np.abs(hist_exact - E_ref) + 1e-12, label="No cutting", lw=2)
    ax.semilogy(np.abs(hist_cut - E_ref) + 1e-12, label="Circuit cutting", lw=1.5)
    ax.semilogy(np.abs(hist_cut_true - E_ref) + 1e-12, "--",
                label="Exact energy at cut-VQE params", lw=1.5)
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("|E - E_exact| (Ha)")
    ax.set_title("Absolute error")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # --- Panel 3: per-Pauli-term expectation values ------------------
    ax = axes[2]
    x = np.arange(len(terms))
    w = 0.4
    ax.bar(x - w / 2, per_term_exact, w, label="No cutting")
    ax.bar(x + w / 2, per_term_cut, w, label="Circuit cutting")
    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=60)
    ax.set_ylabel("<P>")
    ax.set_title("Per-term expectation (final cut-VQE parameters)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"Saved plot to {outfile}")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--cpus", type=int, default=48)
    parser.add_argument("--plot", type=str, default="h2_vqe_comparison.png")
    parser.add_argument("--data", type=str, default="h2_vqe_results.json")
    args = parser.parse_args()

    theta0 = np.zeros(8)

    # Reference: exact diagonalisation of the Hamiltonian
    E_ref = float(np.linalg.eigvalsh(H2.to_matrix()).min())
    print(f"Exact ground-state energy (diagonalisation) = {E_ref:.8f} Ha")

    # ---- VQE without circuit cutting (fast) ----
    res_exact, hist_exact, _ = run_vqe(energy_exact, theta0, args.maxiter, "exact")

    # ---- VQE with circuit cutting ----
    agent = RayAgent(num_cpus_per_node=args.cpus)
    agent.initialise()
    try:
        res_cut, hist_cut, thetas_cut = run_vqe(
            lambda th: energy_cut(th, agent), theta0, args.maxiter, "cut"
        )

        # Per-term comparison at the cut-VQE optimum
        terms = H2.paulis.to_labels()
        per_term_exact = [expectation_exact(res_cut.x, p) for p in terms]
        per_term_cut = [expectation_cut(res_cut.x, p, agent) for p in terms]
    finally:
        agent.stop_ray_clusters()

    # Exact energy evaluated at every parameter vector the cut run visited
    # (shows how good the parameters are, free of sampling noise)
    hist_cut_true = np.array([energy_exact(th) for th in thetas_cut])

    # ---- Summary ----
    print("\n==============================")
    print("VQE RESULT")
    print("==============================")
    print(f"Reference (diagonalisation)      : {E_ref:.8f} Ha")
    print(f"VQE, no cutting                  : {res_exact.fun:.8f} Ha")
    print(f"VQE, circuit cutting (sampled)   : {res_cut.fun:.8f} Ha")
    print(f"Exact energy at cut-VQE params   : {energy_exact(res_cut.x):.8f} Ha")
    print("Optimal theta (no cutting):", res_exact.x)
    print("Optimal theta (cutting)   :", res_cut.x)

    # Save raw data so plots can be regenerated without rerunning
    with open(args.data, "w") as f:
        json.dump(
            {
                "E_ref": E_ref,
                "hist_exact": hist_exact.tolist(),
                "hist_cut": hist_cut.tolist(),
                "theta_exact": res_exact.x.tolist(),
                "theta_cut": res_cut.x.tolist(),
                "terms": list(terms),
                "per_term_exact": per_term_exact,
                "per_term_cut": per_term_cut,
            },
            f,
            indent=2,
        )

    make_plots(hist_exact, hist_cut, hist_cut_true, E_ref,
               terms, per_term_exact, per_term_cut, args.plot)


if __name__ == "__main__":
    main()