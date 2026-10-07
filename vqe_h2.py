# h2_vqe_cut.py

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

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
    ("XY YX".replace(" ", ""), 0.0447501439),
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
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.ry(theta[2], 2)
    qc.ry(theta[3], 3)

    # Entanglement
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    # Second variational layer
    qc.ry(theta[4], 0)
    qc.ry(theta[5], 1)
    qc.ry(theta[6], 2)
    qc.ry(theta[7], 3)

    return qc


# ============================================================
# Evaluate one Pauli term using circuit cutting
# ============================================================

def expectation_cut(theta, pauli, agent):
    # Build variational circuit
    qc = h2_ansatz(theta)
    for q, p in enumerate(pauli):
        if p == "X":
            qc.h(q)

        elif p == "Y":
            qc.sdg(q)
            qc.h(q)

    z_qubits = [
        q for q, p in enumerate(pauli)
        if p != "I"
    ]
    if len(z_qubits) == 0:
        return 1.0

    cc = CircuitCutting(
        qc,
        agent=agent,
        observable=z_qubits,
        qubits_per_subcircuit=2,
        partitioner="gurobi",
        shots=2048,
        samples=10000,
        run_actual=True,
        optimization_level=1,
        save=False,
        verbose=False,
    )

    metrics, artifacts = cc.run()

    return metrics["cutting_expval"]


# ============================================================
# VQE energy
# ============================================================

def energy(theta, agent, verbose=False):

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
    print(f"E(theta) = {E:.8f} Ha")
    return E


# ============================================================
# Main VQE
# ============================================================

def main():
    agent = RayAgent(num_cpus_per_node=48)
    agent.initialise()
    theta0 = np.zeros(8)

    result = minimize(lambda theta: energy(theta, agent), theta0, method="COBYLA", options={"maxiter": 20,},)

    print("\n==============================")
    print("VQE RESULT")
    print("==============================")
    print("Optimal theta:")
    print(result.x)
    print(f"\nGround-state energy = {result.fun:.8f} Ha")
    print("\nOptimizer:")
    print(result.message)

    agent.stop_ray_clusters()

if __name__ == "__main__":
    main()