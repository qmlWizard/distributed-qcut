import numpy as np
import ray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers import PrimitiveResult
from qiskit_aer.primitives import SamplerV2

from qiskit_addon_cutting import (
    cut_wires,
    expand_observables,
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)
from qiskit_addon_cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)

# ---------- Settings ----------
SHOTS = 2**14
QUBITS_PER_SUBCIRCUIT = 2      # max width of any subcircuit (main knob)
MAX_GAMMA = 100_000            # max sampling overhead find_cuts may accept
MC_SAMPLES = 10_000            # Monte Carlo samples, used only if overhead is too big to enumerate
ENUMERATE_BELOW = 1e4          # enumerate all terms (num_samples=np.inf) if overhead <= this

# ---------- Grover building blocks ----------
def mcz(qc, qubits):
    """Multi-controlled Z across the given qubits."""
    qc.append(ZGate().control(len(qubits) - 1), qubits)

def oracle(n, marked):
    qc = QuantumCircuit(n, name="oracle")
    zeros = [q for q in range(n) if marked[n - 1 - q] == "0"]
    qc.x(zeros)
    mcz(qc, list(range(n)))
    qc.x(zeros)
    return qc

def diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    mcz(qc, list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc

def grover(n, marked, iterations=None):
    N = 2**n
    if iterations is None:
        iterations = int(np.floor(np.pi / 4 * np.sqrt(N)))
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle(n, marked), inplace=True)
        qc.compose(diffuser(n), inplace=True)
    return qc, iterations

def marked_projector(marked):
    proj = None
    for ch in marked:
        sign = 1 if ch == "0" else -1
        op = SparsePauliOp.from_list([("I", 0.5), ("Z", 0.5 * sign)])
        proj = op if proj is None else proj.tensor(op)
    return proj.simplify()


# ---------- Ray worker ----------
@ray.remote(num_cpus=1)
def run_chunk(circuits, shots):
    sampler = SamplerV2(options={"backend_options": {"max_parallel_threads": 1}})
    result = sampler.run(circuits, shots=shots).result()
    return list(result)

def chunked(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]

def run_parallel(subexperiments, shots):
    n_cpus = int(ray.cluster_resources()["CPU"])
    futures = {}
    for label, circuits in subexperiments.items():
        circuits = list(circuits)
        size = max(1, len(circuits) // (n_cpus * 4))   # ~4 chunks per core
        futures[label] = [run_chunk.remote(c, shots) for c in chunked(circuits, size)]

    results = {}
    for label, refs in futures.items():
        # ray.get on a list preserves submission order -> stays aligned with `coefficients`
        pub_results = [pr for chunk in ray.get(refs) for pr in chunk]
        results[label] = PrimitiveResult(pub_results)
    return results


# ---------- Main ----------
if __name__ == "__main__":
    n, marked = 4, "1011"
    circuit, iters = grover(n, marked)

    circuit = transpile(circuit, basis_gates=["cx", "u"], optimization_level=1)
    print(f"Grover n={n}, marked={marked}, iterations={iters}")
    print("CX count:", circuit.count_ops().get("cx", 0))

    observable = marked_projector(marked)

    # Exact reference
    exact = StatevectorEstimator().run([(circuit, observable)]).result()[0].data.evs
    print("Exact P(marked):", exact)

    # ---------- Automated cut finding (replaces hand-picked partition_labels) ----------
    optimization = OptimizationParameters(seed=111, gate_lo=True, wire_lo=True, max_gamma=MAX_GAMMA)
    constraints = DeviceConstraints(qubits_per_subcircuit=QUBITS_PER_SUBCIRCUIT)

    try:
        cut_circuit, metadata = find_cuts(
            circuit, optimization=optimization, constraints=constraints
        )
    except ValueError as e:
        raise SystemExit(
            f"find_cuts found no solution within max_gamma={MAX_GAMMA}: {e}\n"
            "Raise QUBITS_PER_SUBCIRCUIT / MAX_GAMMA, or reduce the CX count first."
        )

    overhead = metadata["sampling_overhead"]
    print(f"find_cuts: {len(metadata['cuts'])} cuts, sampling overhead {overhead}")
    for cut in metadata["cuts"]:
        print(f"  {cut[0]} at instruction index {cut[1]}")

    # Apply wire cuts, expand the observable to match, then partition automatically
    qc_cut = cut_wires(cut_circuit)
    obs_expanded = expand_observables(observable.paulis, circuit, qc_cut)

    problem = partition_problem(circuit=qc_cut, observables=obs_expanded)
    print("Subcircuits:", {k: v.num_qubits for k, v in problem.subcircuits.items()})

    num_samples = np.inf if overhead <= ENUMERATE_BELOW else MC_SAMPLES
    print("num_samples:", num_samples)

    subexperiments, coefficients = generate_cutting_experiments(
        circuits=problem.subcircuits,
        observables=problem.subobservables,
        num_samples=num_samples,
    )
    print({k: len(v) for k, v in subexperiments.items()}, "experiments")

    # Run in parallel with Ray
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _metrics_export_port=None,
    )
    results = run_parallel(subexperiments, SHOTS)
    ray.shutdown()

    # Knit (use the ORIGINAL observable's coefficients, as in the Qiskit docs)
    reconstructed = reconstruct_expectation_values(
        results, coefficients, problem.subobservables
    )
    knitted = np.dot(reconstructed, observable.coeffs).real
    print("Knitted P(marked):", knitted)
    print("Error:", abs(exact - knitted))