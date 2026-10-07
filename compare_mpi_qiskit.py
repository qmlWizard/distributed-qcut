import argparse
import csv
import os
import time
import warnings

from qiskit import QuantumCircuit, qasm2, qasm3, transpile
from qiskit_aer import AerSimulator

CIRCUIT_DIR = "circuits/bchmrk_data"
IGNORED_OPS = {"measure", "barrier", "reset", "delay", "snapshot"}

# Rank is only used so that just one process prints / writes the CSV when
# the script is launched under mpirun/srun. No MPI library is used here.
RANK = int(
    os.environ.get("OMPI_COMM_WORLD_RANK")
    or os.environ.get("PMI_RANK")
    or os.environ.get("SLURM_PROCID")
    or 0
)

CSV_FIELDS = [
    "circuit_name",
    "circuit_path",
    "num_qubits",
    "layers",              # depth of the original circuit
    "num_gates",
    "single_qubit_gates",
    "two_qubit_gates",
    "multi_qubit_gates",   # 3+ qubit gates (ccx, cswap, ...)
    "transpiled_layers",
    "transpiled_num_gates",
    "shots",
    "execution_time_s",    # wall-clock time of backend.run(...).result()
    "aer_time_taken_s",    # time reported by Aer itself
    "status",
]


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


# ------------------------------------------------------------
# LOAD CIRCUITS
# ------------------------------------------------------------

def load_circuit(qasm_dir):
    """Load the first non-transpiled .qasm file in qasm_dir. Returns (name, path, circuit)."""
    files = sorted(os.listdir(qasm_dir))
    qasm_files = [f for f in files if f.endswith(".qasm") and "_transpiled" not in f]
    if not qasm_files:
        raise FileNotFoundError(f"No .qasm files found in {qasm_dir}")
    qasm_file = qasm_files[0]
    path = os.path.join(qasm_dir, qasm_file)
    log(f"Loading: {path}")

    with open(path, "r") as f:
        qasm = f.read()

    try:
        qc = qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    except Exception:
        try:
            qc = qasm3.loads(qasm)
        except Exception as e:
            print(f"Failed to load {qasm_file}: {type(e).__name__}: {e}")
            return os.path.basename(qasm_dir), path, None
    return os.path.basename(qasm_dir), path, qc


def get_circuits(size="small", num=1):
    path = os.path.join(CIRCUIT_DIR, size)
    dirs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    dirs = dirs[: min(num, len(dirs))]
    return [load_circuit(os.path.join(path, d)) for d in dirs]


# ------------------------------------------------------------
# CIRCUIT STATS
# ------------------------------------------------------------

def gate_stats(qc: QuantumCircuit):
    single = two = multi = 0
    for inst in qc.data:
        if inst.operation.name in IGNORED_OPS:
            continue
        n = len(inst.qubits)
        if n == 1:
            single += 1
        elif n == 2:
            two += 1
        else:
            multi += 1
    return {
        "num_gates": single + two + multi,
        "single_qubit_gates": single,
        "two_qubit_gates": two,
        "multi_qubit_gates": multi,
    }


# ------------------------------------------------------------
# BACKEND
# ------------------------------------------------------------

def build_backend(args):
    opts = dict(
        method="statevector",
        device=args.device,
        precision=args.precision,
        max_parallel_threads=args.threads,
    )
    # Aer's own option for distributing the statevector when run under mpirun.
    if args.blocking_qubits is not None:
        opts["blocking_enable"] = True
        opts["blocking_qubits"] = args.blocking_qubits
    return AerSimulator(**opts)


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

def run_one(name, path, qc, backend, args):
    row = {k: "" for k in CSV_FIELDS}
    row.update(circuit_name=name, circuit_path=path, shots=args.shots)

    if qc is None:
        row["status"] = "load_failed"
        return row

    row["num_qubits"] = qc.num_qubits
    row["layers"] = qc.depth()
    row.update(gate_stats(qc))

    run_qc = qc.copy()
    if not any(i.operation.name == "measure" for i in run_qc.data):
        run_qc.measure_all()

    try:
        tqc = run_qc
        row["transpiled_layers"] = tqc.depth()
        row["transpiled_num_gates"] = sum(
            v for k, v in tqc.count_ops().items() if k not in IGNORED_OPS
        )
        t0 = time.perf_counter()
        result = backend.run(tqc, shots=args.shots).result()
        elapsed = time.perf_counter() - t0

        row["status"] = "ok" if result.success else f"failed: {result.status}"
        row["execution_time_s"] = f"{elapsed:.6f}"
        row["aer_time_taken_s"] = f"{getattr(result, 'time_taken', float('nan')):.6f}"
    except Exception as e:
        row["status"] = f"error: {type(e).__name__}: {e}"

    return row


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run QASMBench circuits on Aer statevector and record execution time to CSV."
    )
    ap.add_argument("--size", type=str, default="small", help="small, medium, large")
    ap.add_argument("--num-circuits", type=int, default=10)
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--device", type=str, default="CPU")
    ap.add_argument("--precision", type=str, default="single")
    ap.add_argument("--blocking_qubits", type=int, default=None, help="If set, enables Aer blocking_enable with this many blocking qubits")
    ap.add_argument("--threads", type=int, default=0, help="max_parallel_threads (0 = all)")
    ap.add_argument("--output", type=str, default="results.csv", help="CSV output path")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")

    circuits = get_circuits(args.size, args.num_circuits)
    backend = build_backend(args)

    rows = []
    for name, path, qc in circuits:
        log(f"Running: {name}")
        row = run_one(name, path, qc, backend, args)
        rows.append(row)
        log(f"  -> {row['status']}  time={row['execution_time_s']}s")

    if RANK == 0:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()