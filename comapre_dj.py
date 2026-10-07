"""
Benchmark quantum algorithms on the Qiskit Aer statevector simulator
and append the timing / configuration / result details to a CSV file.

Example:
    python dj_benchmark.py --algorithm dj --qubits 10 12 14 --oracle balanced \
        --shots 2048 --repeats 3 --output results.csv
"""
import argparse
import csv
import json
import os
import platform
import socket
import time
from datetime import datetime

import qiskit
import qiskit_aer
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

try:
    import resource  # Unix only
except ImportError:
    resource = None


# --------------------------------------------------------------------------
# Circuits
# --------------------------------------------------------------------------
def build_dj_circuit(n, oracle="balanced"):
    """
    Deutsch-Jozsa circuit on n input qubits + 1 ancilla.

    oracle:
        "balanced"  -> f(x) = x_0 xor x_1 xor ... xor x_{n-1}
                       expected measurement: anything other than 00...0
                       (for this oracle: 11...1)
        "constant0" -> f(x) = 0, expected measurement: 00...0
        "constant1" -> f(x) = 1, expected measurement: 00...0

    Returns (circuit, expected_bitstring).
    """
    q = QuantumRegister(n, "q")
    ancilla = QuantumRegister(1, "ancilla")
    c = ClassicalRegister(n, "c")
    qc = QuantumCircuit(q, ancilla, c)

    qc.x(ancilla[0])
    qc.h(q)
    qc.h(ancilla[0])

    if oracle == "constant0":
        pass
    elif oracle == "constant1":
        qc.x(ancilla[0])
    elif oracle == "balanced":
        for i in range(n):
            qc.cx(q[i], ancilla[0])
    else:
        raise ValueError(f"Invalid oracle type: {oracle}")

    qc.h(q)
    qc.measure(q, c)

    expected = "1" * n if oracle == "balanced" else "0" * n
    return qc, expected


# name -> builder(n, **kwargs) returning (circuit, expected_bitstring or None)
ALGORITHMS = {
    "dj": lambda n, args: build_dj_circuit(n, args.oracle),
}


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------
def build_backend(args):
    opts = dict(
        method="statevector",
        device=args.device,
        precision=args.precision,
        max_parallel_threads=args.threads,
    )
    if args.blocking_qubits is not None:
        opts["blocking_enable"] = True
        opts["blocking_qubits"] = args.blocking_qubits
    return AerSimulator(**opts)


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
def peak_rss_mb():
    if resource is None:
        return ""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes
    return round(rss / 1024 if platform.system() == "Linux" else rss / 1024 / 1024, 2)


def run_once(qc, backend, shots):
    t0 = time.perf_counter()
    job = backend.run(qc, shots=shots)
    result = job.result()
    wall = time.perf_counter() - t0
    return result, wall


def benchmark(args, n, backend):
    # ---- build ----
    t0 = time.perf_counter()
    qc, expected = ALGORITHMS[args.algorithm](n, args)
    build_time = time.perf_counter() - t0

    # ---- transpile ----
    t0 = time.perf_counter()
    tqc = transpile(qc, backend)
    transpile_time = time.perf_counter() - t0

    # ---- timed runs ----
    rows = []
    for rep in range(1, args.repeats + 1):
        result, wall = run_once(tqc, backend, args.shots)
        counts = result.get_counts()
        top_state, top_count = max(counts.items(), key=lambda kv: kv[1])
        success = (counts.get(expected, 0) / args.shots) if expected else ""
        sim_time = getattr(result, "time_taken", "")
        exp_time = ""
        try:
            exp_time = result.results[0].time_taken
        except Exception:
            pass

        rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "host": socket.gethostname(),
                "algorithm": args.algorithm,
                "oracle": args.oracle if args.algorithm == "dj" else "",
                "qubits_input": n,
                "qubits_total": qc.num_qubits,
                "shots": args.shots,
                "repeat": rep,
                "method": "statevector",
                "device": args.device,
                "precision": args.precision,
                "threads": args.threads,
                "blocking_qubits": args.blocking_qubits if args.blocking_qubits is not None else "",
                "circuit_depth": qc.depth(),
                "transpiled_depth": tqc.depth(),
                "gate_count": sum(qc.count_ops().values()),
                "transpiled_gate_count": sum(tqc.count_ops().values()),
                "build_time_s": round(build_time, 6),
                "transpile_time_s": round(transpile_time, 6),
                "run_wall_time_s": round(wall, 6),
                "aer_time_taken_s": sim_time,
                "aer_experiment_time_s": exp_time,
                "peak_rss_mb": peak_rss_mb(),
                "expected_state": expected or "",
                "top_state": top_state,
                "top_state_count": top_count,
                "success_rate": success,
                "unique_outcomes": len(counts),
                "result_counts": json.dumps(counts, sort_keys=True),
                "qiskit_version": qiskit.__version__,
                "qiskit_aer_version": qiskit_aer.__version__,
                "python_version": platform.python_version(),
            }
        )
    return rows

def append_csv(path, rows):
    if not rows:
        return
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Benchmark quantum algorithms with the Aer statevector simulator."
    )
    ap.add_argument("--algorithm", type=str, default="dj", choices=sorted(ALGORITHMS),
                    help="Algorithm to benchmark")
    ap.add_argument("--oracle", type=str, default="balanced",
                    choices=["balanced", "constant0", "constant1"],
                    help="Oracle type (used by dj)")
    ap.add_argument("--qubits", type=int, nargs="+", default=[10],
                    help="One or more input-qubit counts, e.g. --qubits 10 12 14")
    ap.add_argument("--blocking_qubits", type=int, default=None,
                    help="Aer cache-blocking qubits (omit to disable blocking)")
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--precision", type=str, default="single", choices=["single", "double"])
    ap.add_argument("--threads", type=int, default=0, help="max_parallel_threads (0 = all)")
    ap.add_argument("--repeats", type=int, default=1, help="Timed runs per configuration")
    ap.add_argument("--warmup", type=int, default=0, help="Untimed warmup runs")
    ap.add_argument("--output", type=str, default="results.csv", help="CSV output path")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    backend = build_backend(args)

    for n in args.qubits:
        rows = benchmark(args, n, backend)
        append_csv(args.output, rows)
        for r in rows:
            print(
                f"[{r['algorithm']}] n={r['qubits_input']} rep={r['repeat']} "
                f"run={r['run_wall_time_s']:.4f}s top={r['top_state']} "
                f"success={r['success_rate']}"
            )

    print(f"Results appended to {args.output}")