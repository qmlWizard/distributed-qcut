#!/usr/bin/env python3
"""
Grover's algorithm runner (single node, or MPI-distributed statevector).

Examples
--------
  # Laptop / single node
  python grover_run.py 4-9
  python grover_run.py 3,5,7 --shots 4096 --json

  # 40 qubits on 128 nodes (MPI-enabled qiskit-aer required), benchmark run
  export OMP_NUM_THREADS=48
  srun -N 128 -n 128 --ntasks-per-node=1 --cpus-per-task=48 --cpu-bind=cores \
       python grover_run.py 40 --mpi --no-ancilla --precision single \
       --max-total-qubits 40 --iterations 1 --blocking-qubits 28 --threads 48

The marked state is |00...0>.

MPI notes
---------
* Aer calls MPI_Init itself; no mpi4py needed.
* With a distributed statevector EVERY rank must run the SAME circuit, so the
  qubit sizes are run sequentially on all ranks (not split between ranks).
* Only rank 0 prints results.
"""
import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# qc.mcx(mode=...) is deprecated in Qiskit >= 2.1 but still works.
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ------------------------------------------------------------
# Circuit
# ------------------------------------------------------------
def n_ancillas(n_data, use_ancilla):
    n_controls = n_data - 1
    return n_controls - 2 if (use_ancilla and n_controls >= 3) else 0


def create_grover_circuit(n_data, iterations, use_ancilla=True):
    """Grover circuit marking |0...0>, measuring only the data qubits."""
    n_anc = n_ancillas(n_data, use_ancilla)

    qc = QuantumCircuit(n_data + n_anc, n_data)
    data = list(range(n_data))
    anc = list(range(n_data, n_data + n_anc))
    controls, target = data[:-1], data[-1]

    def mcz():
        # Phase flip on |1...1> = H(target) . MCX . H(target)
        qc.h(target)
        if n_anc:
            qc.mcx(controls, target, ancilla_qubits=anc, mode="v-chain")
        else:
            qc.mcx(controls, target)   # native 'mcx' gate in Aer, no ancillas
        qc.h(target)

    def phase_flip_zeros():
        qc.x(data)
        mcz()
        qc.x(data)

    qc.h(data)
    for _ in range(iterations):
        phase_flip_zeros()   # oracle
        qc.h(data)
        phase_flip_zeros()   # reflection about |0...0>
        qc.h(data)           # = diffusion (up to global phase)

    qc.measure(data, data)
    return qc


def optimal_iterations(n_data):
    return max(1, int(math.floor(math.pi / 4 * math.sqrt(2 ** n_data))))


# ------------------------------------------------------------
# Analysis / MPI helpers
# ------------------------------------------------------------
def expectation_zn(counts):
    """<Z^{(x)n}> from counts: +1 for even parity, -1 for odd."""
    total = sum(counts.values())
    e = sum((1 if bs.count("1") % 2 == 0 else -1) * c for bs, c in counts.items())
    return e / total


def env_rank():
    for k in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID"):
        if k in os.environ:
            return int(os.environ[k])
    return 0


def run_grover(n, args, backend):
    iters = args.iterations if args.iterations is not None else optimal_iterations(n)
    qc = create_grover_circuit(n, iters, use_ancilla=not args.no_ancilla)
    tqc = qc 
    print("Grover Circuit Created. Running on Backend")
    t0 = time.perf_counter()
    result = backend.run(tqc, shots=args.shots, seed_simulator=args.seed).result()
    elapsed = time.perf_counter() - t0
    print("Run Complete")

    counts = Counter({k.replace(" ", ""): v for k, v in result.get_counts().items()})
    meta = result.to_dict().get("metadata", {}) or {}
    marked = "0" * n
    top, top_count = counts.most_common(1)[0]
    return {
        "n_qubits": n,
        "total_qubits": qc.num_qubits,
        "iterations": iters,
        "shots": args.shots,
        "marked_state": marked,
        "p_marked": counts.get(marked, 0) / args.shots,
        "p_theory": math.sin((2 * iters + 1) * math.asin(2 ** (-n / 2))) ** 2,
        "top_outcome": top,
        "top_count": top_count,
        "success": top == marked,
        "expval_Zn": expectation_zn(counts),
        "depth": tqc.depth(),
        "run_time_s": round(elapsed, 4),
        "mpi_rank": meta.get("mpi_rank", env_rank()),
        "num_mpi_processes": meta.get("num_mpi_processes", 1),
    }


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_range(text):
    """'4-8' -> [4..8]; '3,5,7' -> [3,5,7]; '6' -> [6]."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    out = sorted(set(out))
    if not out or min(out) < 2:
        raise argparse.ArgumentTypeError("qubit counts must be >= 2")
    return out


def build_backend(args):
    opts = dict(
        method="statevector",
        device=args.device,
        precision=args.precision,
        max_parallel_threads=args.threads,
    )
    if args.mpi:
        opts["blocking_enable"] = True
        opts["blocking_qubits"] = args.blocking_qubits
        if args.device == "GPU":
            opts["batched_shots_gpu"] = args.batched_shots_gpu
    return AerSimulator(**opts)


def main():
    ap = argparse.ArgumentParser(description="Run Grover's algorithm for a range of qubit counts.")
    ap.add_argument("--qubits", type=parse_range, help="e.g. 4-8, 3,5,7, or 40")
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--iterations", type=int, default=None,
                    help="Grover iterations (default: optimal floor(pi/4*sqrt(2^n)); "
                         "infeasible for large n -- set a small number)")
    ap.add_argument("--no-ancilla", action="store_true",
                    help="Use plain mcx without v-chain ancillas (total qubits = n). Needed for n~40.")
    ap.add_argument("--precision", choices=["single", "double"], default="double",
                    help="Statevector precision (single halves memory)")
    ap.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--threads", type=int, default=0,
                    help="OpenMP threads per process (0 = all available)")
    ap.add_argument("--mpi", action="store_true",
                    help="Enable distributed statevector (blocking_enable); launch with mpirun/srun")
    ap.add_argument("--blocking-qubits", type=int, default=28,
                    help="Chunk size in qubits for --mpi. Need 16*2^(bq+4) bytes < smallest memory space "
                         "(double precision; halve for single)")
    ap.add_argument("--batched-shots-gpu", action="store_true",
                    help="GPU only: distribute shots across processes")
    ap.add_argument("--seed", type=int, default=12345,
                    help="simulator seed, identical on all ranks")
    ap.add_argument("--opt-level", type=int, default=0, choices=[0, 1, 2, 3],
                    help="transpile optimization level (0 recommended for huge circuits)")
    ap.add_argument("--max-total-qubits", type=int, default=30,
                    help="Skip sizes whose circuit (incl. ancillas) exceeds this")
    ap.add_argument("--strict", action="store_true",
                    help="Exit code 1 if any size fails to find the marked state "
                         "(off by default so benchmark runs don't look failed)")
    ap.add_argument("--json", action="store_true", help="Print results as JSON")
    args = ap.parse_args()

    print("Inside the code")
    rank0 = env_rank() == 0
    backend = build_backend(args)

    results = []
    for n in args.qubits:
        total = n + n_ancillas(n, not args.no_ancilla)
        if total > args.max_total_qubits:
            if rank0:
                print(f"[skip] n={n}: needs {total} qubits > limit {args.max_total_qubits}", file=sys.stderr)
            continue
        results.append(run_grover(n, args, backend))

    # Only rank 0 prints. Rank is taken from Aer's metadata (falls back to env vars).
    my_rank = results[0]["mpi_rank"] if results else env_rank()
    if my_rank == 0:
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            w = max(6, args.qubits[-1])
            hdr = (f"{'n':>3} {'tot':>4} {'iter':>7} {'P(marked)':>10} {'P(theory)':>10} "
                   f"{'<Z^n>':>8} {'top':>{w}} {'ok':>3} {'depth':>7} {'time(s)':>9}")
            print(hdr)
            print("-" * len(hdr))
            for r in results:
                print(f"{r['n_qubits']:>3} {r['total_qubits']:>4} {r['iterations']:>7} "
                      f"{r['p_marked']:>10.4g} {r['p_theory']:>10.4g} {r['expval_Zn']:>8.4f} "
                      f"{r['top_outcome']:>{w}} {'Y' if r['success'] else 'N':>3} "
                      f"{r['depth']:>7} {r['run_time_s']:>9.3f}")
            if results:
                print(f"\n(processes: {results[0]['num_mpi_processes']}, "
                      f"precision: {args.precision}, device: {args.device})")

    if args.strict and my_rank == 0:
        sys.exit(0 if results and all(r["success"] for r in results) else 1)
    sys.exit(0)


if __name__ == "__main__":
    main()