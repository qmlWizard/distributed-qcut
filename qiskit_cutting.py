"""
Distributed Qiskit circuit cutting + knitting sweep.

Slurm and Ray cluster process management are handled outside this program.
This Python driver only connects to an already-running Ray cluster (via the
RAY_ADDRESS environment variable) and executes the circuit-cutting workload.

    Slurm
      +-- Ray head
      +-- Ray workers
      +-- Python driver
"""

import argparse
import csv
import json
import math
import os
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime

import gurobipy as gp
import numpy as np
import ray
from gurobipy import GRB
from qiskit import QuantumCircuit, transpile
from qiskit.primitives.containers import PrimitiveResult
from qiskit.quantum_info import Pauli, PauliList
from qiskit_addon_cutting import generate_cutting_experiments, partition_problem, reconstruct_expectation_values
from qiskit_aer.primitives import SamplerV2

RAY_TASK_CPUS = 1
SEP_WIDTH = 96
OK_STATUSES = ("ok", "skipped_gamma")


# ============================================================
# LOGGING / TIMING
# ============================================================

def log(message):
    print(f"[QISKIT] {message}", flush=True)


@contextmanager
def timed(store, key, tag=""):
    """Time a block, store seconds in store[key], and log it."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - t0
        log(f"{tag}{key:<26}: {store[key]:10.3f} s")


def finalize_timings(metrics, timings, run_start):
    """Record total runtime and copy all timings into metrics as t_<name>."""
    timings["total"] = time.perf_counter() - run_start
    metrics.update({f"t_{k}": v for k, v in timings.items()})


# ============================================================
# RAY
# ============================================================

def connect_ray():
    """Connect to the Ray cluster started by the Slurm job script."""
    ray_address = os.environ.get("RAY_ADDRESS")
    if not ray_address:
        raise RuntimeError("RAY_ADDRESS is not set.\nThe Slurm job script must start the Ray cluster and export RAY_ADDRESS before launching this Python driver.")

    log(f"Connecting to Ray at: {ray_address}")
    t0 = time.perf_counter()
    ray.init(address=ray_address, ignore_reinit_error=True, include_dashboard=False, logging_level="WARNING")
    init_time = time.perf_counter() - t0

    total_cpus = int(ray.cluster_resources().get("CPU", 0))
    ray_nodes = [node for node in ray.nodes() if node.get("Alive")]

    log("=" * 70)
    log("RAY CONNECTION")
    log("=" * 70)
    log(f"Ray address    : {ray_address}")
    log(f"Ray nodes      : {len(ray_nodes)}")
    log(f"Ray CPUs       : {total_cpus}")
    log(f"Connection time: {init_time:.3f} s")
    log("=" * 70)

    if total_cpus <= 0:
        raise RuntimeError("Connected to Ray, but Ray reports zero CPUs.")
    return total_cpus, ray_nodes, init_time


@ray.remote(num_cpus=RAY_TASK_CPUS)
def run_chunk(circuits, shots):
    """Execute a chunk of circuits on one CPU (Aer restricted to one thread: 1 Ray task = 1 CPU)."""
    sampler = SamplerV2(options={"backend_options": {"max_parallel_threads": 1, "max_parallel_experiments": 1}})
    return list(sampler.run(circuits, shots=shots).result())


def chunked(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def run_parallel(subexperiments, shots, total_cpus, task_multiplier):
    """Submit all subexperiments to Ray. Returns (results_dict, info_dict)."""
    labels = list(subexperiments.keys())
    total_circuits = sum(len(subexperiments[label]) for label in labels)

    if total_circuits == 0:
        empty_info = {"num_tasks": 0, "chunk_size": 0, "total_circuits": 0, "est_cpu_utilization": 0.0}
        return {label: PrimitiveResult([]) for label in labels}, empty_info

    target_tasks = total_cpus * task_multiplier
    chunk_size = max(1, math.ceil(total_circuits / target_tasks))

    submitted = [
        (label, run_chunk.options(scheduling_strategy="SPREAD").remote(chunk, shots))
        for label in labels
        for chunk in chunked(list(subexperiments[label]), chunk_size)
    ]
    num_tasks = len(submitted)
    log(f"Submitted {total_circuits} circuits as {num_tasks} Ray tasks (chunk size {chunk_size}, {total_cpus} CPUs)")

    outputs = ray.get([future for _, future in submitted])
    results = {label: [] for label in labels}
    for (label, _), output in zip(submitted, outputs):
        results[label].extend(output)

    info = {
        "num_tasks": num_tasks,
        "chunk_size": chunk_size,
        "total_circuits": total_circuits,
        "target_tasks": target_tasks,
        # Fraction of the CPU pool covered by the initially runnable task set, not measured utilization.
        "est_cpu_utilization": min(1.0, num_tasks / total_cpus),
    }
    return {label: PrimitiveResult(values) for label, values in results.items()}, info


# ============================================================
# GROVER CIRCUIT
# ============================================================

def create_grover_circuit(n_data, iterations):
    n_controls = n_data - 1
    use_vchain = n_controls >= 3
    n_anc = n_controls - 2 if use_vchain else 0

    qc = QuantumCircuit(n_data + n_anc)
    data = list(range(n_data))
    anc = list(range(n_data, n_data + n_anc))
    controls, target = data[:-1], data[-1]

    def mcx():
        if use_vchain:
            qc.mcx(controls, target, ancilla_qubits=anc, mode="v-chain")
        else:
            qc.mcx(controls, target)

    def flipped_mcx():
        qc.x(data)
        mcx()
        qc.x(data)

    qc.h(data)
    for _ in range(iterations):
        flipped_mcx()
        qc.h(data)
        flipped_mcx()
        qc.h(data)
    return qc


def build_observable(n_total, n_data):
    """Z on the first n_data qubits, identity elsewhere."""
    z = np.zeros(n_total, dtype=bool)
    z[:n_data] = True
    x = np.zeros(n_total, dtype=bool)
    return PauliList([Pauli((z, x))])


def expectation_from_counts(counts, n_data):
    """Compute <Z^n> for the first n_data qubits from all-qubit measurement counts."""
    total_shots = sum(counts.values())
    if total_shots == 0:
        raise RuntimeError("Actual-circuit execution returned zero shots.")

    expectation = 0.0
    for bitstring, count in counts.items():
        # measure_all() maps q_i -> c_i. Qiskit prints classical bits
        # from highest index to lowest, so the last n_data bits are q_0...q_(n_data-1).
        bits = bitstring.replace(" ", "")[-n_data:]
        parity = bits.count("1") % 2
        expectation += (1.0 if parity == 0 else -1.0) * count

    return expectation / total_shots


def stage_actual(tqc, n_data, args, timings, metrics, artifacts, tag):
    """Execute the complete, uncut circuit and compute the reference <Z^n>."""
    with timed(timings, "actual_execution", tag):
        actual_qc = tqc.copy()
        actual_qc.measure_all()

        # Reuse the existing Ray/Aer execution path so the actual run also
        # uses one Aer thread and one Ray CPU.
        future = run_chunk.options(scheduling_strategy="SPREAD").remote(
            [actual_qc], args.shots
        )
        result = ray.get(future)[0]
        counts = result.data.meas.get_counts()
        actual_expval = expectation_from_counts(counts, n_data)

    metrics["actual_expval"] = float(actual_expval)
    metrics["actual_shots"] = int(args.shots)
    artifacts["actual_counts"] = np.array(
        [[str(k), int(v)] for k, v in counts.items()], dtype=object
    )
    log(f"{tag}actual (uncut) <Z^n> = {actual_expval:.6g}")


# ============================================================
# INTERACTION GRAPH + GUROBI
# ============================================================

def build_edge_weights(qc):
    """Weighted interaction graph: weight(a, b) = number of two-qubit gates between a and b."""
    weights = Counter()
    for inst in qc.data:
        if len(inst.qubits) != 2:
            continue
        a, b = sorted(qc.find_bit(q).index for q in inst.qubits)
        weights[(a, b)] += 1
    return weights


def optimize_partition(env, edge_weights, n_qubits, max_per_fragment, time_limit, mip_gap):
    n_frag = math.ceil(n_qubits / max_per_fragment)
    edges = list(edge_weights.keys())

    model = gp.Model("quantum_partition", env=env)
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    x = model.addVars(n_qubits, n_frag, vtype=GRB.BINARY, name="x")
    y = model.addVars(len(edges), lb=0.0, ub=1.0, name="y")

    for q in range(n_qubits):
        model.addConstr(gp.quicksum(x[q, f] for f in range(n_frag)) == 1)

    for f in range(n_frag):
        model.addConstr(gp.quicksum(x[q, f] for q in range(n_qubits)) <= max_per_fragment)

    # Symmetry breaking
    for q in range(n_qubits):
        for f in range(q + 1, n_frag):
            x[q, f].UB = 0

    for e, (a, b) in enumerate(edges):
        for f in range(n_frag):
            model.addConstr(y[e] >= x[a, f] - x[b, f])

    model.setObjective(gp.quicksum(edge_weights[edges[e]] * y[e] for e in range(len(edges))), GRB.MINIMIZE)

    # Warm start
    for q in range(n_qubits):
        for f in range(n_frag):
            x[q, f].Start = 1.0 if f == q // max_per_fragment else 0.0

    t0 = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - t0

    if model.SolCount == 0:
        model.dispose()
        raise RuntimeError("Gurobi did not find a feasible partition.")

    raw = [max(range(n_frag), key=lambda f: x[q, f].X) for q in range(n_qubits)]
    remap = {old: new for new, old in enumerate(sorted(set(raw)))}
    labels = [remap[r] for r in raw]

    info = {
        "gurobi_status": int(model.Status),
        "gurobi_objective": float(model.ObjVal),
        "gurobi_bound": float(model.ObjBound),
        "gurobi_gap": float(model.MIPGap),
        "gurobi_solve_seconds": float(solve_time),
        "gurobi_n_fragments_allowed": n_frag,
        "gurobi_n_edges": len(edges),
    }
    model.dispose()
    return labels, info


# ============================================================
# CUTTING HELPERS
# ============================================================

def basis_kappa(basis):
    if hasattr(basis, "kappa"):
        return float(basis.kappa)
    return math.sqrt(float(basis.overhead))


def cut_statistics(bases):
    """Returns (num_cuts, log10 gamma, log10 number of QPD terms)."""
    log10_gamma = sum(math.log10(basis_kappa(b)) for b in bases)
    log10_terms = sum(math.log10(len(b.maps)) for b in bases)
    return len(bases), log10_gamma, log10_terms


# ============================================================
# RUN STAGES
# ============================================================

def stage_build_and_transpile(n_data, args, timings, metrics, tag):
    with timed(timings, "build_circuit", tag):
        qc = create_grover_circuit(n_data, args.iterations)

    with timed(timings, "transpile", tag):
        tqc = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)

    metrics.update(
        circuit_qubits=tqc.num_qubits,
        n_ancilla=tqc.num_qubits - n_data,
        circuit_depth=tqc.depth(),
        circuit_ops=len(tqc.data),
        circuit_cx=int(tqc.count_ops().get("cx", 0)),
    )
    log(f"{tag}circuit: {tqc.num_qubits} qubits, depth {tqc.depth()}, {len(tqc.data)} ops, {metrics['circuit_cx']} cx")
    return tqc


def stage_partition(tqc, args, gurobi_env, timings, metrics, artifacts, tag):
    with timed(timings, "gurobi_partition", tag):
        edge_weights = build_edge_weights(tqc)
        labels, ginfo = optimize_partition(gurobi_env, edge_weights, tqc.num_qubits, args.qubits_per_subcircuit, args.gurobi_time_limit, args.gurobi_mip_gap)

    metrics.update(ginfo)
    metrics["n_fragments"] = len(set(labels))
    log(f"{tag}partition: {metrics['n_fragments']} fragments, {ginfo['gurobi_objective']:.0f} cut gates (gap {ginfo['gurobi_gap']:.3g})")
    artifacts["partition_labels"] = np.array(labels, dtype=int)
    return labels


def stage_cut(tqc, labels, n_data, args, timings, metrics, tag):
    with timed(timings, "cut_circuit", tag):
        observable = build_observable(tqc.num_qubits, n_data)
        problem = partition_problem(circuit=tqc, partition_labels=labels, observables=observable)
        subcircuits, subobservables, bases = problem.subcircuits, problem.subobservables, problem.bases

    num_cuts, log10_gamma, log10_terms = cut_statistics(bases)
    sizes = [sc.num_qubits for sc in subcircuits.values()]

    metrics.update(
        n_subcircuits=len(subcircuits),
        max_subcircuit_qubits=max(sizes),
        min_subcircuit_qubits=min(sizes),
        n_cuts=num_cuts,
        log10_gamma=log10_gamma,
        gamma=10 ** log10_gamma if log10_gamma < 300 else float("inf"),
        log10_qpd_terms=log10_terms,
        gamma_exceeds_limit=bool(log10_gamma > math.log10(args.max_gamma)),
    )
    log(f"{tag}cutting: {len(subcircuits)} subcircuits, {num_cuts} cuts, log10(gamma)={log10_gamma:.2f}")
    return subcircuits, subobservables, log10_terms


def stage_generate_experiments(subcircuits, subobservables, log10_terms, args, timings, metrics, artifacts, tag):
    enumerate_all = log10_terms < math.log10(args.enumerate_below)
    num_samples = np.inf if enumerate_all else args.samples

    metrics["sampling_mode"] = "enumerate" if enumerate_all else "monte_carlo"
    metrics["mc_samples"] = -1 if enumerate_all else args.samples

    with timed(timings, "generate_experiments", tag):
        subexperiments, coefficients = generate_cutting_experiments(circuits=subcircuits, observables=subobservables, num_samples=num_samples)

    per_fragment = {str(k): len(v) for k, v in subexperiments.items()}
    total_subexp = sum(per_fragment.values())

    metrics.update(
        total_subexperiments=total_subexp,
        n_coefficients=len(coefficients),
        subexperiments_per_fragment=json.dumps(per_fragment),
    )
    log(f"{tag}experiments: {total_subexp} circuits ({metrics['sampling_mode']}, {len(coefficients)} weights)")

    artifacts["subexperiments_per_fragment"] = np.array([per_fragment[str(k)] for k in subexperiments.keys()], dtype=int)
    artifacts["qpd_weights"] = np.array([w for w, _ in coefficients], dtype=float)
    return subexperiments, coefficients, total_subexp


def stage_execute(subexperiments, total_subexp, args, total_cpus, timings, metrics, tag):
    with timed(timings, "ray_execution", tag):
        results, rinfo = run_parallel(subexperiments, args.shots, total_cpus, args.task_multiplier)

    metrics.update(
        ray_num_tasks=rinfo["num_tasks"],
        ray_chunk_size=rinfo["chunk_size"],
        ray_est_cpu_utilization=rinfo["est_cpu_utilization"],
        ray_cpus=total_cpus,
        shots=args.shots,
    )
    metrics["circuits_per_second"] = total_subexp / timings["ray_execution"] if timings["ray_execution"] > 0 else float("nan")
    return results


def stage_reconstruct(results, coefficients, subobservables, timings, metrics, artifacts, tag):
    with timed(timings, "reconstruction", tag):
        expvals = reconstruct_expectation_values(results, coefficients, subobservables)

    expval = float(np.real(expvals[0]))
    metrics["reconstructed_expval"] = expval
    artifacts["reconstructed_expvals"] = np.real(np.array(expvals, dtype=complex))
    log(f"{tag}reconstructed <Z^n> = {expval:.6g}")


# ============================================================
# ONE RUN
# ============================================================

def run_single(n_data, args, gurobi_env, total_cpus):
    tag = f"[N={n_data}] "
    timings, artifacts = {}, {}
    metrics = {
        "n_qubits": n_data,
        "actual_expval": None,
        "cutting_expval": None,
        "absolute_error": None,
        "relative_error_percent": None,
        "actual_run": bool(args.run_actual),
    }
    run_start = time.perf_counter()

    log(f"{tag}{'=' * 40}")
    log(f"{tag}starting run: {n_data} data qubits, max {args.qubits_per_subcircuit} qubits/subcircuit")

    tqc = stage_build_and_transpile(n_data, args, timings, metrics, tag)

    if args.run_actual:
        stage_actual(tqc, n_data, args, timings, metrics, artifacts, tag)
    else:
        log(f"{tag}actual (uncut) run disabled")

    labels = stage_partition(tqc, args, gurobi_env, timings, metrics, artifacts, tag)
    subcircuits, subobservables, log10_terms = stage_cut(tqc, labels, n_data, args, timings, metrics, tag)

    if metrics["gamma_exceeds_limit"]:
        log(f"{tag}WARNING: gamma exceeds {args.max_gamma}; the reconstructed value will have very large variance.")
        if args.skip_high_gamma:
            metrics["status"] = "skipped_gamma"
            finalize_timings(metrics, timings, run_start)
            return metrics, artifacts

    subexperiments, coefficients, total_subexp = stage_generate_experiments(subcircuits, subobservables, log10_terms, args, timings, metrics, artifacts, tag)
    results = stage_execute(subexperiments, total_subexp, args, total_cpus, timings, metrics, tag)
    stage_reconstruct(results, coefficients, subobservables, timings, metrics, artifacts, tag)

    metrics["cutting_expval"] = metrics.get("reconstructed_expval")

    if args.run_actual and metrics["actual_expval"] is not None:
        metrics["absolute_error"] = abs(
            metrics["cutting_expval"] - metrics["actual_expval"]
        )
        if abs(metrics["actual_expval"]) > 1e-15:
            metrics["relative_error_percent"] = (
                metrics["absolute_error"] / abs(metrics["actual_expval"])
            ) * 100.0
        else:
            metrics["relative_error_percent"] = None

        log(
            f"{tag}comparison: actual={metrics['actual_expval']:.6g}, "
            f"cutting={metrics['cutting_expval']:.6g}, "
            f"abs_error={metrics['absolute_error']:.6g}"
        )

    finalize_timings(metrics, timings, run_start)
    log(f"{tag}{'total':<26}: {timings['total']:10.3f} s")
    metrics["status"] = "ok"

    artifacts["timings_keys"] = np.array(list(timings.keys()))
    artifacts["timings_values"] = np.array(list(timings.values()), dtype=float)
    return metrics, artifacts


# ============================================================
# SAVING
# ============================================================

def save_metrics(out_dir, all_metrics):
    if not all_metrics:
        return

    fieldnames = []
    for row in all_metrics:
        fieldnames.extend(key for key in row if key not in fieldnames)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    arr = np.empty(len(all_metrics), dtype=object)
    for i, row in enumerate(all_metrics):
        arr[i] = row
    np.save(os.path.join(out_dir, "metrics.npy"), arr, allow_pickle=True)


def save_artifacts(out_dir, n_data, artifacts):
    if artifacts:
        np.savez_compressed(os.path.join(out_dir, "artifacts", f"n{n_data}.npz"), **artifacts)


def create_output_dir(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"run_{stamp}")
    os.makedirs(os.path.join(out_dir, "artifacts"), exist_ok=True)
    return out_dir


def save_config(out_dir, args, total_cpus, ray_nodes, ray_init_time):
    config = vars(args).copy()
    config.update({"ray_address": os.environ.get("RAY_ADDRESS"), "ray_nodes": ray_nodes, "ray_total_cpus": total_cpus, "ray_init_time": ray_init_time})
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_bool(value):
    """Parse a command-line boolean such as true/false."""
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Use true or false."
    )


def parse_arguments():
    p = argparse.ArgumentParser(description="Distributed Qiskit circuit cutting sweep")

    p.add_argument(
        "--run-actual",
        type=parse_bool,
        default=False,
        help="Run the complete uncut circuit for a reference value. Use true or false.",
    )
    p.add_argument("--min-qubits", type=int, default=16)
    p.add_argument("--max-qubits", type=int, default=20)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--qubits-per-subcircuit", type=int, default=10)
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--enumerate-below", type=int, default=10000)
    p.add_argument("--max-gamma", type=float, default=1000)
    p.add_argument("--skip-high-gamma", action="store_true")
    p.add_argument("--shots", type=int, default=2 ** 14)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--gurobi-time-limit", type=float, default=300)
    p.add_argument("--gurobi-mip-gap", type=float, default=0.0)
    p.add_argument("--task-multiplier", type=int, default=4, help="Target Ray tasks per CPU.")
    p.add_argument("--output-dir", type=str, default="results")

    args = p.parse_args()

    if args.min_qubits < 3:
        p.error("--min-qubits must be >= 3")
    if args.max_qubits < args.min_qubits:
        p.error("--max-qubits must be >= --min-qubits")
    if args.step < 1:
        p.error("--step must be >= 1")
    if args.qubits_per_subcircuit < 2:
        p.error("--qubits-per-subcircuit must be >= 2")
    return args


# ============================================================
# SUMMARY
# ============================================================

def format_time(metrics, key):
    value = metrics.get(key)
    return f"{'-':>8}" if value is None else f"{value:8.2f}"


def print_summary(all_metrics, total_time, out_dir):
    print()
    print("=" * SEP_WIDTH)
    print("SUMMARY")
    print("=" * SEP_WIDTH)
    print(
        f"{'N':>4} {'status':>12} {'actual':>10} {'cutting':>10} "
        f"{'abs_err':>10} {'subckt':>7} {'cuts':>6} {'log10g':>7} "
        f"{'subexp':>8} {'ray':>8} {'total':>8}"
    )

    for m in all_metrics:
        if m.get("status") not in OK_STATUSES:
            print(f"{m['n_qubits']:>4} {m.get('status', '?'):>12} {m.get('error', '')}")
            continue

        actual = m.get("actual_expval")
        cutting = m.get("cutting_expval")
        abs_err = m.get("absolute_error")

        print(
            f"{m['n_qubits']:>4} {m['status']:>12} "
            f"{actual if actual is not None else float('nan'):>10.5f} "
            f"{cutting if cutting is not None else float('nan'):>10.5f} "
            f"{abs_err if abs_err is not None else float('nan'):>10.5f} "
            f"{m['n_subcircuits']:>7} {m['n_cuts']:>6} {m['log10_gamma']:>7.1f} "
            f"{m.get('total_subexperiments', 0):>8} "
            f"{format_time(m, 't_ray_execution')} {format_time(m, 't_total')}"
        )

    print("=" * SEP_WIDTH)
    print(f"Whole sweep time : {total_time:.2f} s")
    print(f"Results saved in : {out_dir}")
    print("  metrics.csv / metrics.npy / artifacts/n<N>.npz / config.json")
    print("=" * SEP_WIDTH)


# ============================================================
# MAIN
# ============================================================

def make_gurobi_env():
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def run_sweep(qubit_values, args, gurobi_env, total_cpus, ray_init_time, out_dir):
    all_metrics = []
    for n_data in qubit_values:
        try:
            metrics, artifacts = run_single(n_data, args, gurobi_env, total_cpus)
            metrics["ray_init_time"] = ray_init_time
            save_artifacts(out_dir, n_data, artifacts)
        except Exception as exc:
            traceback.print_exc()
            log(f"[N={n_data}] FAILED: {exc}")
            metrics = {"n_qubits": n_data, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        all_metrics.append(metrics)
        save_metrics(out_dir, all_metrics)  # incremental saving
    return all_metrics


def main():
    args = parse_arguments()
    sweep_start = time.perf_counter()
    
    print("Into Code")

    total_cpus, ray_nodes, ray_init_time = connect_ray()
    
    print("ray connect complete")

    qubit_values = list(range(args.min_qubits, args.max_qubits + 1, args.step))
    out_dir = create_output_dir(args)
    save_config(out_dir, args, total_cpus, ray_nodes, ray_init_time)

    log(f"Qubit sweep           : {qubit_values}")
    log(f"Run actual (uncut)    : {args.run_actual}")
    log(f"Qubits/subcircuit     : {args.qubits_per_subcircuit}")
    log(f"Monte Carlo samples   : {args.samples}")
    log(f"Output directory      : {out_dir}")

    gurobi_env = make_gurobi_env()
    try:
        all_metrics = run_sweep(qubit_values, args, gurobi_env, total_cpus, ray_init_time, out_dir)
    finally:
        gurobi_env.dispose()

    print_summary(all_metrics, time.perf_counter() - sweep_start, out_dir)


if __name__ == "__main__":
    print("Inside")
    main()