"""
GHZ circuit benchmark with and without circuit cutting: time, accuracy and sampling-overhead comparison.
Examples
--------
# 20-qubit GHZ, subcircuits of at most 5 qubits
python ghz_benchmark.py --algo-qubits 20 --max-qubits-per-subcircuit 5

# 40-qubit GHZ, subcircuits of at most 8 qubits, benchmark overhead
python ghz_benchmark.py --algo-qubits 40 --max-qubits-per-subcircuit 8 --benchmark

# 50-qubit GHZ with joint gate + wire cutting
python ghz_benchmark.py --algo-qubits 50 --max-qubits-per-subcircuit 8 --cut-strategy joint --benchmark

Notes
-----
* --algo-qubits defines the size of the original GHZ circuit.
* The GHZ circuit is:
      H(0)
      CX(0,1)
      CX(1,2)
      ...
      CX(n-2,n-1)
* The ideal GHZ state is:
      (|00...0> + |11...1>) / sqrt(2)
* The benchmark evaluates GHZ stabilizer observables.
* Z0 ZN-1 has ideal expectation value +1.
* X0 X1 ... XN-1 has ideal expectation value +1.
* Circuit cutting reconstructs the requested observables using CircuitCutting.
* --max-qubits-per-subcircuit controls the maximum number of qubits in one
  subcircuit and is independent of --algo-qubits.
* --cut-strategy selects how cuts are placed: "gate" or "joint".
* The sampling-overhead benchmark compares the variance of the cut estimator
  against the variance of the uncut shot-sampled estimator at the same shot budget.
"""

import argparse
import json
import time
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from circuit_cutting.distribute import RayAgent
from circuit_cutting.cutting import CircuitCutting

# ============================================================
# GHZ observables
# ============================================================
def ghz_z_correlation(n_qubits):
    """Return Z0 Z(n-1)."""
    label = ["I"] * n_qubits
    label[0] = "Z"
    label[n_qubits - 1] = "Z"
    return "".join(label)

def ghz_x_correlation(n_qubits):
    """Return X0 X1 ... X(n-1)."""
    return "X" * n_qubits

def ghz_observables(n_qubits):
    """Return the main GHZ stabilizer observables."""
    return [ghz_z_correlation(n_qubits), ghz_x_correlation(n_qubits)]

def ghz_witness(n_qubits):
    """Return W = -(Z0Z(n-1) + X0X1...X(n-1)) / 2."""
    labels = ghz_observables(n_qubits)
    coeffs = [-0.5, -0.5]
    return SparsePauliOp.from_list(list(zip(labels, coeffs)))

# ============================================================
# Problem
# ============================================================
@dataclass
class Problem:
    n_qubits: int
    op: SparsePauliOp
    def __post_init__(self):
        self.labels = list(self.op.paulis.to_labels())
        self.coeffs = np.array(self.op.coeffs.real)

# ============================================================
# GHZ circuit
# ============================================================
def ghz_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc

# ============================================================
# Helpers
# ============================================================

def label_to_qubit_ops(label):
    """Qiskit labels are little-endian: the RIGHTMOST character acts on qubit 0."""
    n = len(label)
    return [(q, label[n - 1 - q]) for q in range(n)]

def pauli_weight(label):
    return sum(c != "I" for c in label)

def is_identity(label):
    return set(label) == {"I"}

def scalar_metrics(metrics):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (bool, int, float, np.integer, np.floating)):
            out[str(k)] = float(v)
    return out

def fmt_time(s):
    if s < 1e-3:
        return f"{s * 1e6:.1f} us"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    if s < 120:
        return f"{s:.2f} s"
    return f"{s / 60:.1f} min"

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))

# ============================================================
# 1) Exact statevector
# ============================================================
def expectation_exact(pauli, prob):
    sv = Statevector(ghz_circuit(prob.n_qubits))
    return float(sv.expectation_value(SparsePauliOp(pauli)).real)

def witness_exact(prob):
    sv = Statevector(ghz_circuit(prob.n_qubits))
    return float(sv.expectation_value(prob.op).real)

def per_term_exact(prob):
    sv = Statevector(ghz_circuit(prob.n_qubits))
    return [float(sv.expectation_value(SparsePauliOp(l)).real) for l in prob.labels]

# ============================================================
# 2) Uncut but shot-sampled
# ============================================================
def measurement_circuit(pauli, prob):
    """GHZ circuit + basis changes so that <pauli> = <Z...Z>."""
    qc = ghz_circuit(prob.n_qubits)
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
    return qc, z_qubits

class SampledEvaluator:
    def __init__(self, prob, shots, rng):
        self.prob, self.shots, self.rng = prob, shots, rng

    def expectation(self, pauli, shots=None):
        shots = shots or self.shots
        qc, z_qubits = measurement_circuit(pauli, self.prob)
        if not z_qubits:
            return 1.0
        probs = Statevector(qc).probabilities()
        probs = probs / probs.sum()
        counts = self.rng.multinomial(shots, probs)
        mask = sum(1 << q for q in z_qubits)
        signs = np.array([1 - 2 * (bin(i & mask).count("1") & 1) for i in range(len(probs))])
        return float(np.dot(counts, signs) / shots)

    def energy(self):
        return float(sum(c * self.expectation(p) for p, c in zip(self.prob.labels, self.prob.coeffs)))

# ============================================================
# 3) Circuit cutting
# ============================================================
class CutEvaluator:
    def __init__(self, prob, agent, qubits_per_subcircuit, shots, samples, partitioner="gurobi",
                 optimization_level=1, run_actual=False, cut_strategy="gate", joint_opts=None, seed=0):
        self.prob, self.agent = prob, agent
        self.cut_strategy = cut_strategy
        self.joint_opts = dict(joint_opts or {})
        self.seed = seed
        self.plan_metrics = None
        self.qps, self.shots, self.samples = qubits_per_subcircuit, shots, samples
        self.partitioner, self.opt_level = partitioner, optimization_level
        self.run_actual = run_actual
        self.n_jobs = 0
        self.total_time = 0.0
        self.term_time = {}
        self.term_jobs = {}
        self.last_metrics = None
        self._printed_keys = False

    def expectations(self, paulis, shots=None, samples=None):
        """
        ONE circuit-cutting job that reconstructs every non-identity Pauli in paulis.
        """
        paulis = list(dict.fromkeys(paulis))
        out = {p: 1.0 for p in paulis if is_identity(p)}
        targets = [p for p in paulis if not is_identity(p)]
        if not targets:
            return out
        qc = ghz_circuit(self.prob.n_qubits)
        t0 = time.perf_counter()
        with CircuitCutting(
            qc,
            agent=self.agent,
            observable=targets,
            qubits_per_subcircuit=self.qps,
            cut_strategy=self.cut_strategy,
            partitioner=self.partitioner,
            seed=self.seed,
            **self.joint_opts,
            shots=shots or self.shots,
            samples=samples or self.samples,
            run_actual=self.run_actual,
            optimization_level=self.opt_level,
            save=False,
            verbose=False,
        ) as cc:
            metrics, artifacts = cc.run()
        dt = time.perf_counter() - t0

        self.n_jobs += 1
        self.total_time += dt
        for p in targets:
            self.term_time[p] = self.term_time.get(p, 0.0) + dt / len(targets)
            self.term_jobs[p] = self.term_jobs.get(p, 0) + 1
        self.last_metrics = scalar_metrics(metrics)

        plan_keys = ("n_fragments", "n_cuts", "log10_gamma", "max_subcircuit_qubits",
                     "partition_cut_gates", "partition_wire_cuts", "joint_gate_cuts",
                     "joint_wire_cuts", "joint_n_groups", "joint_grouped_gates",
                     "joint_est_log10_kappa", "joint_indiv_log10_kappa",
                     "total_subexperiments", "t_partition")

        self.plan_metrics = {k: metrics[k] for k in plan_keys if metrics.get(k) is not None}
        self.plan_metrics["cut_strategy"] = self.cut_strategy

        if not self._printed_keys:
            print("[cut] scalar metrics exposed by circuit_cut:", sorted(self.last_metrics))
            self._printed_keys = True

        values = json.loads(metrics["cutting_expvals"])
        out.update(dict(zip(targets, values)))
        return out

    def expectation(self, pauli, shots=None, samples=None):
        """Single-term cutting job used by the sampling-overhead benchmark."""
        return float(self.expectations([pauli], shots=shots, samples=samples)[pauli])
    
    def energy_with_terms(self):
        exp = self.expectations(self.prob.labels)
        vals = [exp[p] for p in self.prob.labels]
        return float(np.dot(self.prob.coeffs, vals)), vals

    def energy(self):
        return self.energy_with_terms()[0]
    
# ============================================================
# Sampling-overhead benchmark
# ============================================================
def benchmark_sampling_overhead(prob, cutter, sampler, paulis, configs, repeats):
    """
    For each Pauli term and each (shots, samples) setting, repeat the estimate with
    circuit cutting and without cutting at the same shot budget.
    """
    rows = []
    for pauli in paulis:
        exact = expectation_exact(pauli, prob)
        for shots, samples in configs:
            cut_vals, cut_t = [], []
            unc_vals, unc_t = [], []
            lib = {}
            for _ in range(repeats):
                t0 = time.perf_counter()
                cut_vals.append(cutter.expectation(pauli, shots=shots, samples=samples))
                cut_t.append(time.perf_counter() - t0)
                lib = cutter.last_metrics or lib
                t0 = time.perf_counter()
                unc_vals.append(sampler.expectation(pauli, shots=shots))
                unc_t.append(time.perf_counter() - t0)
            ddof = 1 if repeats > 1 else 0
            cut_std = float(np.std(cut_vals, ddof=ddof))
            unc_std = float(np.std(unc_vals, ddof=ddof))
            var_theory = max(1.0 - exact ** 2, 1e-12) / shots
            row = {
                "pauli": pauli,
                "weight": pauli_weight(pauli),
                "shots": shots,
                "samples": samples,
                "repeats": repeats,
                "exact": exact,
                "cut_mean": float(np.mean(cut_vals)),
                "cut_bias": float(np.mean(cut_vals) - exact),
                "cut_std": cut_std,
                "uncut_std_emp": unc_std,
                "uncut_std_theory": float(np.sqrt(var_theory)),
                "overhead_var": cut_std ** 2 / var_theory,
                "overhead_var_emp": (cut_std ** 2 / unc_std ** 2) if unc_std > 0 else float("nan"),
                "cut_time": float(np.mean(cut_t)),
                "uncut_time": float(np.mean(unc_t)),
                "time_ratio": float(np.mean(cut_t) / max(np.mean(unc_t), 1e-12)),
                "lib_metrics": lib,
            }
            rows.append(row)
            print(f"[bench] {pauli} shots={shots:<7d} samples={samples:<6d} "
                  f"std_cut={cut_std:.2e} std_uncut={np.sqrt(var_theory):.2e} "
                  f"overhead={row['overhead_var']:.1f}x t_cut={fmt_time(row['cut_time'])}")
    return rows
# ============================================================
# Plotting
# ============================================================
def make_plots(d, outfile, show):
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    # --- Panel 1: exact vs cut expectation values ---
    ax = axes[0, 0]
    terms = np.array(d["observables"])
    x = np.arange(len(terms))
    w = 0.4
    ax.bar(x - w / 2, d["per_term_exact"], w, label="Exact")
    ax.bar(x + w / 2, d["per_term_cut"], w, label="Circuit cutting")
    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=75, fontsize=7)
    ax.set_ylabel("<P>")
    ax.set_title(f"GHZ observables ({d['n_qubits']} qubits)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # --- Panel 2: observable error ---
    ax = axes[0, 1]
    errors = np.abs(np.array(d["per_term_cut"]) - np.array(d["per_term_exact"]))
    ax.bar(x, errors)
    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=75, fontsize=7)
    ax.set_ylabel("|<P>cut - <P>exact|")
    ax.set_title("Circuit-cutting observable error")
    ax.grid(alpha=0.3, axis="y")
    # --- Panel 3: GHZ witness ---
    ax = axes[0, 2]
    names = ["Exact", "Uncut sampled", "Circuit cutting"]
    vals = [d["witness_exact"], d["witness_sampled"], d["witness_cut"]]
    ax.bar(names, vals)
    ax.axhline(-1.0, ls=":", label="Ideal GHZ witness")
    ax.set_ylabel("Witness value")
    ax.set_title("GHZ witness")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # --- Panel 4: total wall time ---
    ax = axes[1, 0]
    names = ["Exact\nstatevector", "Sampled\nuncut", "Circuit\ncutting"]
    vals = [d["time"]["exact_wall"], d["time"]["sampled_wall"], d["time"]["cut_wall"]]
    bars = ax.bar(names, vals)
    ax.set_yscale("log")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("Execution time")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt_time(v), ha="center", va="bottom")
    ax.grid(alpha=0.3, axis="y", which="both")
    # --- Panel 5: per-observable time ---
    ax = axes[1, 1]
    ax.bar(["Exact", "Sampled", "Cutting"], [
        d["time"]["exact_eval"],
        d["time"]["sampled_eval"],
        d["time"]["cut_eval"],
    ])
    ax.set_yscale("log")
    ax.set_ylabel("Time per observable (s)")
    ax.set_title("Observable evaluation time")
    ax.grid(alpha=0.3, axis="y", which="both")

    # --- Panel 6: sampling overhead ---
    ax = axes[1, 2]
    rows = [r for r in d["benchmark"] if r["samples"] == d["samples"]]

    if rows:
        for p in sorted({r["pauli"] for r in rows}):
            rr = sorted([r for r in rows if r["pauli"] == p], key=lambda r: r["shots"])
            ax.loglog(
                [r["shots"] for r in rr],
                [max(r["overhead_var"], 1e-3) for r in rr],
                "o-",
                label=p,
            )
        ax.axhline(1, ls=":", label="No overhead")
        ax.set_xlabel("Shots per subcircuit setting")
        ax.set_ylabel("Var(cut) / Var(uncut)")
        ax.set_title(f"Sampling overhead (samples={d['samples']})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Run with --benchmark to\nshow the sampling-overhead panel",
                ha="center", va="center", fontsize=12)

    fig.suptitle(f"GHZ circuit benchmark ({d['n_qubits']} qubits)", fontsize=16)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"Saved plot to {outfile}")
    if show:
        plt.show()

def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="GHZ circuit: exact vs circuit cutting (time, accuracy, sampling overhead)")
    # problem
    ap.add_argument("--algo-qubits", type=int, default=20, help="number of qubits in the full GHZ algorithm")
    ap.add_argument("--qubits", type=int, default=None, help="alias for --algo-qubits")
    # cutting
    ap.add_argument("--max-qubits-per-subcircuit", "--qubits-per-subcircuit", dest="qps", type=int, default=5, help="max qubits available in one subcircuit")
    ap.add_argument("--shots", type=int, default=2 ** 14)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--cut-strategy", type=str, default="gate", choices=["gate", "joint"], help="gate: gate cuts; joint: gate+wire cuts")
    ap.add_argument("--partitioner", type=str, default="gurobi", choices=["gurobi", "kernighan_lin", "spectral", "contiguous"], help="used only with --cut-strategy gate")
    ap.add_argument("--no-wire-cuts", action="store_true", help="(joint) disable wire cuts")
    ap.add_argument("--no-gate-groups", action="store_true", help="(joint) disable gate-group detection")
    ap.add_argument("--kl-runs", type=int, default=10, help="(joint) stage-1 random restarts")
    ap.add_argument("--kl-max-passes", type=int, default=10, help="(joint) max KL passes per run")
    ap.add_argument("--kl-patience", type=int, default=15, help="(joint) non-improving moves before a pass stops")
    ap.add_argument("--optimization-level", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=48)
    ap.add_argument("--cut-run-actual", action="store_true", help="also run the uncut reference circuit inside every cutting job")
    # benchmark
    ap.add_argument("--benchmark", action="store_true", help="run the sampling-overhead benchmark")
    ap.add_argument("--bench-terms", type=int, default=2, help="number of GHZ observables to benchmark")
    ap.add_argument("--bench-repeats", type=int, default=5, help="repeats per setting")
    ap.add_argument("--bench-shots", type=parse_int_list, default=[1024, 4096, 16384], help="shots sweep")
    ap.add_argument("--bench-samples", type=parse_int_list, default=[1000, 10000], help="samples sweep")
    # output
    ap.add_argument("--plot", type=str, default="ghz_comparison.png")
    ap.add_argument("--data", type=str, default="ghz_results.json")
    ap.add_argument("--no-show", action="store_true", help="do not open the matplotlib window")
    args = ap.parse_args()

    if args.qubits is not None:
        args.algo_qubits = args.qubits

    if args.algo_qubits < 2:
        ap.error("--algo-qubits must be >= 2")

    if args.qps < 1:
        ap.error("--max-qubits-per-subcircuit must be >= 1")

    if args.qps >= args.algo_qubits:
        print(f"WARNING: subcircuit size {args.qps} >= {args.algo_qubits} qubits -> "
              f"nothing needs to be cut.")

    # ---- Build problem ----
    op = ghz_witness(args.algo_qubits)
    prob = Problem(args.algo_qubits, op)

    print(f"Algorithm    : GHZ")
    print(f"Algo qubits  : {prob.n_qubits}")
    print(f"Subcircuit   : <= {args.qps} qubits")
    print(f"GHZ gates    : {prob.n_qubits}")
    print(f"Observables  : {len(prob.labels)}")
    print(f"Shots        : {args.shots}")
    print(f"Samples      : {args.samples}")
    print(f"Cut strategy : {args.cut_strategy}")

    # ---- Exact reference ----
    t0 = time.perf_counter()
    per_term_ex = per_term_exact(prob)
    witness_ex = witness_exact(prob)
    exact_wall = time.perf_counter() - t0
    print(f"Exact GHZ witness = {witness_ex:.8f}")

    # ---- Uncut shot-sampled baseline ----
    sampler = SampledEvaluator(prob, args.shots, np.random.default_rng(args.algo_qubits + 1))
    t0 = time.perf_counter()
    sampled_terms = [sampler.expectation(p) for p in prob.labels]
    witness_sampled = float(np.dot(prob.coeffs, sampled_terms))
    sampled_wall = time.perf_counter() - t0
    print(f"Sampled GHZ witness = {witness_sampled:.8f}")

    # ---- Circuit cutting ----
    t_init = time.perf_counter()
    agent = RayAgent(num_cpus_per_node=args.cpus)
    agent.initialise()
    ray_init_time = time.perf_counter() - t_init
    print(f"Ray initialisation took {fmt_time(ray_init_time)} (excluded from benchmark)")

    joint_opts = dict(use_wire_cuts=not args.no_wire_cuts,
                      use_gate_groups=not args.no_gate_groups,
                      kl_runs=args.kl_runs,
                      kl_max_passes=args.kl_max_passes,
                      kl_patience=args.kl_patience)

    cutter = CutEvaluator(prob, agent, args.qps, args.shots, args.samples,
                          partitioner=args.partitioner,
                          optimization_level=args.optimization_level,
                          run_actual=args.cut_run_actual,
                          cut_strategy=args.cut_strategy,
                          joint_opts=joint_opts if args.cut_strategy == "joint" else None,
                          seed=args.algo_qubits)
    benchmark_rows = []

    try:
        # ---- Full cutting evaluation ----
        t0 = time.perf_counter()
        E_cut, per_term_cut = cutter.energy_with_terms()
        cut_wall = time.perf_counter() - t0
        witness_cut = E_cut
        print(f"Circuit-cut GHZ witness = {witness_cut:.8f}")

        # ---- Sampling-overhead benchmark ----
        if args.benchmark:
            cand = [(pauli_weight(p), p) for p in prob.labels if pauli_weight(p) > 0]
            cand.sort(reverse=True)
            bench_paulis = [p for _, p in cand[:args.bench_terms]]
            configs = [(s, args.samples) for s in args.bench_shots]
            configs += [(args.shots, m) for m in args.bench_samples]
            configs = list(dict.fromkeys(configs))
            print(f"\nBenchmarking sampling overhead on {bench_paulis} with configs {configs}")
            benchmark_rows = benchmark_sampling_overhead(prob, cutter, sampler, bench_paulis, configs, args.bench_repeats,)
    finally:
        agent.stop_ray_clusters()

    # ---- Accuracy ----
    pt_err = np.abs(np.array(per_term_cut) - np.array(per_term_ex))
    pt_rms = float(np.sqrt(np.mean(pt_err ** 2)))
    pt_max = float(pt_err.max())
    witness_error = abs(witness_cut - witness_ex)

    # ---- Summary ----
    print("\n" + "=" * 100)
    print(f"GHZ RESULT ({prob.n_qubits} qubits, {len(prob.labels)} observables, "
          f"subcircuit <= {args.qps} qubits, shots={args.shots}, samples={args.samples})")
    print("=" * 100)

    print(f"Exact GHZ witness       : {witness_ex:.8f}")
    print(f"Uncut sampled witness   : {witness_sampled:.8f}")
    print(f"Circuit-cut witness     : {witness_cut:.8f}")
    print(f"Cut witness error       : {witness_error:.3e}")

    print("\nMETHOD")
    print(f"{'method':24s}{'wall time':>15s}{'witness':>18s}{'|error|':>15s}")
    print(f"{'Exact statevector':24s}{fmt_time(exact_wall):>15s}{witness_ex:>18.8f}{0.0:>15.3e}")
    print(f"{'Uncut sampled':24s}{fmt_time(sampled_wall):>15s}{witness_sampled:>18.8f}"
          f"{abs(witness_sampled - witness_ex):>15.3e}")
    print(f"{'Circuit cutting':24s}{fmt_time(cut_wall):>15s}{witness_cut:>18.8f}"
          f"{witness_error:>15.3e}")

    print("\nTIME")
    print(f"  Ray initialisation                 : {fmt_time(ray_init_time)}")
    print(f"  Circuit cutting / exact            : {cut_wall / max(exact_wall, 1e-12):.1f}x")
    print(f"  Circuit cutting / sampled          : {cut_wall / max(sampled_wall, 1e-12):.1f}x")
    print(f"  Cutting jobs                       : {cutter.n_jobs}")
    print(f"  Mean cutting job time              : "
          f"{fmt_time(cutter.total_time / max(cutter.n_jobs, 1))}")

    print(f"\nCUT PLAN (strategy: {args.cut_strategy})")
    for k, v in (cutter.plan_metrics or {}).items():
        print(f"  {k:30s}: {v:.4g}" if isinstance(v, float) else f"  {k:30s}: {v}")
    if args.cut_strategy == "joint" and cutter.plan_metrics:
        print("  (joint estimates depend on the selected gate-group configuration)")
    print("\nACCURACY")
    print(f"  Per-term |<P>cut - <P>exact| : RMS {pt_rms:.3e}, max {pt_max:.3e}")

    for p, exact, cut in zip(prob.labels, per_term_ex, per_term_cut):
        print(f"  {p} exact={exact:+.8f} cut={cut:+.8f} error={abs(cut - exact):.3e}")

    # ---- Sampling overhead ----
    if benchmark_rows:
        print("\nSAMPLING OVERHEAD (Var(cut)/Var(uncut) at equal shots)")
        print(f"{'pauli':>{max(6, prob.n_qubits)}s} {'shots':>8s} {'samples':>9s} "
              f"{'<P>exact':>10s} {'bias':>12s} {'std_cut':>12s} {'std_uncut':>12s} "
              f"{'overhead':>10s} {'t_cut':>10s} {'t_uncut':>10s}")

        for r in benchmark_rows:
            print(f"{r['pauli']:>{max(6, prob.n_qubits)}s} "
                  f"{r['shots']:8d} {r['samples']:9d} {r['exact']:+10.4f} "
                  f"{r['cut_bias']:+12.2e} {r['cut_std']:12.2e} "
                  f"{r['uncut_std_theory']:12.2e} {r['overhead_var']:9.1f}x "
                  f"{fmt_time(r['cut_time']):>10s} {fmt_time(r['uncut_time']):>10s}")
        print(f"(std estimated from {args.bench_repeats} repeats)")

    data = {
        "args": vars(args),
        "algorithm": "GHZ",
        "n_qubits": prob.n_qubits,
        "n_gates": prob.n_qubits,
        "observables": prob.labels,
        "coeffs": prob.coeffs.tolist(),
        "samples": args.samples,
        "shots": args.shots,
        "witness_exact": witness_ex,
        "witness_sampled": witness_sampled,
        "witness_cut": witness_cut,
        "witness_error": witness_error,
        "per_term_exact": per_term_ex,
        "per_term_cut": per_term_cut,
        "per_term_error_rms": pt_rms,
        "per_term_error_max": pt_max,
        "per_term_cut_time_mean": {p: cutter.term_time[p] / cutter.term_jobs[p] for p in cutter.term_time},
        "time": {
            "ray_init": ray_init_time,
            "exact_wall": exact_wall,
            "sampled_wall": sampled_wall,
            "cut_wall": cut_wall,
            "cut_jobs": cutter.n_jobs,
            "exact_eval": exact_wall / max(len(prob.labels), 1),
            "sampled_eval": sampled_wall / max(len(prob.labels), 1),
            "cut_eval": cut_wall / max(len(prob.labels), 1),
            "slowdown_total": cut_wall / max(exact_wall, 1e-12),
        },
        "cut_plan": cutter.plan_metrics,
        "benchmark": benchmark_rows,
    }

    with open(args.data, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)

    print(f"Saved data to {args.data}")
    make_plots(data, args.plot, show=not args.no_show)

if __name__ == "__main__":
    main()