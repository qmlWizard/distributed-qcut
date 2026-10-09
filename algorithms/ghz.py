"""
GHZ circuit benchmark with and without circuit cutting: time, accuracy and sampling-overhead comparison.

Examples
--------
# 12-qubit GHZ (chain of CNOTs), subcircuits of at most 6 qubits
python ghz_benchmark.py --algo-qubits 12 --max-qubits-per-subcircuit 6

# 20-qubit GHZ with a star topology, benchmark overhead
python ghz_benchmark.py --algo-qubits 20 --max-qubits-per-subcircuit 10 --topology star --benchmark

# joint gate + wire cutting
python ghz_benchmark.py --algo-qubits 16 --max-qubits-per-subcircuit 8 --cut-strategy joint --benchmark

Notes
-----
* Circuit: H on qubit 0, then a CNOT fan-out that spreads the superposition:
      chain : CX(i, i+1)            i = 0..n-2   (default, depth O(n))
      star  : CX(0, i)              i = 1..n-1
      tree  : CX((i-1)//2, i)       i = 1..n-1   (binary tree, depth O(log n))
  Result: |GHZ> = (|0...0> + |1...1>) / sqrt(2). Only 1- and 2-qubit gates (transpiled to {u, cx}).
* Exact reference is ANALYTIC (works for any n). For a Pauli string P on the GHZ state:
      - only I/Z, even number of Z  -> +1      (odd number of Z -> 0)
      - X/Y on EVERY qubit, b Y's   -> cos(b*pi/2)  (b even: (-1)^(b/2), b odd: 0)
      - anything else               -> 0
* Cutting reconstructs *Pauli expectation values*, not state fidelities. The GHZ fidelity
  F = <GHZ|rho|GHZ> is therefore lower-bounded through a stabilizer witness:
      F = (P_0 + P_1)/2 + <X^{(x)n}>/2,
      P_0 + P_1 >= 1 - sum_i (1 - <Z_i Z_{i+1}>)/2        (union bound)
      =>  F >= LB = (1/2)(1 - (n-1)/2) + (1/4) sum_i <Z_i Z_{i+1}> + (1/2) <X^{(x)n}>
  LB is the "GHZ witness" benchmarked here (ideal value = 1). Extra observables Z0, Z0 Z(n-1) and
  Z^{(x)n} are evaluated as additional correlators.
* If n <= --sv-max, the real circuit is also simulated with a statevector (verification of the circuit
  against the analytic formulas, and the uncut sampled baseline uses the real circuit's expectations).
  Above that, the uncut sampled baseline draws binomial samples from the analytic expectations.
* The sampling-overhead benchmark compares the variance of the cut estimator against the variance of the
  uncut shot-sampled estimator at the same shot budget.
* WARNING: the cutting overhead depends strongly on the topology: a chain is cheap to cut
  (O(1) cuts per boundary), a star needs many gate cuts through qubit 0.
"""

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Pauli

from circuit_cutting.distribute import RayAgent
from circuit_cutting.cutting import CircuitCutting


# ============================================================
# Problem
# ============================================================
@dataclass
class Problem:
    n_qubits: int
    topology: str = "chain"
    labels: list = field(default_factory=list)
    coeffs: np.ndarray = None
    offset: float = 0.0  # identity coefficient of the witness

    def __post_init__(self):
        n = self.n_qubits
        labels, coeffs = [], []
        # Witness LB = (1/2)(1 - (n-1)/2) + (1/4) sum_i <Z_i Z_{i+1}> + (1/2) <X^n>
        self.offset = 0.5 * (1.0 - (n - 1) / 2.0)
        for q in range(n - 1):
            labels.append(pauli_string(n, {q: "Z", q + 1: "Z"}))
            coeffs.append(0.25)
        labels.append("X" * n)
        coeffs.append(0.5)
        # Extra (non-witness) correlators
        extras = (pauli_string(n, {0: "Z"}),
                  pauli_string(n, {0: "Z", n - 1: "Z"}),
                  "Z" * n)
        for extra in extras:
            if extra not in labels:
                labels.append(extra)
                coeffs.append(0.0)
        self.labels = labels
        self.coeffs = np.array(coeffs)


# ============================================================
# GHZ circuit
# ============================================================
def ghz_circuit(n_qubits, topology="chain", opt_level=1):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    if topology == "chain":
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    elif topology == "star":
        for i in range(1, n_qubits):
            qc.cx(0, i)
    elif topology == "tree":
        for i in range(1, n_qubits):
            qc.cx((i - 1) // 2, i)
    else:
        raise ValueError(f"unknown topology {topology}")
    return transpile(qc, basis_gates=["u", "cx"], optimization_level=opt_level, seed_transpiler=0)


def circuit_stats(qc):
    ops = qc.count_ops()
    return {"depth": int(qc.depth()), "size": int(qc.size()), "cx": int(ops.get("cx", 0)),
            "n_2q_gates": int(sum(1 for i in qc.data if len(i.qubits) == 2))}


# ============================================================
# Helpers
# ============================================================
def pauli_string(n_qubits, ops):
    """ops: {qubit: 'X'|'Y'|'Z'}. Qiskit little-endian label (rightmost char = qubit 0)."""
    label = ["I"] * n_qubits
    for q, p in ops.items():
        label[n_qubits - 1 - q] = p
    return "".join(label)


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
# 1) Exact (analytic) reference
# ============================================================
def expectation_exact(pauli, prob=None):
    """<GHZ|P|GHZ> for a Pauli string (see module docstring)."""
    if is_identity(pauli):
        return 1.0
    chars = set(pauli)
    if chars <= {"I", "Z"}:
        return 1.0 if pauli.count("Z") % 2 == 0 else 0.0
    if "I" not in chars and "Z" not in chars:       # X/Y on every qubit
        b = pauli.count("Y")
        return 0.0 if b % 2 else float((-1) ** (b // 2))
    return 0.0


def per_term_exact(prob):
    return [expectation_exact(l) for l in prob.labels]


def witness_from_terms(prob, vals):
    return float(prob.offset + np.dot(prob.coeffs, vals))


# ============================================================
# 1b) Optional statevector verification (small n)
# ============================================================
class StatevectorRef:
    def __init__(self, qc, prob):
        self.prob = prob
        t0 = time.perf_counter()
        self.sv = Statevector(qc)
        self.build_time = time.perf_counter() - t0
        self._cache = {}

    def expectation(self, pauli):
        if is_identity(pauli):
            return 1.0
        if pauli not in self._cache:
            self._cache[pauli] = float(np.real(self.sv.expectation_value(Pauli(pauli))))
        return self._cache[pauli]

    def fidelity(self):
        n = self.prob.n_qubits
        ghz = np.zeros(2 ** n, dtype=complex)
        ghz[0] = ghz[-1] = 1 / np.sqrt(2)
        return float(np.abs(np.vdot(ghz, self.sv.data)) ** 2)


# ============================================================
# 2) Uncut but shot-sampled
# ============================================================
class SampledEvaluator:
    def __init__(self, prob, shots, rng, sv_ref=None):
        self.prob, self.shots, self.rng, self.sv = prob, shots, rng, sv_ref

    def expectation(self, pauli, shots=None):
        shots = shots or self.shots
        if is_identity(pauli):
            return 1.0
        mu = self.sv.expectation(pauli) if self.sv is not None else expectation_exact(pauli)
        p_plus = min(max((1.0 + mu) / 2.0, 0.0), 1.0)
        n_plus = self.rng.binomial(shots, p_plus)
        return float((2 * n_plus - shots) / shots)

    def terms(self):
        return [self.expectation(p) for p in self.prob.labels]


# ============================================================
# 3) Circuit cutting
# ============================================================
class CutEvaluator:
    def __init__(self, prob, qc, agent, qubits_per_subcircuit, shots, samples, partitioner="gurobi",
                 optimization_level=1, run_actual=False, cut_strategy="gate", joint_opts=None, seed=0):
        self.prob, self.qc, self.agent = prob, qc, agent
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
        """ONE circuit-cutting job that reconstructs every non-identity Pauli in paulis."""
        paulis = list(dict.fromkeys(paulis))
        out = {p: 1.0 for p in paulis if is_identity(p)}
        targets = [p for p in paulis if not is_identity(p)]
        if not targets:
            return out
        t0 = time.perf_counter()
        with CircuitCutting(
            self.qc,
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

    def witness_with_terms(self):
        exp = self.expectations(self.prob.labels)
        vals = [exp[p] for p in self.prob.labels]
        return witness_from_terms(self.prob, vals), vals


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
        exact = expectation_exact(pauli)
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
            # Binomial variance of a +-1 observable; floored at one flipped count so that
            # (near-)deterministic terms (<P> = +-1, typical for GHZ stabilizers) don't give var=0.
            var_theory = max(1.0 - exact ** 2, 1.0 / shots) / shots
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
    ax.bar(x - w / 2, d["per_term_exact"], w, label="Exact (analytic)")
    ax.bar(x + w / 2, d["per_term_cut"], w, label="Circuit cutting")
    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=75, fontsize=7)
    ax.set_ylabel("<P>")
    ax.set_title(f"GHZ observables ({d['n_qubits']} qubits, {d['topology']})")
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

    # --- Panel 3: fidelity lower bound ---
    ax = axes[0, 2]
    names = ["Exact", "Uncut sampled", "Circuit cutting"]
    vals = [d["witness_exact"], d["witness_sampled"], d["witness_cut"]]
    ax.bar(names, vals)
    ax.axhline(d["fidelity_exact"], ls=":", color="k", label=f"True fidelity = {d['fidelity_exact']:.3f}")
    ax.set_ylabel("Lower bound on GHZ fidelity")
    ax.set_title("GHZ witness  LB = (P0+P1 bound + <X^n>)/2")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # --- Panel 4: total wall time ---
    ax = axes[1, 0]
    t = d["time"]
    items = [("Analytic", t["exact_wall"])]
    if t.get("statevector_wall") is not None:
        items.append(("Statevector", t["statevector_wall"]))
    items += [("Sampled\nuncut", t["sampled_wall"]), ("Circuit\ncutting", t["cut_wall"])]
    bars = ax.bar([a for a, _ in items], [max(v, 1e-9) for _, v in items])
    ax.set_yscale("log")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("Execution time")
    for b, (_, v) in zip(bars, items):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 1e-9), fmt_time(v), ha="center", va="bottom")
    ax.grid(alpha=0.3, axis="y", which="both")

    # --- Panel 5: per-observable time ---
    ax = axes[1, 1]
    ax.bar(["Analytic", "Sampled", "Cutting"], [
        max(t["exact_eval"], 1e-9), max(t["sampled_eval"], 1e-9), max(t["cut_eval"], 1e-9)])
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
            ax.loglog([r["shots"] for r in rr],
                      [max(r["overhead_var"], 1e-3) for r in rr], "o-", label=p)
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

    fig.suptitle(f"GHZ circuit benchmark ({d['n_qubits']} qubits, topology={d['topology']})", fontsize=16)
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
    ap.add_argument("--algo-qubits", type=int, default=8, help="number of qubits in the GHZ state")
    ap.add_argument("--qubits", type=int, default=None, help="alias for --algo-qubits")
    ap.add_argument("--topology", type=str, default="chain", choices=["chain", "star", "tree"],
                    help="CNOT fan-out pattern used to build the GHZ state")
    ap.add_argument("--sv-max", type=int, default=20, help="max qubits for statevector verification / real-circuit sampling")
    ap.add_argument("--seed", type=int, default=0)
    # cutting
    ap.add_argument("--max-qubits-per-subcircuit", "--qubits-per-subcircuit", dest="qps", type=int, default=4, help="max qubits available in one subcircuit")
    ap.add_argument("--shots", type=int, default=2 ** 14)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--cut-strategy", type=str, default="gate", choices=["gate", "joint"], help="gate: gate cuts; joint: gate+wire cuts")
    ap.add_argument("--partitioner", type=str, default="gurobi", choices=["gurobi", "kernighan_lin", "spectral", "contiguous"], help="used only with --cut-strategy gate")
    ap.add_argument("--no-wire-cuts", action="store_true", help="(joint) disable wire cuts")
    ap.add_argument("--no-gate-groups", action="store_true", help="(joint) disable gate-group detection")
    ap.add_argument("--kl-runs", type=int, default=10, help="(joint) stage-1 random restarts")
    ap.add_argument("--kl-max-passes", type=int, default=10, help="(joint) max KL passes per run")
    ap.add_argument("--kl-patience", type=int, default=15, help="(joint) non-improving moves before a pass stops")
    ap.add_argument("--optimization-level", type=int, default=1, help="transpiler/optimisation level")
    ap.add_argument("--cpus", type=int, default=48)
    ap.add_argument("--cut-run-actual", action="store_true", help="also run the uncut reference circuit inside every cutting job")
    # benchmark
    ap.add_argument("--benchmark", action="store_true", help="run the sampling-overhead benchmark")
    ap.add_argument("--bench-terms", type=int, default=2, help="number of observables to benchmark")
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
    n = args.algo_qubits
    if n < 2:
        ap.error("--algo-qubits must be >= 2")
    if args.qps < 1:
        ap.error("--max-qubits-per-subcircuit must be >= 1")
    if args.qps >= n:
        print(f"WARNING: subcircuit size {args.qps} >= {n} qubits -> nothing needs to be cut.")

    # ---- Build problem & circuit ----
    prob = Problem(n, args.topology)
    qc = ghz_circuit(n, args.topology, opt_level=args.optimization_level)
    cstats = circuit_stats(qc)
    fid_exact = 1.0

    print(f"Algorithm    : GHZ")
    print(f"Algo qubits  : {n}")
    print(f"Topology     : {args.topology}")
    print(f"Circuit      : depth {cstats['depth']}, {cstats['size']} gates, {cstats['n_2q_gates']} two-qubit gates")
    print(f"Subcircuit   : <= {args.qps} qubits")
    print(f"Observables  : {len(prob.labels)}")
    print(f"Shots        : {args.shots}")
    print(f"Samples      : {args.samples}")
    print(f"Cut strategy : {args.cut_strategy}")

    # ---- Exact reference (analytic) ----
    t0 = time.perf_counter()
    per_term_ex = per_term_exact(prob)
    witness_ex = witness_from_terms(prob, per_term_ex)
    exact_wall = time.perf_counter() - t0
    print(f"Exact GHZ witness (LB on fidelity) = {witness_ex:.8f}")

    # ---- Optional statevector verification ----
    sv_ref, sv_wall, sv_dev, sv_terms = None, None, None, None
    if n <= args.sv_max:
        sv_ref = StatevectorRef(qc, prob)
        t0 = time.perf_counter()
        sv_terms = [sv_ref.expectation(p) for p in prob.labels]
        sv_wall = sv_ref.build_time + (time.perf_counter() - t0)
        sv_dev = float(np.max(np.abs(np.array(sv_terms) - np.array(per_term_ex))))
        print(f"Statevector check: fidelity={sv_ref.fidelity():.8f} (analytic {fid_exact:.8f}), "
              f"max |<P>sv - <P>analytic| = {sv_dev:.2e}")
        if sv_dev > 1e-6:
            print("WARNING: circuit disagrees with the analytic GHZ formulas -- check the circuit construction.")
    else:
        print(f"n={n} > --sv-max={args.sv_max}: skipping statevector; sampled baseline uses analytic expectations.")

    # ---- Uncut shot-sampled baseline ----
    sampler = SampledEvaluator(prob, args.shots, np.random.default_rng(args.seed + n + 1), sv_ref)
    t0 = time.perf_counter()
    sampled_terms = sampler.terms()
    witness_sampled = witness_from_terms(prob, sampled_terms)
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

    cutter = CutEvaluator(prob, qc, agent, args.qps, args.shots, args.samples,
                          partitioner=args.partitioner,
                          optimization_level=args.optimization_level,
                          run_actual=args.cut_run_actual,
                          cut_strategy=args.cut_strategy,
                          joint_opts=joint_opts if args.cut_strategy == "joint" else None,
                          seed=args.seed + n)
    benchmark_rows = []

    try:
        t0 = time.perf_counter()
        witness_cut, per_term_cut = cutter.witness_with_terms()
        cut_wall = time.perf_counter() - t0
        print(f"Circuit-cut GHZ witness = {witness_cut:.8f}")

        if args.benchmark:
            cand = sorted(((pauli_weight(p), p) for p in prob.labels if pauli_weight(p) > 0), reverse=True)
            bench_paulis = [p for _, p in cand[:args.bench_terms]]
            configs = [(s, args.samples) for s in args.bench_shots]
            configs += [(args.shots, m) for m in args.bench_samples]
            configs = list(dict.fromkeys(configs))
            print(f"\nBenchmarking sampling overhead on {bench_paulis} with configs {configs}")
            benchmark_rows = benchmark_sampling_overhead(prob, cutter, sampler, bench_paulis, configs, args.bench_repeats)
    finally:
        agent.stop_ray_clusters()

    # ---- Accuracy ----
    pt_err = np.abs(np.array(per_term_cut) - np.array(per_term_ex))
    pt_rms = float(np.sqrt(np.mean(pt_err ** 2)))
    pt_max = float(pt_err.max())
    witness_error = abs(witness_cut - witness_ex)

    # ---- Summary ----
    print("\n" + "=" * 100)
    print(f"GHZ RESULT ({n} qubits, topology={args.topology}, {len(prob.labels)} observables, "
          f"subcircuit <= {args.qps} qubits, shots={args.shots}, samples={args.samples})")
    print("=" * 100)
    print(f"True fidelity           : {fid_exact:.8f}")
    print(f"Exact witness (LB)      : {witness_ex:.8f}")
    print(f"Uncut sampled witness   : {witness_sampled:.8f}")
    print(f"Circuit-cut witness     : {witness_cut:.8f}")
    print(f"Cut witness error       : {witness_error:.3e}")

    print("\nMETHOD")
    print(f"{'method':24s}{'wall time':>15s}{'witness':>18s}{'|error|':>15s}")
    print(f"{'Analytic':24s}{fmt_time(exact_wall):>15s}{witness_ex:>18.8f}{0.0:>15.3e}")
    if sv_wall is not None:
        print(f"{'Statevector':24s}{fmt_time(sv_wall):>15s}{witness_from_terms(prob, sv_terms):>18.8f}"
              f"{abs(witness_from_terms(prob, sv_terms) - witness_ex):>15.3e}")
    print(f"{'Uncut sampled':24s}{fmt_time(sampled_wall):>15s}{witness_sampled:>18.8f}"
          f"{abs(witness_sampled - witness_ex):>15.3e}")
    print(f"{'Circuit cutting':24s}{fmt_time(cut_wall):>15s}{witness_cut:>18.8f}"
          f"{witness_error:>15.3e}")

    print("\nTIME")
    print(f"  Ray initialisation                 : {fmt_time(ray_init_time)}")
    if sv_wall is not None:
        print(f"  Circuit cutting / statevector      : {cut_wall / max(sv_wall, 1e-12):.1f}x")
    print(f"  Circuit cutting / sampled          : {cut_wall / max(sampled_wall, 1e-12):.1f}x")
    print(f"  Cutting jobs                       : {cutter.n_jobs}")
    print(f"  Mean cutting job time              : {fmt_time(cutter.total_time / max(cutter.n_jobs, 1))}")

    print(f"\nCUT PLAN (strategy: {args.cut_strategy})")
    for key, v in (cutter.plan_metrics or {}).items():
        print(f"  {key:30s}: {v:.4g}" if isinstance(v, float) else f"  {key:30s}: {v}")
    if args.cut_strategy == "joint" and cutter.plan_metrics:
        print("  (joint estimates depend on the selected gate-group configuration)")

    print("\nACCURACY")
    print(f"  Per-term |<P>cut - <P>exact| : RMS {pt_rms:.3e}, max {pt_max:.3e}")
    for p, exact, cut in zip(prob.labels, per_term_ex, per_term_cut):
        print(f"  {p} exact={exact:+.8f} cut={cut:+.8f} error={abs(cut - exact):.3e}")

    # ---- Sampling overhead ----
    if benchmark_rows:
        print("\nSAMPLING OVERHEAD (Var(cut)/Var(uncut) at equal shots)")
        w_ = max(6, n)
        print(f"{'pauli':>{w_}s} {'shots':>8s} {'samples':>9s} "
              f"{'<P>exact':>10s} {'bias':>12s} {'std_cut':>12s} {'std_uncut':>12s} "
              f"{'overhead':>10s} {'t_cut':>10s} {'t_uncut':>10s}")
        for r in benchmark_rows:
            print(f"{r['pauli']:>{w_}s} "
                  f"{r['shots']:8d} {r['samples']:9d} {r['exact']:+10.4f} "
                  f"{r['cut_bias']:+12.2e} {r['cut_std']:12.2e} "
                  f"{r['uncut_std_theory']:12.2e} {r['overhead_var']:9.1f}x "
                  f"{fmt_time(r['cut_time']):>10s} {fmt_time(r['uncut_time']):>10s}")
        print(f"(std estimated from {args.bench_repeats} repeats)")

    nl = max(len(prob.labels), 1)
    data = {
        "args": vars(args),
        "algorithm": "GHZ",
        "n_qubits": n,
        "topology": args.topology,
        "fidelity_exact": fid_exact,
        "circuit": cstats,
        "observables": prob.labels,
        "coeffs": prob.coeffs.tolist(),
        "witness_offset": prob.offset,
        "samples": args.samples,
        "shots": args.shots,
        "witness_exact": witness_ex,
        "witness_sampled": witness_sampled,
        "witness_cut": witness_cut,
        "witness_error": witness_error,
        "statevector_max_dev": sv_dev,
        "per_term_exact": per_term_ex,
        "per_term_cut": per_term_cut,
        "per_term_error_rms": pt_rms,
        "per_term_error_max": pt_max,
        "per_term_cut_time_mean": {p: cutter.term_time[p] / cutter.term_jobs[p] for p in cutter.term_time},
        "time": {
            "ray_init": ray_init_time,
            "exact_wall": exact_wall,
            "statevector_wall": sv_wall,
            "sampled_wall": sampled_wall,
            "cut_wall": cut_wall,
            "cut_jobs": cutter.n_jobs,
            "exact_eval": exact_wall / nl,
            "sampled_eval": sampled_wall / nl,
            "cut_eval": cut_wall / nl,
            "slowdown_total": cut_wall / max(sv_wall if sv_wall is not None else exact_wall, 1e-12),
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