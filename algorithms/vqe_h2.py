#!/usr/bin/env python
# h2_vqe_cut.py
"""
H2 VQE with and without circuit cutting: time, accuracy and sampling-overhead comparison.

Examples
--------
# 4-qubit H2 (STO-3G, built-in Hamiltonian), subcircuits of at most 2 qubits
python h2_vqe_cut.py --qubits 4 --max-qubits-per-subcircuit 2

# 8-qubit H2 (6-31G basis; needs `pip install pyscf qiskit-nature`), subcircuits of at most 4 qubits,
# keep only the 60 largest Pauli terms to keep the cutting run affordable, and benchmark overhead
python h2_vqe_cut.py --qubits 8 --max-qubits-per-subcircuit 4 --max-terms 60 --benchmark

Notes
-----
* H2 in the minimal basis (STO-3G) needs 4 qubits.  8 qubits means H2 in a larger basis
  (6-31G: 4 spatial orbitals -> 8 spin orbitals, Jordan-Wigner), generated with PySCF via qiskit-nature.
  You can also supply your own Hamiltonian with --hamiltonian-file (JSON list of [label, coeff]).
* All Pauli terms of one energy evaluation are reconstructed as separate OBSERVABLES of a SINGLE
  circuit-cutting job (X/Y terms are passed directly as Pauli strings, no manual basis rotation).
  Extra non-Z terms still add measurement-basis subexperiments, so cost grows with the number of terms.
  Use --threshold / --max-terms to truncate the Hamiltonian (applied identically to ALL methods).
* --cut-strategy selects how cuts are placed: "gate" (qubit partition by --partitioner, gate cuts only) or
  "joint" (Frohler et al. 2026: gate + wire cuts with gate-group-aware costs; see joint_cutting.py).
  NOTE: gate groups are costed jointly during placement, but circuit_cut executes them as individual cuts, so the
  measured overhead equals the 'individual' estimate reported in the cut-plan summary.
* The sampling-overhead benchmark still runs one single-observable cutting job per Pauli term, because it
  measures the variance of an individual term's estimator.
"""

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from circuit_cut.distribute import RayAgent
from circuit_cut.cutting import CircuitCutting


CHEMICAL_ACCURACY = 1.6e-3  # Ha


# ============================================================
# Hamiltonians
# ============================================================

# H2 at ~0.74 A, STO-3G, Jordan-Wigner, 4 qubits (built-in, no extra dependencies)
H2_4Q = SparsePauliOp.from_list([
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


def _to_real(op):
    op = op.simplify()
    if len(op.coeffs) and np.max(np.abs(op.coeffs.imag)) > 1e-8:
        raise ValueError("Hamiltonian has significant imaginary coefficients.")
    return SparsePauliOp(op.paulis, op.coeffs.real)


def _pyscf_hamiltonian(bond_length, basis):
    """H2 -> qubit Hamiltonian via PySCF + qiskit-nature (Jordan-Wigner, nuclear repulsion included)."""
    try:
        from qiskit_nature.units import DistanceUnit
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
    except ImportError as e:
        raise ImportError(
            "Building this Hamiltonian needs PySCF and qiskit-nature:\n"
            "    pip install pyscf qiskit-nature\n"
            "Alternatively pass --hamiltonian-file with a JSON list of [pauli_label, coeff]."
        ) from e

    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis=basis, charge=0, spin=0, unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    qubit_op = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
    return _to_real(qubit_op)


def build_hamiltonian(n_qubits, bond_length, basis, ham_file):
    """Return (SparsePauliOp, default HF-occupied qubits, description)."""
    if ham_file:
        with open(ham_file) as f:
            data = json.load(f)
        op = _to_real(SparsePauliOp.from_list([(lab, float(c)) for lab, c in data]))
        if op.num_qubits != n_qubits:
            raise ValueError(f"{ham_file} has {op.num_qubits} qubits but --qubits={n_qubits}.")
        return op, [0, n_qubits // 2], f"file {ham_file}"

    if n_qubits == 4 and basis is None and abs(bond_length - 0.74) < 1e-9:
        return H2_4Q, [0, 1], "built-in H2 / STO-3G / 0.74 A / Jordan-Wigner"

    basis = basis or ("sto-3g" if n_qubits == 4 else "6-31g")
    op = _pyscf_hamiltonian(bond_length, basis)
    if op.num_qubits != n_qubits:
        raise ValueError(
            f"H2 in basis '{basis}' needs {op.num_qubits} qubits, but --qubits={n_qubits}. "
            "Use sto-3g for 4 qubits and 6-31g for 8 qubits."
        )
    # qiskit-nature ordering: alpha spin orbitals first, then beta -> HF = orbital 0 alpha + orbital 0 beta
    return op, [0, n_qubits // 2], f"PySCF H2 / {basis} / {bond_length} A / Jordan-Wigner"


def truncate_hamiltonian(op, threshold, max_terms):
    """Drop small terms (identity is always kept). Same truncated H is used by every method."""
    labels = op.paulis.to_labels()
    coeffs = op.coeffs.real
    ident = [i for i, l in enumerate(labels) if set(l) == {"I"}]
    rest = [i for i, l in enumerate(labels) if set(l) != {"I"} and abs(coeffs[i]) >= threshold]
    rest.sort(key=lambda i: -abs(coeffs[i]))
    if max_terms is not None:
        rest = rest[:max_terms]
    keep = ident + rest
    return SparsePauliOp.from_list([(labels[i], coeffs[i]) for i in keep])


def reference_energies(op, n_electrons=2):
    """Full-space ground energy, and ground energy inside the n_electrons particle sector (if conserved)."""
    M = op.to_matrix()
    e_full = float(np.linalg.eigvalsh(M).min())
    pop = np.array([bin(i).count("1") for i in range(M.shape[0])])
    idx = np.where(pop == n_electrons)[0]
    oth = np.where(pop != n_electrons)[0]
    leak = np.abs(M[np.ix_(idx, oth)]).max() if len(oth) else 0.0
    e_sec = float(np.linalg.eigvalsh(M[np.ix_(idx, idx)]).min()) if leak < 1e-8 else None
    return e_full, e_sec


@dataclass
class Problem:
    op: SparsePauliOp
    n_qubits: int
    hf_qubits: list
    labels: list = field(init=False)
    coeffs: np.ndarray = field(init=False)

    def __post_init__(self):
        self.labels = list(self.op.paulis.to_labels())
        self.coeffs = np.array(self.op.coeffs.real)


# ============================================================
# VQE ansatz (generalised to n qubits; identical to the original for n = 4)
# ============================================================

def h2_ansatz(theta, n_qubits, hf_qubits):
    qc = QuantumCircuit(n_qubits)

    for q in hf_qubits:                      # Hartree-Fock reference
        qc.x(q)

    for i in range(n_qubits):                # variational layer 1
        qc.ry(theta[i], i)

    for i in range(n_qubits - 1):            # linear entanglement chain
        qc.cx(i, i + 1)

    for i in range(n_qubits):                # variational layer 2
        qc.ry(theta[n_qubits + i], i)

    return qc


def n_params(n_qubits):
    return 2 * n_qubits


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


def measurement_circuit(theta, pauli, prob):
    """Ansatz + basis change so that <pauli> = <Z...Z> on the returned qubits.
    (Used by the uncut sampled baseline; the cutting path passes Pauli strings straight to CircuitCutting.)"""
    qc = h2_ansatz(theta, prob.n_qubits, prob.hf_qubits)
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
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ============================================================
# 1) Exact (statevector, no cutting, no shot noise)
# ============================================================

def expectation_exact(theta, pauli, prob):
    sv = Statevector(h2_ansatz(theta, prob.n_qubits, prob.hf_qubits))
    return float(sv.expectation_value(SparsePauliOp(pauli)).real)


def energy_exact(theta, prob):
    sv = Statevector(h2_ansatz(theta, prob.n_qubits, prob.hf_qubits))
    return float(sv.expectation_value(prob.op).real)


def per_term_exact(theta, prob):
    sv = Statevector(h2_ansatz(theta, prob.n_qubits, prob.hf_qubits))
    return [float(sv.expectation_value(SparsePauliOp(l)).real) for l in prob.labels]


# ============================================================
# 2) Uncut but shot-sampled  (the fair baseline for sampling overhead)
# ============================================================

class SampledEvaluator:
    def __init__(self, prob, shots, rng):
        self.prob, self.shots, self.rng = prob, shots, rng

    def expectation(self, theta, pauli, shots=None):
        shots = shots or self.shots
        qc, z_qubits = measurement_circuit(theta, pauli, self.prob)
        if not z_qubits:
            return 1.0
        probs = Statevector(qc).probabilities()
        probs = probs / probs.sum()
        counts = self.rng.multinomial(shots, probs)
        mask = sum(1 << q for q in z_qubits)
        signs = np.array([1 - 2 * (bin(i & mask).count("1") & 1) for i in range(len(probs))])
        return float(np.dot(counts, signs) / shots)

    def energy(self, theta):
        return float(sum(c * self.expectation(theta, p)
                         for p, c in zip(self.prob.labels, self.prob.coeffs)))


# ============================================================
# 3) Circuit cutting
# ============================================================

class CutEvaluator:
    def __init__(self, prob, agent, qubits_per_subcircuit, shots, samples,
                 partitioner="gurobi", optimization_level=1, run_actual=False,
                 cut_strategy="gate", joint_opts=None, seed=0):
        self.prob, self.agent = prob, agent
        self.cut_strategy = cut_strategy
        self.joint_opts = dict(joint_opts or {})   # use_wire_cuts, use_gate_groups, kl_runs, kl_max_passes, kl_patience
        self.seed = seed
        self.plan_metrics = None                   # cut-plan statistics of the most recent job
        self.qps, self.shots, self.samples = qubits_per_subcircuit, shots, samples
        self.partitioner, self.opt_level = partitioner, optimization_level
        self.run_actual = run_actual      # also run the uncut reference inside every cutting job (slower)
        self.n_jobs = 0
        self.total_time = 0.0
        self.term_time = {}               # per-term time is the job time AMORTISED over its observables
        self.term_jobs = {}
        self.last_metrics = None
        self._printed_keys = False

    def expectations(self, theta, paulis, shots=None, samples=None):
        """
        ONE circuit-cutting job that reconstructs every non-identity Pauli in `paulis` as a separate
        observable. Returns {pauli_label: <P>} (identity terms are 1.0 without running anything).
        """
        paulis = list(dict.fromkeys(paulis))
        out = {p: 1.0 for p in paulis if is_identity(p)}
        targets = [p for p in paulis if not is_identity(p)]
        if not targets:
            return out

        qc = h2_ansatz(theta, self.prob.n_qubits, self.prob.hf_qubits)

        t0 = time.perf_counter()
        with CircuitCutting(
            qc,
            agent=self.agent,
            observable=targets,            # list of Pauli strings -> multi-observable run
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
        plan_keys = ("n_fragments", "n_cuts", "log10_gamma", "max_subcircuit_qubits", "partition_cut_gates",
                     "partition_wire_cuts", "joint_gate_cuts", "joint_wire_cuts", "joint_n_groups",
                     "joint_grouped_gates", "joint_est_log10_kappa", "joint_indiv_log10_kappa",
                     "total_subexperiments", "t_partition")
        self.plan_metrics = {k: metrics[k] for k in plan_keys if metrics.get(k) is not None}
        self.plan_metrics["cut_strategy"] = self.cut_strategy
        if not self._printed_keys:
            print("[cut] scalar metrics exposed by circuit_cut:", sorted(self.last_metrics))
            self._printed_keys = True

        values = json.loads(metrics["cutting_expvals"])
        out.update(dict(zip(targets, values)))
        return out

    def expectation(self, theta, pauli, shots=None, samples=None):
        """Single-term cutting job (used by the sampling-overhead benchmark)."""
        return float(self.expectations(theta, [pauli], shots=shots, samples=samples)[pauli])

    def energy_with_terms(self, theta):
        exp = self.expectations(theta, self.prob.labels)
        vals = [exp[p] for p in self.prob.labels]
        return float(np.dot(self.prob.coeffs, vals)), vals

    def energy(self, theta):
        return self.energy_with_terms(theta)[0]


# ============================================================
# VQE runner with timing
# ============================================================

@dataclass
class VQERun:
    result: object
    history: np.ndarray
    thetas: list
    eval_times: np.ndarray
    wall: float


def run_vqe(energy_fn, theta0, maxiter, label):
    history, thetas, eval_times = [], [], []

    def wrapped(theta):
        t0 = time.perf_counter()
        E = energy_fn(theta)
        dt = time.perf_counter() - t0
        history.append(E)
        thetas.append(np.array(theta, copy=True))
        eval_times.append(dt)
        print(f"[{label}] eval {len(history):4d}  E = {E:.8f} Ha   ({fmt_time(dt)})")
        return E

    t_start = time.perf_counter()
    result = minimize(wrapped, theta0, method="COBYLA", options={"maxiter": maxiter})
    wall = time.perf_counter() - t_start
    return VQERun(result, np.array(history), thetas, np.array(eval_times), wall)


# ============================================================
# Sampling-overhead benchmark
# ============================================================

def benchmark_sampling_overhead(prob, cutter, sampler, theta, paulis, configs, repeats):
    """
    For each Pauli term and each (shots, samples) setting, repeat the estimate `repeats` times with
    circuit cutting and without cutting (both shot-sampled) at the SAME shot budget.

        overhead = Var(cut estimator) / Var(uncut sampled estimator)

    = the factor by which the cut run needs more shots to reach the same statistical error.
    Var(uncut) is the analytic binomial variance (1 - <P>^2) / shots (less noisy than an empirical one);
    the empirical uncut std is stored as well.
    Each cut estimate is a single-observable cutting job, so the variance belongs to that term alone.
    """
    rows = []
    for pauli in paulis:
        exact = expectation_exact(theta, pauli, prob)
        for shots, samples in configs:
            cut_vals, cut_t, unc_vals, unc_t = [], [], [], []
            lib = {}
            for _ in range(repeats):
                t0 = time.perf_counter()
                cut_vals.append(cutter.expectation(theta, pauli, shots=shots, samples=samples))
                cut_t.append(time.perf_counter() - t0)
                lib = cutter.last_metrics or lib

                t0 = time.perf_counter()
                unc_vals.append(sampler.expectation(theta, pauli, shots=shots))
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
                  f"overhead={row['overhead_var']:.1f}x  t_cut={fmt_time(row['cut_time'])}")
    return rows


# ============================================================
# Plotting
# ============================================================

def make_plots(d, outfile, show):
    E_ref = d["E_ref"]
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))

    # --- Panel 1: energy convergence ---
    ax = axes[0, 0]
    ax.plot(d["hist_exact"], label="No cutting (exact statevector)", lw=2)
    if d["hist_sampled"]:
        ax.plot(d["hist_sampled"], label="No cutting (sampled)", lw=1.2, alpha=0.8)
    ax.plot(d["hist_cut"], label="Circuit cutting (sampled)", lw=1.5, alpha=0.85)
    ax.plot(d["hist_cut_true"], "--", label="Exact energy at cut-VQE parameters", lw=1.5)
    ax.axhline(E_ref, color="k", ls=":", label=f"Exact diagonalisation ({E_ref:.5f} Ha)")
    ax.set_xlabel("Function evaluation"); ax.set_ylabel("Energy (Ha)")
    ax.set_title(f"H2 VQE convergence ({d['n_qubits']} qubits)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Panel 2: error vs reference ---
    ax = axes[0, 1]
    eps = 1e-12
    ax.semilogy(np.abs(np.array(d["hist_exact"]) - E_ref) + eps, label="No cutting (exact)", lw=2)
    if d["hist_sampled"]:
        ax.semilogy(np.abs(np.array(d["hist_sampled"]) - E_ref) + eps, label="No cutting (sampled)", lw=1.2)
    ax.semilogy(np.abs(np.array(d["hist_cut"]) - E_ref) + eps, label="Circuit cutting", lw=1.5)
    ax.semilogy(np.abs(np.array(d["hist_cut_true"]) - E_ref) + eps, "--",
                label="Exact energy at cut-VQE params", lw=1.5)
    ax.axhline(CHEMICAL_ACCURACY, color="gray", ls=":", label="Chemical accuracy (1.6 mHa)")
    ax.set_xlabel("Function evaluation"); ax.set_ylabel("|E - E_exact| (Ha)")
    ax.set_title("Absolute error"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # --- Panel 3: per-term expectation values (top 30 by |coeff|) ---
    ax = axes[0, 2]
    terms = np.array(d["terms"])
    order = np.argsort(-np.abs(np.array(d["coeffs"])))[:30]
    x = np.arange(len(order)); w = 0.4
    ax.bar(x - w / 2, np.array(d["per_term_exact"])[order], w, label="No cutting")
    ax.bar(x + w / 2, np.array(d["per_term_cut"])[order], w, label="Circuit cutting")
    ax.set_xticks(x); ax.set_xticklabels(terms[order], rotation=75, fontsize=7)
    ax.set_ylabel("<P>")
    ax.set_title(f"Per-term <P> at final cut-VQE params (top {len(order)} terms by |coeff|)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # --- Panel 4: total wall time ---
    ax = axes[1, 0]
    names, vals = ["Exact\n(no cut)"], [d["time"]["exact_wall"]]
    if d["time"]["sampled_wall"] is not None:
        names.append("Sampled\n(no cut)"); vals.append(d["time"]["sampled_wall"])
    names.append("Circuit\ncutting"); vals.append(d["time"]["cut_wall"])
    bars = ax.bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"][:len(vals)])
    ax.set_yscale("log"); ax.set_ylabel("Total VQE wall time (s)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt_time(v), ha="center", va="bottom")
    ax.set_title(f"Execution time (cut / exact = {d['time']['slowdown_total']:.0f}x)")
    ax.grid(alpha=0.3, axis="y", which="both")

    # --- Panel 5: per-evaluation time ---
    ax = axes[1, 1]
    ax.semilogy(d["time"]["exact_eval_times"], label="Exact", lw=1.5)
    if d["time"]["sampled_eval_times"] is not None:
        ax.semilogy(d["time"]["sampled_eval_times"], label="Sampled (no cut)", lw=1.2)
    ax.semilogy(d["time"]["cut_eval_times"], label="Circuit cutting", lw=1.5)
    ax.set_xlabel("Function evaluation"); ax.set_ylabel("Time per energy evaluation (s)")
    ax.set_title(f"Per-evaluation time (cut / exact = {d['time']['slowdown_per_eval']:.0f}x)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # --- Panel 6: sampling overhead vs shots ---
    ax = axes[1, 2]
    rows = [r for r in d["benchmark"] if r["samples"] == d["samples"]]
    if rows:
        for p in sorted({r["pauli"] for r in rows}):
            rr = sorted([r for r in rows if r["pauli"] == p], key=lambda r: r["shots"])
            ax.loglog([r["shots"] for r in rr], [max(r["overhead_var"], 1e-3) for r in rr],
                      "o-", label=p)
        ax.axhline(1, color="k", ls=":", label="No overhead")
        ax.set_xlabel("Shots per subcircuit setting"); ax.set_ylabel("Var(cut) / Var(uncut)  (shot overhead)")
        ax.set_title(f"Sampling overhead (samples={d['samples']})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Run with --benchmark to\nshow the sampling-overhead panel",
                ha="center", va="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"Saved plot to {outfile}")
    if show:
        plt.show()


# ============================================================
# Main
# ============================================================

def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="H2 VQE: exact vs circuit cutting (time, accuracy, sampling overhead)")
    # problem
    ap.add_argument("--qubits", type=int, choices=[4, 8], default=4,
                    help="4 = H2/STO-3G, 8 = H2/6-31G (needs pyscf + qiskit-nature)")
    ap.add_argument("--bond-length", type=float, default=0.74, help="H-H distance in Angstrom")
    ap.add_argument("--basis", type=str, default=None, help="override basis (default sto-3g / 6-31g)")
    ap.add_argument("--hamiltonian-file", type=str, default=None,
                    help="JSON list of [pauli_label, coeff]; must match --qubits")
    ap.add_argument("--hf-qubits", type=parse_int_list, default=None,
                    help="comma list of qubits set to |1> for the HF state (override default)")
    ap.add_argument("--threshold", type=float, default=0.0, help="drop Pauli terms with |coeff| < threshold")
    ap.add_argument("--max-terms", type=int, default=None, help="keep only the N largest non-identity terms")
    # cutting
    ap.add_argument("--max-qubits-per-subcircuit", "--qubits-per-subcircuit", dest="qps",
                    type=int, default=2, help="max qubits available in one subcircuit")
    ap.add_argument("--shots", type=int, default=2 ** 14)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--cut-strategy", type=str, default="gate", choices=["gate", "joint"],
                    help="gate: qubit partition by --partitioner (gate cuts only); "
                         "joint: gate+wire cuts with gate groups (paper)")
    ap.add_argument("--partitioner", type=str, default="gurobi",
                    choices=["gurobi", "kernighan_lin", "spectral", "contiguous"],
                    help="used only with --cut-strategy gate")
    ap.add_argument("--no-wire-cuts", action="store_true", help="(joint) disable stage 2 (wire cuts)")
    ap.add_argument("--no-gate-groups", action="store_true", help="(joint) disable gate-group detection")
    ap.add_argument("--kl-runs", type=int, default=10, help="(joint) stage-1 random restarts (paper: 50)")
    ap.add_argument("--kl-max-passes", type=int, default=10, help="(joint) max KL passes per run")
    ap.add_argument("--kl-patience", type=int, default=15, help="(joint) non-improving moves before a pass stops")
    ap.add_argument("--optimization-level", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=48)
    ap.add_argument("--cut-run-actual", action="store_true",
                    help="also run the uncut reference circuit(s) inside every cutting job "
                         "(adds time to the cut measurements; off by default, exact reference comes from Statevector)")
    # optimisation
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--init-scale", type=float, default=0.0, help="std of random initial angles (0 = zeros)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sampled-baseline", action="store_true",
                    help="skip the uncut shot-sampled VQE baseline")
    # benchmark
    ap.add_argument("--benchmark", action="store_true", help="run the sampling-overhead benchmark")
    ap.add_argument("--bench-terms", type=int, default=3, help="number of Pauli terms to benchmark")
    ap.add_argument("--bench-repeats", type=int, default=5, help="repeats per setting (more = tighter std)")
    ap.add_argument("--bench-shots", type=parse_int_list, default=[1024, 4096, 16384],
                    help="shots sweep (samples fixed to --samples)")
    ap.add_argument("--bench-samples", type=parse_int_list, default=[1000, 10000],
                    help="samples sweep (shots fixed to --shots)")
    # output
    ap.add_argument("--plot", type=str, default="h2_vqe_comparison.png")
    ap.add_argument("--data", type=str, default="h2_vqe_results.json")
    ap.add_argument("--no-show", action="store_true", help="do not open the matplotlib window")
    args = ap.parse_args()

    if args.qps < 1:
        ap.error("--max-qubits-per-subcircuit must be >= 1")
    if args.qps >= args.qubits:
        print(f"WARNING: subcircuit size {args.qps} >= {args.qubits} qubits -> nothing needs to be cut.")

    rng = np.random.default_rng(args.seed)

    # ---- Build problem ----
    op_full, hf_default, desc = build_hamiltonian(args.qubits, args.bond_length, args.basis,
                                                  args.hamiltonian_file)
    op = truncate_hamiltonian(op_full, args.threshold, args.max_terms)
    prob = Problem(op, args.qubits, args.hf_qubits if args.hf_qubits is not None else hf_default)

    print(f"Hamiltonian : {desc}")
    print(f"Qubits      : {prob.n_qubits}   HF qubits: {prob.hf_qubits}   parameters: {n_params(prob.n_qubits)}")
    print(f"Pauli terms : {len(prob.labels)} used (of {len(op_full)} in the full Hamiltonian)")

    E_full_untrunc, _ = reference_energies(op_full)
    E_ref, E_sector = reference_energies(op)
    print(f"Exact ground-state energy (diagonalisation, used H) = {E_ref:.8f} Ha")
    if E_sector is not None and abs(E_sector - E_ref) > 1e-9:
        print(f"  note: 2-electron-sector ground energy = {E_sector:.8f} Ha "
              f"(full Fock-space minimum is lower; the ansatz does not conserve particle number)")
    if len(op) != len(op_full):
        print(f"  truncation shifted the exact ground energy by {E_ref - E_full_untrunc:+.3e} Ha "
              f"(untruncated: {E_full_untrunc:.8f} Ha)")

    theta0 = rng.normal(0.0, args.init_scale, n_params(prob.n_qubits)) if args.init_scale > 0 \
        else np.zeros(n_params(prob.n_qubits))

    # ---- VQE 1: exact statevector (no cutting) ----
    run_exact = run_vqe(lambda th: energy_exact(th, prob), theta0, args.maxiter, "exact")

    # ---- VQE 2: uncut but shot-sampled baseline ----
    run_samp = None
    if not args.no_sampled_baseline:
        sampler = SampledEvaluator(prob, args.shots, np.random.default_rng(args.seed + 1))
        run_samp = run_vqe(sampler.energy, theta0, args.maxiter, "sampled")
    else:
        sampler = SampledEvaluator(prob, args.shots, np.random.default_rng(args.seed + 1))

    # ---- VQE 3: circuit cutting ----
    t_init = time.perf_counter()
    agent = RayAgent(num_cpus_per_node=args.cpus)
    agent.initialise()
    ray_init_time = time.perf_counter() - t_init
    print(f"Ray initialisation took {fmt_time(ray_init_time)} (excluded from VQE wall times)")

    joint_opts = dict(use_wire_cuts=not args.no_wire_cuts, use_gate_groups=not args.no_gate_groups,
                      kl_runs=args.kl_runs, kl_max_passes=args.kl_max_passes, kl_patience=args.kl_patience)
    cutter = CutEvaluator(prob, agent, args.qps, args.shots, args.samples,
                          partitioner=args.partitioner, optimization_level=args.optimization_level,
                          run_actual=args.cut_run_actual, cut_strategy=args.cut_strategy,
                          joint_opts=joint_opts if args.cut_strategy == "joint" else None, seed=args.seed)
    benchmark_rows = []
    try:
        run_cut = run_vqe(cutter.energy, theta0, args.maxiter, "cut")
        theta_cut = run_cut.result.x

        # Per-term comparison + one more full cut energy at the cut-VQE optimum
        t0 = time.perf_counter()
        E_cut_final, per_term_cut = cutter.energy_with_terms(theta_cut)
        final_pass_time = time.perf_counter() - t0
        per_term_ex = per_term_exact(theta_cut, prob)

        # Sampling-overhead benchmark at the cut-VQE optimum
        if args.benchmark:
            cand = [(pauli_weight(p), abs(c), p) for p, c in zip(prob.labels, prob.coeffs)
                    if pauli_weight(p) > 0]
            cand.sort(reverse=True)  # highest weight first, then biggest coefficient
            bench_paulis = [p for _, _, p in cand[:args.bench_terms]]
            configs = [(s, args.samples) for s in args.bench_shots]
            configs += [(args.shots, m) for m in args.bench_samples]
            configs = list(dict.fromkeys(configs))
            print(f"\nBenchmarking sampling overhead on {bench_paulis} with configs {configs}")
            benchmark_rows = benchmark_sampling_overhead(
                prob, cutter, sampler, theta_cut, bench_paulis, configs, args.bench_repeats)
    finally:
        agent.stop_ray_clusters()

    # ---- Post-processing ----
    hist_cut_true = np.array([energy_exact(th, prob) for th in run_cut.thetas])
    E_exact_at_exact = energy_exact(run_exact.result.x, prob)
    E_exact_at_cut = energy_exact(theta_cut, prob)
    E_exact_at_samp = energy_exact(run_samp.result.x, prob) if run_samp else None

    pt_err = np.abs(np.array(per_term_cut) - np.array(per_term_ex))
    pt_rms, pt_max = float(np.sqrt(np.mean(pt_err ** 2))), float(pt_err.max())

    slowdown_total = run_cut.wall / max(run_exact.wall, 1e-12)
    slowdown_eval = float(np.mean(run_cut.eval_times) / max(np.mean(run_exact.eval_times), 1e-12))

    # ---- Summary ----
    def row(name, run, e_noisy, e_at_params):
        errn = f"{abs(e_noisy - E_ref):.3e}" if e_noisy is not None else "-"
        erra = f"{abs(e_at_params - E_ref):.3e}"
        print(f"{name:24s}{fmt_time(run.wall):>12s}{len(run.history):>8d}"
              f"{fmt_time(float(np.mean(run.eval_times))):>13s}"
              f"{(f'{e_noisy:.8f}' if e_noisy is not None else '-'):>16s}{errn:>13s}{erra:>16s}")

    print("\n" + "=" * 100)
    print(f"VQE RESULT  ({prob.n_qubits} qubits, {len(prob.labels)} Pauli terms, "
          f"subcircuit <= {args.qps} qubits, shots={args.shots}, samples={args.samples})")
    print("=" * 100)
    print(f"Reference (diagonalisation): {E_ref:.8f} Ha")
    print(f"{'method':24s}{'wall time':>12s}{'evals':>8s}{'time/eval':>13s}"
          f"{'final E (Ha)':>16s}{'|E-Eref|':>13s}{'|E(theta)-Eref|':>16s}")
    row("No cutting (exact)", run_exact, run_exact.result.fun, E_exact_at_exact)
    if run_samp:
        row("No cutting (sampled)", run_samp, run_samp.result.fun, E_exact_at_samp)
    row("Circuit cutting", run_cut, run_cut.result.fun, E_exact_at_cut)
    print("(|E(theta)-Eref| = noise-free energy at the optimiser's final parameters: parameter quality)")

    print("\nTIME")
    print(f"  Ray initialisation (one-off)          : {fmt_time(ray_init_time)}")
    print(f"  Cut vs exact, total wall time         : {slowdown_total:.1f}x slower")
    print(f"  Cut vs exact, per energy evaluation   : {slowdown_eval:.1f}x slower")
    if run_samp:
        print(f"  Cut vs uncut-sampled, total wall time : {run_cut.wall / max(run_samp.wall, 1e-12):.1f}x slower")
    print(f"  Cutting jobs run: {cutter.n_jobs}, mean {fmt_time(cutter.total_time / max(cutter.n_jobs, 1))} per job "
          f"(one multi-observable job per energy evaluation)")
    print(f"  One full energy at the optimum        : {fmt_time(final_pass_time)}")

    print(f"\nCUT PLAN  (strategy: {args.cut_strategy})")
    for k, v in (cutter.plan_metrics or {}).items():
        print(f"  {k:26s}: {v:.4g}" if isinstance(v, float) else f"  {k:26s}: {v}")
    if args.cut_strategy == "joint" and cutter.plan_metrics:
        print("  (joint_est_* assumes gate groups are cut jointly; execution cuts them individually, "
              "so log10_gamma tracks joint_indiv_log10_kappa)")

    print("\nACCURACY")
    print(f"  Cut energy at optimum (re-sampled)    : {E_cut_final:.8f} Ha "
          f"(exact at same params: {E_exact_at_cut:.8f}, sampling error {E_cut_final - E_exact_at_cut:+.3e})")
    print(f"  Per-term |<P>_cut - <P>_exact|        : RMS {pt_rms:.3e}, max {pt_max:.3e}")
    ok = abs(E_exact_at_cut - E_ref) < CHEMICAL_ACCURACY
    print(f"  Cut-VQE parameters within chemical accuracy (1.6 mHa) of reference: {ok}")
    print("Optimal theta (no cutting):", run_exact.result.x)
    print("Optimal theta (cutting)   :", theta_cut)

    if benchmark_rows:
        print("\nSAMPLING OVERHEAD (Var(cut)/Var(uncut) at equal shots = extra shots needed by cutting)")
        print(f"{'pauli':>{max(6, prob.n_qubits)}s} {'shots':>7s} {'samples':>8s} {'<P>exact':>9s} "
              f"{'bias':>10s} {'std_cut':>10s} {'std_uncut':>10s} {'overhead':>9s} {'t_cut':>10s} {'t_uncut':>10s}")
        for r in benchmark_rows:
            print(f"{r['pauli']:>{max(6, prob.n_qubits)}s} {r['shots']:7d} {r['samples']:8d} {r['exact']:+9.4f} "
                  f"{r['cut_bias']:+10.2e} {r['cut_std']:10.2e} {r['uncut_std_theory']:10.2e} "
                  f"{r['overhead_var']:8.1f}x {fmt_time(r['cut_time']):>10s} {fmt_time(r['uncut_time']):>10s}")
        print(f"(std estimated from {args.bench_repeats} repeats; use --bench-repeats 20+ for tighter numbers)")

    # ---- Save data ----
    data = {
        "args": vars(args),
        "hamiltonian": desc,
        "n_qubits": prob.n_qubits,
        "hf_qubits": prob.hf_qubits,
        "n_terms": len(prob.labels),
        "samples": args.samples,
        "E_ref": E_ref,
        "E_ref_2e_sector": E_sector,
        "hist_exact": run_exact.history.tolist(),
        "hist_sampled": run_samp.history.tolist() if run_samp else [],
        "hist_cut": run_cut.history.tolist(),
        "hist_cut_true": hist_cut_true.tolist(),
        "theta_exact": run_exact.result.x.tolist(),
        "theta_sampled": run_samp.result.x.tolist() if run_samp else None,
        "theta_cut": theta_cut.tolist(),
        "final_energy": {
            "exact": run_exact.result.fun,
            "sampled": run_samp.result.fun if run_samp else None,
            "cut": run_cut.result.fun,
            "exact_at_exact_params": E_exact_at_exact,
            "exact_at_sampled_params": E_exact_at_samp,
            "exact_at_cut_params": E_exact_at_cut,
            "cut_resampled_at_cut_params": E_cut_final,
        },
        "terms": prob.labels,
        "coeffs": prob.coeffs.tolist(),
        "per_term_exact": per_term_ex,
        "per_term_cut": per_term_cut,
        "per_term_error_rms": pt_rms,
        "per_term_error_max": pt_max,
        "per_term_cut_time_mean": {p: cutter.term_time[p] / cutter.term_jobs[p] for p in cutter.term_time},
        "time": {
            "ray_init": ray_init_time,
            "exact_wall": run_exact.wall,
            "sampled_wall": run_samp.wall if run_samp else None,
            "cut_wall": run_cut.wall,
            "exact_eval_times": run_exact.eval_times.tolist(),
            "sampled_eval_times": run_samp.eval_times.tolist() if run_samp else None,
            "cut_eval_times": run_cut.eval_times.tolist(),
            "slowdown_total": slowdown_total,
            "slowdown_per_eval": slowdown_eval,
            "cut_jobs": cutter.n_jobs,
            "final_pass_time": final_pass_time,
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