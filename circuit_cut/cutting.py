"""
CircuitCutting: configurable Qiskit circuit cutting + knitting, executed through RayAgent.

    from ray_agent import RayAgent
    from circuit_cutting import CircuitCutting

    agent = RayAgent(num_cpus_per_node=48)
    agent.initialise()

    # circuit = a QuantumCircuit (single run) or a callable n -> QuantumCircuit (sweeps)
    with CircuitCutting(my_circuit_factory, agent, partitioner="gurobi", qubits_per_subcircuit=10) as cc:
        cc.sweep(min_qubits=10, max_qubits=30, step=2)   # needs a callable circuit
        cc.compare_partitioners(24)                      # cheap plan-only comparison
    agent.stop_ray_clusters()

Observables (the `observable` argument) - one OR several can be reconstructed in the same run:

    observable="data"                  # single Z-product on the first n_data qubits   (default)
    observable="all"                   # single Z-product on all qubits
    observable="data_each"             # one observable per data qubit: Z0, Z1, ...
    observable="all_each"              # one observable per qubit
    observable=[0, 2, 5]               # single Z-product Z0 Z2 Z5            (flat list of ints)
    observable=[[0], [1], [0, 1]]      # several Z-products                   (list of int lists)
    observable=["ZZIII", "XIIIX"]      # several Pauli strings (qiskit order: rightmost char = qubit 0,
                                       # length must equal the transpiled circuit width)
    observable=[Pauli("ZZ"), [0, 1]]   # Pauli objects / int lists / strings can be mixed

Cut strategies (the `cut_strategy` argument):

    cut_strategy="gate"    # default: qubit partition from `partitioner`, gate cuts only (original behaviour)
    cut_strategy="joint"   # Frohler et al. 2026: two-stage KL (gate cuts, then wire cuts) with gate-group-aware costs
                           # extra args: use_wire_cuts, use_gate_groups, kl_runs, kl_max_passes, kl_patience
    CircuitCutting(factory, agent, cut_strategy="joint", qubits_per_subcircuit=10)

All observables share the same cut + the same partition. Note that each extra observable that needs a
different measurement basis adds subexperiments (all-Z observables are the cheapest to mix).
The first observable is also reported in the legacy columns (actual_expval / cutting_expval); the
error columns (absolute_error, error_in_sigmas, ...) are the MAX over observables, and the per-observable
values are stored as JSON lists (actual_expvals, cutting_expvals, absolute_errors, ...).

This module contains no circuit/algorithm code: the circuit is always supplied by the caller.
Circuits must be measurement-free. Pass agent=None to execute in-process (small local tests).
"""

import csv
import json
import math
import os
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives.containers import PrimitiveResult
from qiskit.quantum_info import Pauli, PauliList
from qiskit_addon_cutting import generate_cutting_experiments, partition_problem, reconstruct_expectation_values
from qiskit_aer.primitives import SamplerV2

from .joint_cutting import expand_paulis, plan_joint_cuts

OK_STATUSES = ("ok", "skipped_gamma", "planned")
SEP_WIDTH = 116

# ============================================================
# Logging / timing
# ============================================================
def log(message):
    print(f"[CUT] {message}", flush=True)

@contextmanager
def timed(store, key, tag="", verbose=True):
    """Time a block, store seconds in store[key], and log it."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - t0
        if verbose:
            log(f"{tag}{key:<26}: {store[key]:10.3f} s")

# ============================================================
# Interaction graph helpers
# ============================================================
def build_edge_weights(qc):
    """weight(a, b) = number of two-qubit gates between qubits a and b."""
    weights = Counter()
    for inst in qc.data:
        if len(inst.qubits) != 2:
            continue
        a, b = sorted(qc.find_bit(q).index for q in inst.qubits)
        weights[(a, b)] += 1
    return weights

def cut_weight(edge_weights, labels):
    """Number of two-qubit gates crossing fragment boundaries."""
    return sum(w for (a, b), w in edge_weights.items() if labels[a] != labels[b])

def normalize_labels(labels):
    """Relabel fragments to 0..k-1 in order of first appearance."""
    remap = {}
    for lab in labels:
        remap.setdefault(lab, len(remap))
    return [remap[lab] for lab in labels]

# ============================================================
# Cutting statistics
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
# Observable helpers
# ============================================================
def _is_int(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)

def pauli_support(pauli):
    """Qubit indices on which the Pauli acts non-trivially."""
    return [int(q) for q in np.flatnonzero(np.asarray(pauli.z) | np.asarray(pauli.x))]

def pauli_name(pauli):
    """Compact human-readable name, e.g. Z0Z1X4 (identity -> 'I')."""
    label = pauli.to_label().lstrip("+-i")
    name = "".join(f"{c}{i}" for i, c in enumerate(reversed(label)) if c != "I")
    return name or "I"

def _json_list(values):
    return json.dumps([None if v is None else float(v) for v in values])

# ============================================================
# Main class
# ============================================================
class CircuitCutting:
    PARTITIONERS = ("gurobi", "kernighan_lin", "spectral", "contiguous")
    CUT_STRATEGIES = ("gate", "joint")
    OBSERVABLE_KEYWORDS = ("data", "all", "data_each", "all_each")

    def __init__(
        self,
        circuit,               
        agent=None,
        *,
        # --- circuit handling ---
        observable="data",             # keyword | list of qubit indices | list of (int lists / Pauli strings / Paulis) - see module docstring
        basis_gates=("cx", "u"),
        optimization_level=1,
        # --- partitioning ---
        qubits_per_subcircuit=10,
        cut_strategy="gate",           # "gate": qubit partition by `partitioner` | "joint": gate+wire cuts with gate groups (paper)
        partitioner="gurobi",          # (cut_strategy="gate") one of PARTITIONERS, or callable(edge_weights, n_qubits, max_per) -> labels
        use_wire_cuts=True,            # (joint) run stage 2 (wire cuts)
        use_gate_groups=True,          # (joint) detect cascades / parallel CNOT groups and cost them jointly
        kl_runs=10,                    # (joint) random restarts of stage 1 (paper: 50)
        kl_max_passes=10,              # (joint) max KL passes per run
        kl_patience=15,                # (joint) stop a pass after this many non-improving moves
        gurobi_time_limit=300,
        gurobi_mip_gap=0.0,
        seed=0,                        # used by kernighan_lin
        # --- sampling / execution ---
        shots=2 ** 14,
        samples=10000,                 # Monte Carlo samples when QPD terms exceed enumerate_below
        enumerate_below=10000,         # enumerate all QPD terms when their count is below this
        max_gamma=1000,
        skip_high_gamma=False,
        task_multiplier=4,
        # --- reference (uncut) run ---
        run_actual="auto",             # True: always | False: never | "auto": only if simulated qubits < actual_max_qubits
        actual_max_qubits=30,
        # --- misc ---
        dry_run=False,                 # stop after partition + cut statistics (no execution)
        output_dir="results",
        save=True,
        verbose=True,
    ):
        if isinstance(partitioner, str) and partitioner not in self.PARTITIONERS:
            raise ValueError(f"partitioner must be one of {self.PARTITIONERS} or a callable")
        if cut_strategy not in self.CUT_STRATEGIES:
            raise ValueError(f"cut_strategy must be one of {self.CUT_STRATEGIES}")
        if qubits_per_subcircuit < 2:
            raise ValueError("qubits_per_subcircuit must be >= 2")
        if run_actual not in (True, False, "auto"):
            raise ValueError("run_actual must be True, False or 'auto'")
        if not (isinstance(circuit, QuantumCircuit) or callable(circuit)):
            raise TypeError("circuit must be a QuantumCircuit or a callable n_data -> QuantumCircuit")
        if isinstance(observable, str) and observable not in self.OBSERVABLE_KEYWORDS:
            raise ValueError(f"observable string must be one of {self.OBSERVABLE_KEYWORDS}")

        self.agent = agent
        self.circuit = circuit
        self.observable = observable
        self.basis_gates = list(basis_gates)
        self.optimization_level = optimization_level
        self.qubits_per_subcircuit = qubits_per_subcircuit
        self.cut_strategy = cut_strategy
        self.partitioner = partitioner
        self.use_wire_cuts = use_wire_cuts
        self.use_gate_groups = use_gate_groups
        self.kl_runs = kl_runs
        self.kl_max_passes = kl_max_passes
        self.kl_patience = kl_patience
        self.gurobi_time_limit = gurobi_time_limit
        self.gurobi_mip_gap = gurobi_mip_gap
        self.seed = seed
        self.shots = shots
        self.samples = samples
        self.enumerate_below = enumerate_below
        self.max_gamma = max_gamma
        self.skip_high_gamma = skip_high_gamma
        self.task_multiplier = task_multiplier
        self.run_actual = run_actual
        self.actual_max_qubits = actual_max_qubits
        self.dry_run = dry_run
        self.output_dir = output_dir
        self.save = save
        self.verbose = verbose

        self.out_dir = None
        self.all_metrics = []
        self._gurobi_env = None

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._gurobi_env is not None:
            self._gurobi_env.dispose()
            self._gurobi_env = None

    def _log(self, msg):
        if self.verbose:
            log(msg)

    def _get_gurobi_env(self):
        if self._gurobi_env is None:
            import gurobipy as gp
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            self._gurobi_env = env
        return self._gurobi_env

    # --------------------------------------------------------
    # Circuit / observables
    # --------------------------------------------------------
    def _make_circuit(self, n_data):
        """Return (circuit, n_data). n_data = size parameter / number of 'data' qubits."""
        if isinstance(self.circuit, QuantumCircuit):
            qc = self.circuit
            return qc, (qc.num_qubits if n_data is None else n_data)
        if n_data is None:
            raise ValueError("n_data is required when circuit is a callable")
        qc = self.circuit(n_data)
        if not isinstance(qc, QuantumCircuit):
            raise TypeError("circuit callable must return a QuantumCircuit")
        return qc, n_data

    @staticmethod
    def _z_product(n_total, z_qubits):
        z_qubits = sorted(int(q) for q in z_qubits)
        if any(q < 0 or q >= n_total for q in z_qubits):
            raise ValueError(f"observable qubit index out of range for a {n_total}-qubit circuit: {z_qubits}")
        z = np.zeros(n_total, dtype=bool)
        z[z_qubits] = True
        x = np.zeros(n_total, dtype=bool)
        return Pauli((z, x))

    def _resolve_observables(self, n_data, n_total):
        """Turn self.observable into a list of Pauli objects (each of length n_total)."""
        obs = self.observable
        if isinstance(obs, str):
            if obs == "data":
                specs = [list(range(n_data))]
            elif obs == "all":
                specs = [list(range(n_total))]
            elif obs == "data_each":
                specs = [[q] for q in range(n_data)]
            else:  # "all_each"
                specs = [[q] for q in range(n_total)]
        elif isinstance(obs, Pauli):
            specs = [obs]
        else:
            obs = list(obs)
            if not obs:
                raise ValueError("observable list is empty")
            if all(_is_int(q) for q in obs):
                specs = [obs]               # legacy: flat list of ints = ONE Z-product
            else:
                specs = obs

        paulis = []
        for spec in specs:
            if isinstance(spec, Pauli):
                p = spec
            elif isinstance(spec, str):
                p = Pauli(spec)
            else:
                p = self._z_product(n_total, spec)
            if p.num_qubits != n_total:
                raise ValueError(f"Pauli observable '{p.to_label()}' acts on {p.num_qubits} qubits, "
                                 f"but the transpiled circuit has {n_total}")
            paulis.append(p)
        return paulis

    @staticmethod
    def _build_observable(paulis):
        return PauliList(list(paulis))

    @staticmethod
    def _expectation_from_counts(counts, support):
        """<prod P_q for q in support> from all-qubit counts, measured in P's eigenbasis
        (qubit q is bit -1-q of the string)."""
        total = sum(counts.values())
        if total == 0:
            raise RuntimeError("Actual-circuit execution returned zero shots.")
        acc = 0.0
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            parity = sum(bits[-1 - q] == "1" for q in support) % 2
            acc += (1.0 if parity == 0 else -1.0) * count
        return acc / total

    @staticmethod
    def _measured_circuit(tqc, pauli=None):
        """Copy of tqc with basis rotations for pauli's X/Y factors (none for Z-only) + measure_all."""
        qc = tqc.copy()
        if pauli is not None:
            for q in range(pauli.num_qubits):
                if pauli.x[q] and pauli.z[q]:      # Y
                    qc.sdg(q)
                    qc.h(q)
                elif pauli.x[q]:                   # X
                    qc.h(q)
        qc.measure_all()
        return qc

    # --------------------------------------------------------
    # Partitioners
    # --------------------------------------------------------
    def _partition_gurobi(self, edge_weights, n_qubits, max_per):
        import gurobipy as gp
        from gurobipy import GRB

        n_frag = math.ceil(n_qubits / max_per)
        edges = list(edge_weights.keys())

        model = gp.Model("quantum_partition", env=self._get_gurobi_env())
        model.Params.TimeLimit = self.gurobi_time_limit
        model.Params.MIPGap = self.gurobi_mip_gap

        x = model.addVars(n_qubits, n_frag, vtype=GRB.BINARY, name="x")
        y = model.addVars(len(edges), lb=0.0, ub=1.0, name="y")

        for q in range(n_qubits):
            model.addConstr(gp.quicksum(x[q, f] for f in range(n_frag)) == 1)
        for f in range(n_frag):
            model.addConstr(gp.quicksum(x[q, f] for q in range(n_qubits)) <= max_per)
        for q in range(n_qubits):                      # symmetry breaking
            for f in range(q + 1, n_frag):
                x[q, f].UB = 0
        for e, (a, b) in enumerate(edges):
            for f in range(n_frag):
                model.addConstr(y[e] >= x[a, f] - x[b, f])

        model.setObjective(gp.quicksum(edge_weights[edges[e]] * y[e] for e in range(len(edges))), GRB.MINIMIZE)

        for q in range(n_qubits):                      # warm start
            for f in range(n_frag):
                x[q, f].Start = 1.0 if f == q // max_per else 0.0

        t0 = time.perf_counter()
        model.optimize()
        solve_time = time.perf_counter() - t0

        if model.SolCount == 0:
            model.dispose()
            raise RuntimeError("Gurobi did not find a feasible partition.")

        labels = [max(range(n_frag), key=lambda f: x[q, f].X) for q in range(n_qubits)]
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

    def _partition_kernighan_lin(self, edge_weights, n_qubits, max_per):
        """Recursive weighted Kernighan-Lin bisection until every part has <= max_per qubits.
        May produce more fragments than ceil(n/max_per) because splits are balanced halves."""
        import networkx as nx
        from networkx.algorithms.community import kernighan_lin_bisection

        g = nx.Graph()
        g.add_nodes_from(range(n_qubits))
        for (a, b), w in edge_weights.items():
            g.add_edge(a, b, weight=w)

        parts, stack = [], [set(range(n_qubits))]
        while stack:
            s = stack.pop()
            if len(s) <= max_per:
                parts.append(s)
                continue
            a, b = kernighan_lin_bisection(g.subgraph(s), weight="weight", seed=self.seed)
            stack.extend([set(a), set(b)])

        labels = [0] * n_qubits
        for i, part in enumerate(sorted(parts, key=min)):
            for q in part:
                labels[q] = i
        return labels, {}

    def _partition_spectral(self, edge_weights, n_qubits, max_per):
        """Order qubits by the Fiedler vector of the weighted Laplacian, then cut into equal blocks."""
        w = np.zeros((n_qubits, n_qubits))
        for (a, b), weight in edge_weights.items():
            w[a, b] += weight
            w[b, a] += weight
        lap = np.diag(w.sum(axis=1)) - w
        _, vecs = np.linalg.eigh(lap)
        order = np.argsort(vecs[:, 1], kind="stable")
        labels = [0] * n_qubits
        for rank, q in enumerate(order):
            labels[int(q)] = rank // max_per
        return labels, {}

    def _partition_contiguous(self, edge_weights, n_qubits, max_per):
        """Naive baseline: consecutive qubit indices share a fragment."""
        return [q // max_per for q in range(n_qubits)], {}

    def _partition(self, edge_weights, n_qubits):
        max_per = self.qubits_per_subcircuit
        if callable(self.partitioner):
            name, fn = getattr(self.partitioner, "__name__", "custom"), self.partitioner
            raw = fn(edge_weights, n_qubits, max_per)
            labels, info = (raw if isinstance(raw, tuple) else (raw, {}))
        else:
            name = self.partitioner
            labels, info = getattr(self, f"_partition_{name}")(edge_weights, n_qubits, max_per)

        labels = normalize_labels(list(labels))
        sizes = Counter(labels)
        if max(sizes.values()) > max_per:
            raise RuntimeError(f"Partitioner '{name}' produced a fragment with {max(sizes.values())} > {max_per} qubits.")

        info = dict(info)
        info.update(
            partitioner=name,
            partition_cut_gates=int(cut_weight(edge_weights, labels)),
            partition_max_fragment=int(max(sizes.values())),
            partition_min_fragment=int(min(sizes.values())),
        )
        return labels, info

    def _plan_joint(self, tqc, paulis):
        """Paper's joint-cutting placement. Returns (cut_circuit, labels, cut_paulis, info). The cut circuit
        contains explicit Move instructions for wire cuts; labels/observables refer to that wider circuit."""
        cut_qc, frag, final_q, info = plan_joint_cuts(
            tqc, self.qubits_per_subcircuit,
            use_wire_cuts=self.use_wire_cuts, use_gate_groups=self.use_gate_groups,
            runs=self.kl_runs, max_passes=self.kl_max_passes, patience=self.kl_patience, seed=self.seed)
        labels = normalize_labels(frag)
        sizes = Counter(labels)
        if max(sizes.values()) > self.qubits_per_subcircuit:
            raise RuntimeError("Joint placement produced a fragment wider than qubits_per_subcircuit.")
        info.update(
            partitioner="joint",
            partition_cut_gates=info["joint_gate_cuts"],
            partition_wire_cuts=info["joint_wire_cuts"],
            partition_max_fragment=int(max(sizes.values())),
            partition_min_fragment=int(min(sizes.values())),
        )
        return cut_qc, labels, expand_paulis(paulis, final_q, cut_qc.num_qubits), info

    # --------------------------------------------------------
    # Execution backends
    # --------------------------------------------------------
    def _run_local(self, subexperiments):
        """In-process fallback when no RayAgent is supplied."""
        t0 = time.perf_counter()
        sampler = SamplerV2()
        results = {}
        total = 0
        for label, circuits in subexperiments.items():
            circuits = list(circuits)
            total += len(circuits)
            results[label] = PrimitiveResult(list(sampler.run(circuits, shots=self.shots).result()) if circuits else [])
        wall = time.perf_counter() - t0
        info = {
            "num_tasks": len(subexperiments), "chunk_size": 0, "total_circuits": total, "total_cpus": os.cpu_count(),
            "timings": {"submit": 0.0, "execute_and_gather": wall, "wall": wall},
            "circuits_per_second": total / wall if wall > 0 else float("nan"),
        }
        return results, info

    def _run_actual_circuit(self, qc):
        """Run one circuit; returns counts."""
        if self.agent is not None:
            import ray
            results, _ = ray.get(self.agent.run_chunk([qc], self.shots))
            result = results[0]
        else:
            result = SamplerV2().run([qc], shots=self.shots).result()[0]
        return result.data.meas.get_counts()

    def _should_run_actual(self, n_sim_qubits):
        if self.run_actual is True:
            return True
        if self.run_actual is False:
            return False
        return n_sim_qubits < self.actual_max_qubits

    # --------------------------------------------------------
    # Stages
    # --------------------------------------------------------
    def _stage_build_and_transpile(self, n_data, timings, metrics, tag):
        with timed(timings, "build_circuit", tag, self.verbose):
            qc, n_data = self._make_circuit(n_data)
        with timed(timings, "transpile", tag, self.verbose):
            tqc = transpile(qc, basis_gates=self.basis_gates, optimization_level=self.optimization_level)

        two_q = sum(1 for inst in tqc.data if len(inst.qubits) == 2)
        metrics.update(
            circuit_qubits=tqc.num_qubits,
            n_ancilla=tqc.num_qubits - n_data,
            circuit_depth=tqc.depth(),
            circuit_ops=len(tqc.data),
            circuit_cx=int(tqc.count_ops().get("cx", 0)),
            circuit_two_qubit_gates=two_q,
        )
        self._log(f"{tag}circuit: {tqc.num_qubits} qubits, depth {tqc.depth()}, {len(tqc.data)} ops, {two_q} two-qubit gates")
        return tqc, n_data

    def _stage_actual(self, tqc, paulis, timings, metrics, artifacts, tag):
        """Reference (uncut) expectation values for all observables.
        All diagonal (Z/I-only) observables share ONE execution; each observable containing X/Y gets its own
        rotated execution. Returns (expvals, stderrs) lists."""
        n_obs = len(paulis)
        expvals = [None] * n_obs
        with timed(timings, "actual_execution", tag, self.verbose):
            diagonal = [i for i, p in enumerate(paulis) if not np.any(p.x)]
            if diagonal:
                counts = self._run_actual_circuit(self._measured_circuit(tqc))
                artifacts["actual_counts"] = np.array([[str(k), int(v)] for k, v in counts.items()], dtype=object)
                for i in diagonal:
                    expvals[i] = self._expectation_from_counts(counts, pauli_support(paulis[i]))
            for i, p in enumerate(paulis):
                if expvals[i] is not None:
                    continue
                counts = self._run_actual_circuit(self._measured_circuit(tqc, p))
                artifacts[f"actual_counts_obs{i}"] = np.array([[str(k), int(v)] for k, v in counts.items()], dtype=object)
                expvals[i] = self._expectation_from_counts(counts, pauli_support(p))

        # shot-noise standard error of each reference value itself
        stderrs = [math.sqrt(max(0.0, 1.0 - e ** 2) / self.shots) for e in expvals]
        metrics["actual_expval"] = float(expvals[0])                  # first observable (legacy column)
        metrics["actual_expvals"] = _json_list(expvals)
        metrics["actual_shots"] = int(self.shots)
        metrics["actual_stderr"] = float(stderrs[0])
        metrics["actual_stderrs"] = _json_list(stderrs)
        metrics["actual_n_executions"] = (1 if diagonal else 0) + (n_obs - len(diagonal))
        artifacts["actual_expvals"] = np.array(expvals, dtype=float)
        if n_obs == 1:
            self._log(f"{tag}actual (uncut) <{pauli_name(paulis[0])}> = {expvals[0]:.6g} (+/- {stderrs[0]:.2g})")
        else:
            self._log(f"{tag}actual (uncut): {n_obs} observables, {metrics['actual_n_executions']} execution(s); "
                      f"first <{pauli_name(paulis[0])}> = {expvals[0]:.6g} (+/- {stderrs[0]:.2g})")
        return expvals, stderrs

    def _stage_partition(self, tqc, paulis, timings, metrics, artifacts, tag):
        """Returns (circuit to cut, fragment labels, observables for that circuit)."""
        with timed(timings, "partition", tag, self.verbose):
            if self.cut_strategy == "joint":
                cut_qc, labels, cut_paulis, info = self._plan_joint(tqc, paulis)
            else:
                edge_weights = build_edge_weights(tqc)
                labels, info = self._partition(edge_weights, tqc.num_qubits)
                cut_qc, cut_paulis = tqc, paulis
        metrics.update(info)
        metrics["cut_strategy"] = self.cut_strategy
        metrics["n_fragments"] = len(set(labels))
        artifacts["partition_labels"] = np.array(labels, dtype=int)
        extra = ""
        if self.cut_strategy == "joint":
            extra = (f", {info['joint_wire_cuts']} wire cuts, {info['joint_n_groups']} gate groups, "
                     f"est log10(kappa)={info['joint_est_log10_kappa']:.2f} "
                     f"(individual cuts: {info['joint_indiv_log10_kappa']:.2f})")
            if info["joint_grouped_gates"]:
                self._log(f"{tag}NOTE: qiskit-addon-cutting has no joint gate-cut decomposition; grouped gates are "
                          f"cut individually at execution, so log10_gamma will match the 'individual' estimate.")
        self._log(f"{tag}partition[{info['partitioner']}]: {metrics['n_fragments']} fragments, "
                  f"{info['partition_cut_gates']} cut gates{extra}")
        return cut_qc, labels, cut_paulis

    def _stage_cut(self, tqc, cut_qc, labels, cut_paulis, timings, metrics, tag):
        with timed(timings, "cut_circuit", tag, self.verbose):
            observable = self._build_observable(cut_paulis)
            problem = partition_problem(circuit=cut_qc, partition_labels=labels, observables=observable)
            subcircuits, subobservables, bases = problem.subcircuits, problem.subobservables, problem.bases

        num_cuts, log10_gamma, log10_terms = cut_statistics(bases)
        sizes = [sc.num_qubits for sc in subcircuits.values()]
        gamma = 10 ** log10_gamma if log10_gamma < 150 else float("inf")

        metrics.update(
            n_subcircuits=len(subcircuits),
            max_subcircuit_qubits=max(sizes),
            min_subcircuit_qubits=min(sizes),
            qubit_reduction_factor=tqc.num_qubits / max(sizes),
            n_cuts=num_cuts,
            cuts_per_two_qubit_gate=num_cuts / max(1, metrics["circuit_two_qubit_gates"]),
            log10_gamma=log10_gamma,
            gamma=gamma,
            sampling_overhead=gamma ** 2 if math.isfinite(gamma) else float("inf"),   # gamma^2
            log10_sampling_overhead=2 * log10_gamma,
            log10_qpd_terms=log10_terms,
            gamma_exceeds_limit=bool(log10_gamma > math.log10(self.max_gamma)),
        )
        self._log(f"{tag}cutting: {len(subcircuits)} subcircuits, {num_cuts} cuts, "
                  f"log10(gamma)={log10_gamma:.2f}, log10(gamma^2)={2 * log10_gamma:.2f}")
        return subcircuits, subobservables, log10_terms

    def _stage_generate(self, subcircuits, subobservables, log10_terms, timings, metrics, artifacts, tag):
        enumerate_all = log10_terms < math.log10(self.enumerate_below)
        num_samples = np.inf if enumerate_all else self.samples
        metrics["sampling_mode"] = "enumerate" if enumerate_all else "monte_carlo"
        metrics["mc_samples"] = -1 if enumerate_all else self.samples

        with timed(timings, "generate_experiments", tag, self.verbose):
            subexperiments, coefficients = generate_cutting_experiments(
                circuits=subcircuits, observables=subobservables, num_samples=num_samples)

        per_fragment = {str(k): len(v) for k, v in subexperiments.items()}
        total_subexp = sum(per_fragment.values())
        metrics.update(
            total_subexperiments=total_subexp,                  # circuit overhead vs 1 uncut circuit
            total_cutting_shots=total_subexp * self.shots,
            n_coefficients=len(coefficients),
            subexperiments_per_fragment=json.dumps(per_fragment),
        )
        artifacts["subexperiments_per_fragment"] = np.array([per_fragment[str(k)] for k in subexperiments], dtype=int)
        artifacts["qpd_weights"] = np.array([w for w, _ in coefficients], dtype=float)
        self._log(f"{tag}experiments: {total_subexp} circuits ({metrics['sampling_mode']}, {len(coefficients)} weights)")
        return subexperiments, coefficients, total_subexp

    def _stage_execute(self, subexperiments, total_subexp, timings, metrics, artifacts, tag):
        with timed(timings, "ray_execution", tag, self.verbose):
            if self.agent is not None:
                results, info = self.agent.run_parallel(subexperiments, self.shots, task_multiplier=self.task_multiplier)
            else:
                results, info = self._run_local(subexperiments)

        t = info.get("timings", {})
        dur = info.get("task_duration", {})
        mean_dur = dur.get("mean")
        metrics.update(
            shots=self.shots,
            ray_num_tasks=info.get("num_tasks"),
            ray_chunk_size=info.get("chunk_size"),
            ray_cpus=info.get("total_cpus"),
            ray_nodes_used=info.get("nodes_used"),
            ray_est_cpu_utilization=info.get("est_cpu_utilization"),
            ray_cpu_efficiency=info.get("measured_cpu_efficiency"),
            ray_speedup_vs_serial=info.get("speedup_vs_serial"),
            ray_t_submit=t.get("submit"),
            ray_t_gather=t.get("execute_and_gather"),
            ray_t_wall=t.get("wall"),
            ray_task_min_s=dur.get("min"),
            ray_task_max_s=dur.get("max"),
            ray_task_mean_s=mean_dur,
            ray_load_imbalance=(dur["max"] / mean_dur) if dur and mean_dur else None,
            ray_per_node=json.dumps(info.get("per_node", {})),
            ray_per_fragment=json.dumps(info.get("per_fragment", {})),
            circuits_per_second=info.get("circuits_per_second"),
        )
        tasks = info.get("tasks") or []
        if tasks:
            artifacts["task_durations"] = np.array([x["duration"] for x in tasks], dtype=float)
            artifacts["task_hosts"] = np.array([x["host"] for x in tasks])
            artifacts["task_labels"] = np.array([x["label"] for x in tasks])
            artifacts["task_n_circuits"] = np.array([x["n_circuits"] for x in tasks], dtype=int)
        return results

    def _stage_reconstruct(self, results, coefficients, subobservables, paulis, timings, metrics, artifacts, tag):
        with timed(timings, "reconstruction", tag, self.verbose):
            expvals = reconstruct_expectation_values(results, coefficients, subobservables)
        cut_vals = [float(np.real(v)) for v in expvals]
        metrics["cutting_expval"] = cut_vals[0]                       # first observable (legacy column)
        metrics["cutting_expvals"] = _json_list(cut_vals)
        artifacts["reconstructed_expvals"] = np.array(cut_vals, dtype=float)
        if len(cut_vals) == 1:
            self._log(f"{tag}reconstructed <{pauli_name(paulis[0])}> = {cut_vals[0]:.6g}")
        else:
            self._log(f"{tag}reconstructed {len(cut_vals)} observables; first <{pauli_name(paulis[0])}> = {cut_vals[0]:.6g}")
        return cut_vals

    @staticmethod
    def _finalize(metrics, timings, run_start):
        timings["total"] = time.perf_counter() - run_start
        metrics.update({f"t_{k}": v for k, v in timings.items()})

    # --------------------------------------------------------
    # One run
    # --------------------------------------------------------
    def run(self, n_data=None):
        """Run one problem. n_data is required for callable circuits; for a fixed QuantumCircuit it
        defaults to circuit.num_qubits. Returns (metrics, artifacts)."""
        if n_data is None and not isinstance(self.circuit, QuantumCircuit):
            raise ValueError("n_data is required when circuit is a callable")
        if n_data is None:
            n_data = self.circuit.num_qubits
        tag = f"[N={n_data}] "
        timings, artifacts = {}, {}
        metrics = {
            "n_qubits": n_data,
            "n_observables": None,
            "actual_expval": None,
            "cutting_expval": None,
            "absolute_error": None,
            "relative_error_percent": None,
            "error_in_sigmas": None,
            "actual_run": False,
        }
        run_start = time.perf_counter()
        self._log(f"{tag}{'=' * 40}")
        self._log(f"{tag}start: {n_data} data qubits, max {self.qubits_per_subcircuit} qubits/subcircuit")

        tqc, n_data = self._stage_build_and_transpile(n_data, timings, metrics, tag)
        paulis = self._resolve_observables(n_data, tqc.num_qubits)
        metrics["n_observables"] = len(paulis)
        metrics["observable_labels"] = json.dumps([pauli_name(p) for p in paulis])
        self._log(f"{tag}observables: {len(paulis)}"
                  + (f" ({', '.join(pauli_name(p) for p in paulis[:6])}{', ...' if len(paulis) > 6 else ''})"))

        do_actual = self._should_run_actual(tqc.num_qubits)
        metrics["actual_run"] = do_actual
        actual_vals, actual_errs = None, None
        if do_actual:
            actual_vals, actual_errs = self._stage_actual(tqc, paulis, timings, metrics, artifacts, tag)
        else:
            self._log(f"{tag}actual (uncut) run skipped ({tqc.num_qubits} simulated qubits, "
                      f"run_actual={self.run_actual!r}, limit < {self.actual_max_qubits})")

        cut_qc, labels, cut_paulis = self._stage_partition(tqc, paulis, timings, metrics, artifacts, tag)
        subcircuits, subobservables, log10_terms = self._stage_cut(
            tqc, cut_qc, labels, cut_paulis, timings, metrics, tag)

        if metrics["gamma_exceeds_limit"]:
            self._log(f"{tag}WARNING: gamma exceeds {self.max_gamma}; reconstruction variance will be very large.")
            if self.skip_high_gamma:
                metrics["status"] = "skipped_gamma"
                self._finalize(metrics, timings, run_start)
                return metrics, artifacts

        if self.dry_run:
            metrics["status"] = "planned"
            self._finalize(metrics, timings, run_start)
            return metrics, artifacts

        subexperiments, coefficients, total_subexp = self._stage_generate(
            subcircuits, subobservables, log10_terms, timings, metrics, artifacts, tag)
        results = self._stage_execute(subexperiments, total_subexp, timings, metrics, artifacts, tag)
        cut_vals = self._stage_reconstruct(results, coefficients, subobservables, paulis, timings, metrics, artifacts, tag)

        # ---- error vs. reference + overhead comparison ----
        if do_actual and actual_vals is not None:
            errs = [abs(c - a) for c, a in zip(cut_vals, actual_vals)]
            rel = [e / abs(a) * 100.0 for e, a in zip(errs, actual_vals) if abs(a) > 1e-15]
            sig = [e / s if s > 0 else None for e, s in zip(errs, actual_errs)]
            sig_defined = [v for v in sig if v is not None]
            err = max(errs)                                            # worst observable
            metrics["absolute_error"] = err
            metrics["absolute_errors"] = _json_list(errs)
            metrics["mean_absolute_error"] = float(np.mean(errs))
            metrics["relative_error_percent"] = max(rel) if rel else None
            metrics["error_in_sigmas"] = max(sig_defined) if sig_defined else None
            metrics["error_in_sigmas_list"] = _json_list(sig)
            pipeline = sum(timings.get(k, 0.0) for k in
                           ("cut_circuit", "generate_experiments", "ray_execution", "reconstruction"))
            metrics["t_cutting_pipeline"] = pipeline
            metrics["cutting_vs_actual_time_ratio"] = pipeline / timings["actual_execution"] if timings["actual_execution"] > 0 else None
            ratio = metrics["cutting_vs_actual_time_ratio"]
            ratio_txt = f"{ratio:.2f}x" if ratio is not None else "n/a"
            self._log(f"{tag}comparison: first actual={actual_vals[0]:.6g}, first cutting={cut_vals[0]:.6g}, "
                      f"max abs_err={err:.4g} over {len(errs)} obs, cutting/actual time={ratio_txt}")

        self._finalize(metrics, timings, run_start)
        self._log(f"{tag}{'total':<26}: {timings['total']:10.3f} s")
        metrics["status"] = "ok"

        artifacts["timings_keys"] = np.array(list(timings.keys()))
        artifacts["timings_values"] = np.array(list(timings.values()), dtype=float)
        return metrics, artifacts

    # --------------------------------------------------------
    # Sweeps / comparison
    # --------------------------------------------------------
    def sweep(self, min_qubits=16, max_qubits=20, step=2):
        if isinstance(self.circuit, QuantumCircuit):
            raise TypeError("sweep() needs a callable circuit (n_data -> QuantumCircuit), not a fixed QuantumCircuit")
        return self.run_sweep(list(range(min_qubits, max_qubits + 1, step)))

    def run_sweep(self, qubit_values):
        sweep_start = time.perf_counter()
        self.all_metrics = []
        if self.save:
            self._create_output_dir()
            self._save_config(qubit_values)

        for n_data in qubit_values:
            try:
                metrics, artifacts = self.run(n_data)
                if self.save:
                    self._save_artifacts(n_data, artifacts)
            except Exception as exc:
                traceback.print_exc()
                log(f"[N={n_data}] FAILED: {exc}")
                metrics = {"n_qubits": n_data, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            self.all_metrics.append(metrics)
            if self.save:
                self._save_metrics()                    # incremental

        self.print_summary(time.perf_counter() - sweep_start)
        return self.all_metrics

    def compare_partitioners(self, n_data=None, partitioners=None):
        """Plan-only (no execution) comparison of partitioners for one size. Returns list of metrics."""
        partitioners = partitioners or (list(self.PARTITIONERS) + ["joint"])   # "joint" = cut_strategy="joint"
        saved = (self.partitioner, self.cut_strategy, self.dry_run, self.run_actual, self.save)
        rows = []
        try:
            self.dry_run, self.run_actual, self.save = True, False, False
            for name in partitioners:
                if name == "joint":
                    self.cut_strategy = "joint"
                else:
                    self.cut_strategy, self.partitioner = "gate", name
                try:
                    metrics, _ = self.run(n_data)
                except Exception as exc:
                    metrics = {"n_qubits": n_data, "partitioner": name, "status": "failed", "error": str(exc)}
                rows.append(metrics)
        finally:
            self.partitioner, self.cut_strategy, self.dry_run, self.run_actual, self.save = saved

        print("\n" + "=" * 80)
        print(f"PARTITIONER COMPARISON (N={n_data})")
        print(f"{'partitioner':>14} {'frags':>6} {'cuts':>6} {'log10g':>8} {'max_q':>6} {'t_part(s)':>10}")
        for m in rows:
            if m.get("status") == "failed":
                print(f"{m['partitioner']:>14} FAILED: {m.get('error')}")
                continue
            print(f"{m['partitioner']:>14} {m['n_fragments']:>6} {m['n_cuts']:>6} {m['log10_gamma']:>8.2f} "
                  f"{m['max_subcircuit_qubits']:>6} {m['t_partition']:>10.3f}")
        print("=" * 80)
        return rows

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------
    def _create_output_dir(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(self.output_dir, f"run_{stamp}")
        os.makedirs(os.path.join(self.out_dir, "artifacts"), exist_ok=True)

    def _save_config(self, qubit_values):
        config = {
            k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else repr(v))
            for k, v in vars(self).items()
            if k not in ("agent", "circuit", "all_metrics", "_gurobi_env", "out_dir")
        }
        config["circuit"] = getattr(self.circuit, "name", None) or type(self.circuit).__name__
        config["qubit_values"] = list(qubit_values)
        if self.agent is not None:
            config.update(
                ray_address=getattr(self.agent, "ray_address", None),
                ray_total_cpus=getattr(self.agent, "total_cpus", None),
                ray_nodes=len(getattr(self.agent, "ray_nodes", [])),
                ray_init_time=getattr(self.agent, "init_time", None),
                ray_cluster_start_time=getattr(self.agent, "cluster_start_time", None),
            )
        with open(os.path.join(self.out_dir, "config.json"), "w") as fh:
            json.dump(config, fh, indent=2, default=str)

    def _save_metrics(self):
        fieldnames = []
        for row in self.all_metrics:
            fieldnames.extend(k for k in row if k not in fieldnames)
        with open(os.path.join(self.out_dir, "metrics.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_metrics)
        arr = np.empty(len(self.all_metrics), dtype=object)
        for i, row in enumerate(self.all_metrics):
            arr[i] = row
        np.save(os.path.join(self.out_dir, "metrics.npy"), arr, allow_pickle=True)

    def _save_artifacts(self, n_data, artifacts):
        if artifacts:
            np.savez_compressed(os.path.join(self.out_dir, "artifacts", f"n{n_data}.npz"), **artifacts)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    @staticmethod
    def _fmt(value, width=10, prec=5):
        return f"{'-':>{width}}" if value is None else f"{value:>{width}.{prec}f}"

    def print_summary(self, total_time=None):
        f = self._fmt
        print("\n" + "=" * SEP_WIDTH)
        print("SUMMARY  (actual/cutting = first observable; abs_err & sigmas = max over observables)")
        print("=" * SEP_WIDTH)
        print(f"{'N':>4} {'obs':>4} {'status':>13} {'actual':>10} {'cutting':>10} {'abs_err':>10} {'sigmas':>7} "
              f"{'frags':>6} {'cuts':>5} {'log10g':>7} {'subexp':>8} {'ray(s)':>8} {'total(s)':>9}")
        for m in self.all_metrics:
            if m.get("status") not in OK_STATUSES:
                print(f"{m['n_qubits']:>4} {'':>4} {m.get('status', '?'):>13} {m.get('error', '')}")
                continue
            print(f"{m['n_qubits']:>4} {m.get('n_observables') or 1:>4} {m['status']:>13} "
                  f"{f(m.get('actual_expval'))} {f(m.get('cutting_expval'))} "
                  f"{f(m.get('absolute_error'))} {f(m.get('error_in_sigmas'), 7, 2)} "
                  f"{m['n_fragments']:>6} {m['n_cuts']:>5} {m['log10_gamma']:>7.1f} "
                  f"{m.get('total_subexperiments', 0):>8} {f(m.get('t_ray_execution'), 8, 2)} {f(m.get('t_total'), 9, 2)}")
        print("=" * SEP_WIDTH)
        if total_time is not None:
            print(f"Whole sweep time : {total_time:.2f} s")
        if self.out_dir:
            print(f"Results saved in : {self.out_dir}")
        print("=" * SEP_WIDTH)