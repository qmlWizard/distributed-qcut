"""
Joint-cutting cut placement (Frohler et al., "Scalable Circuit Cutting: A Framework for Combined Gate
and Wire Cuts Using Gate Groups", 2026).

Two-stage Kernighan-Lin-style heuristic:

  Stage 1 (interaction graph)    : k-way partition of the qubits -> gate cuts only.
  Stage 2 (circuit topology graph): every two-qubit gate gets one vertex per qubit wire. Wire segments may
                                    change fragment between consecutive gates (= wire cut). Refines the
                                    stage-1 partition, so it is never worse than stage 1 (on the cost model).

Edge weights are log10(kappa): gate cut log10(kappa_g) (cx: 3), wire cut log10(4).
Gate groups (cascades on shared control/target, parallel gates in one layer) are partition-dependent cost
functions instead of edges:
    cascade (any number of crossing gates > 0)   : kappa = 3
    parallel, n crossing gates                    : kappa = 2^(n+1) - 1
Balance constraint = number of qubit wires (segments) per fragment <= max_per.

The result is a circuit in which every wire cut is an explicit qiskit_addon_cutting `Move`, plus fragment labels
for the new, wider circuit, ready for `partition_problem`.
"""

import math
import random
from collections import Counter, defaultdict, namedtuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_addon_cutting.instructions import Move

LOG3 = math.log10(3.0)
LOG_WIRE = math.log10(4.0)
EPS = 1e-12

Gate2 = namedtuple("Gate2", "gid instr name qubits logk")   # qubits = (control, target) for cx


# ============================================================
# Circuit analysis
# ============================================================
def gate_kappa(op):
    """Minimal gate-cut kappa of a two-qubit gate (no classical communication)."""
    try:
        theta = float(op.params[0]) if op.params else None
    except (TypeError, ValueError):
        theta = None
    if op.name == "swap":
        return 7.0
    if theta is not None and op.name in ("rzz", "rxx", "ryy", "rzx"):
        return 1.0 + 2.0 * abs(math.sin(theta))
    if theta is not None and op.name in ("crx", "cry", "crz", "cp"):
        return 1.0 + 2.0 * abs(math.sin(theta / 2))
    return 3.0


def two_qubit_gates(qc):
    gates = []
    for i, inst in enumerate(qc.data):
        if len(inst.qubits) != 2 or inst.operation.name == "barrier":
            continue
        qs = tuple(qc.find_bit(q).index for q in inst.qubits)
        gates.append(Gate2(len(gates), i, inst.operation.name, qs, math.log10(gate_kappa(inst.operation))))
    return gates


def find_gate_groups(qc, gates):
    """Automatic detection of cascades (shared control / shared target, adjacent on the wire) and parallel
    CNOTs (same DAG layer). Every gate belongs to at most one group. Returns [(kind, [gid, ...]), ...]."""
    by_instr = {g.instr: g for g in gates}
    wire_ops = defaultdict(list)
    depth = defaultdict(int)
    layer_of = {}
    for i, inst in enumerate(qc.data):
        qs = [qc.find_bit(q).index for q in inst.qubits]
        layer = 1 + max((depth[q] for q in qs), default=0)
        for q in qs:
            depth[q] = layer
            wire_ops[q].append(i)
        if i in by_instr:
            layer_of[by_instr[i].gid] = layer

    cascades = []
    for q, seq in wire_ops.items():
        for role, kind in ((0, "cascade_control"), (1, "cascade_target")):
            run, partners = [], set()
            for idx in seq + [None]:
                g = by_instr.get(idx) if idx is not None else None
                if g is not None and g.name == "cx" and g.qubits[role] == q and g.qubits[1 - role] not in partners:
                    run.append(g.gid)
                    partners.add(g.qubits[1 - role])     # partners must differ so the gates commute
                else:
                    if len(run) >= 2:
                        cascades.append((kind, run))
                    run, partners = [], set()
                    if g is not None and g.name == "cx" and g.qubits[role] == q:
                        run, partners = [g.gid], {g.qubits[1 - role]}

    groups, used = [], set()
    for kind, run in sorted(cascades, key=lambda c: -len(c[1])):
        seg = []
        for gid in run + [None]:
            if gid is not None and gid not in used:
                seg.append(gid)
            else:
                if len(seg) >= 2:
                    groups.append((kind, seg))
                    used.update(seg)
                seg = []

    layers = defaultdict(list)
    for g in gates:
        if g.name == "cx" and g.gid not in used:
            layers[layer_of[g.gid]].append(g.gid)
    for ids in layers.values():
        if len(ids) >= 2:
            groups.append(("parallel", ids))
    return groups


def _group_cost_fn(kind):
    if kind.startswith("cascade"):
        return lambda n: LOG3 if n > 0 else 0.0
    return lambda n: math.log10(2 ** (n + 1) - 1) if n > 0 else 0.0


# ============================================================
# Balance-constraint models
# ============================================================
class _SizeWidths:
    """Stage 1: fragment width = number of qubits."""

    def __init__(self, n_frag, cap, labels):
        self.cap, self.w = cap, [0] * n_frag
        for lab in labels.values():
            self.w[lab] += 1

    def delta(self, labels, v, f):
        return {labels[v]: -1, f: 1}

    def fits(self, d):
        return all(self.w[k] + dv <= self.cap for k, dv in d.items() if dv > 0)

    def apply(self, d):
        for k, dv in d.items():
            self.w[k] += dv


class _RunWidths:
    """Stage 2: fragment width = number of maximal same-label runs along the wires (+ idle qubits)."""

    def __init__(self, wires, base, n_frag, cap, labels):
        self.wires, self.cap = wires, cap
        self.pos = {v: (q, i) for q, seq in wires.items() for i, v in enumerate(seq)}
        self.w = list(base)
        for seq in wires.values():
            for j in range(len(seq)):
                s = self._start(labels, seq, -1, j, None)
                if s is not None:
                    self.w[s] += 1

    @staticmethod
    def _start(labels, seq, i, j, lab_v):
        """Label of the run that starts at position j (None if j continues a run); position i is hypothetically lab_v."""
        if j >= len(seq):
            return None
        lab_j = lab_v if j == i else labels[seq[j]]
        if j == 0:
            return lab_j
        prev = lab_v if j - 1 == i else labels[seq[j - 1]]
        return lab_j if lab_j != prev else None

    def delta(self, labels, v, f):
        q, i = self.pos[v]
        seq, old, d = self.wires[q], labels[v], Counter()
        for j in (i, i + 1):
            before, after = self._start(labels, seq, i, j, old), self._start(labels, seq, i, j, f)
            if before is not None:
                d[before] -= 1
            if after is not None:
                d[after] += 1
        return d

    def fits(self, d):
        return all(self.w[k] + dv <= self.cap for k, dv in d.items() if dv > 0)

    def apply(self, d):
        for k, dv in d.items():
            self.w[k] += dv


# ============================================================
# KL / FM-style refinement on the exact objective
# ============================================================
class _PartitionState:
    """cost = sum_{(u,v,w) differing} w  +  sum_groups cost_fn(#crossing pairs). Moves are evaluated exactly
    (only the terms touching the moved vertex are recomputed)."""

    def __init__(self, vertices, pair_terms, groups, n_frag):
        self.vertices, self.n_frag, self.groups = list(vertices), n_frag, groups
        self.pair_terms = [t for t in pair_terms if t[2] != 0 and t[0] != t[1]]
        adj = defaultdict(lambda: defaultdict(float))
        for u, v, w in self.pair_terms:
            adj[u][v] += w
            adj[v][u] += w
        self.adj = {v: list(nb.items()) for v, nb in adj.items()}
        self.vgroups = defaultdict(list)
        for gi, (pairs, _) in enumerate(groups):
            for x in {x for pair in pairs for x in pair}:
                self.vgroups[x].append(gi)
        self.labels, self.widths = None, None

    def bind(self, labels, widths):
        self.labels, self.widths = labels, widths

    def _group_cost(self, gi):
        pairs, fn = self.groups[gi]
        L = self.labels
        return fn(sum(1 for u, v in pairs if L[u] != L[v]))

    def total_cost(self):
        L = self.labels
        return (sum(w for u, v, w in self.pair_terms if L[u] != L[v])
                + sum(self._group_cost(gi) for gi in range(len(self.groups))))

    def delta_move(self, v, f):
        L, old = self.labels, self.labels[v]
        d = 0.0
        for u, w in self.adj.get(v, ()):
            d += w * ((f != L[u]) - (old != L[u]))
        for gi in self.vgroups.get(v, ()):
            before = self._group_cost(gi)
            L[v] = f
            after = self._group_cost(gi)
            L[v] = old
            d += after - before
        return d

    def _apply(self, v, new):
        self.widths.apply(self.widths.delta(self.labels, v, new))
        self.labels[v] = new

    def optimize(self, max_passes=10, patience=15, swaps=False):
        for _ in range(max_passes):
            if self._pass(patience, swaps) <= EPS:
                break
        return self.total_cost()

    def _pass(self, patience, swaps):
        L = self.labels
        locked, hist = set(), []
        cum = best = 0.0
        best_len = 0
        while len(locked) < len(self.vertices):
            unlocked = [v for v in self.vertices if v not in locked]
            cand = None
            for v in unlocked:
                old = L[v]
                for f in range(self.n_frag):
                    if f == old or not self.widths.fits(self.widths.delta(L, v, f)):
                        continue
                    d = self.delta_move(v, f)
                    if cand is None or d < cand[0] - EPS:
                        cand = (d, ((v, f),))
            if swaps:
                for ia, a in enumerate(unlocked):
                    for b in unlocked[ia + 1:]:
                        la, lb = L[a], L[b]
                        if la == lb:
                            continue
                        d = self.delta_move(a, lb)
                        L[a] = lb
                        d += self.delta_move(b, la)
                        L[a] = la
                        if cand is None or d < cand[0] - EPS:
                            cand = (d, ((a, lb), (b, la)))
            if cand is None:
                break
            d, moves = cand
            undo = []
            for v, new in moves:
                undo.append((v, L[v]))
                self._apply(v, new)
                locked.add(v)
            hist.append(undo)
            cum += d
            if cum < best - EPS:
                best, best_len = cum, len(hist)
            elif len(hist) - best_len >= patience:
                break
        for undo in reversed(hist[best_len:]):          # roll back to the best prefix
            for v, old in reversed(undo):
                self._apply(v, old)
        return -best


# ============================================================
# Building the cut circuit
# ============================================================
def _build_cut_circuit(tqc, gates, labels, wires, idle_labels):
    """Insert an explicit Move (= wire cut) wherever a wire changes fragment. Returns
    (circuit, fragment label per qubit of the new circuit, final qubit of each original qubit, n_wire_cuts)."""
    n = tqc.num_qubits
    n_cuts = sum(labels[a] != labels[b] for seq in wires.values() for a, b in zip(seq, seq[1:]))
    out = QuantumCircuit(n + n_cuts, name=getattr(tqc, "name", None))
    out.global_phase = tqc.global_phase

    seg = {q: (labels[wires[q][0]] if q in wires else idle_labels[q]) for q in range(n)}
    frag = [None] * (n + n_cuts)
    for q in range(n):
        frag[q] = seg[q]
    cur, nxt = list(range(n)), n
    by_instr = {g.instr: g for g in gates}

    for i, inst in enumerate(tqc.data):
        g = by_instr.get(i)
        if g is not None:
            for q in g.qubits:
                lab = labels[(g.gid, q)]
                if lab != seg[q]:
                    out.append(Move(), [cur[q], nxt])
                    frag[nxt], cur[q], seg[q] = lab, nxt, lab
                    nxt += 1
        out.append(inst.operation, [cur[tqc.find_bit(q).index] for q in inst.qubits])
    return out, frag, cur, n_cuts


def expand_paulis(paulis, final_qubit, n_total):
    """Move every observable onto the last wire segment of each original qubit."""
    out = []
    for p in paulis:
        z, x = np.zeros(n_total, dtype=bool), np.zeros(n_total, dtype=bool)
        for q, fq in enumerate(final_qubit):
            z[fq], x[fq] = p.z[q], p.x[q]
        out.append(Pauli((z, x, p.phase)))
    return out


# ============================================================
# Public entry point
# ============================================================
def plan_joint_cuts(tqc, max_per, *, use_wire_cuts=True, use_gate_groups=True,
                    runs=10, max_passes=10, patience=15, seed=0):
    """Returns (cut_circuit, fragment labels per cut_circuit qubit, final qubit per original qubit, info)."""
    n = tqc.num_qubits
    n_frag = math.ceil(n / max_per)
    gates = two_qubit_gates(tqc)
    by_gid = {g.gid: g for g in gates}
    group_specs = find_gate_groups(tqc, gates) if use_gate_groups else []
    in_group = {gid for _, ids in group_specs for gid in ids}

    def make_groups(vertex):
        return [([(vertex(by_gid[i], by_gid[i].qubits[0]), vertex(by_gid[i], by_gid[i].qubits[1])) for i in ids],
                 _group_cost_fn(kind)) for kind, ids in group_specs]

    # ---- stage 1: interaction graph, gate cuts only ----
    if n_frag == 1:
        labels1, cost1 = {q: 0 for q in range(n)}, 0.0
    else:
        rng = random.Random(seed)
        pairs1 = [(g.qubits[0], g.qubits[1], g.logk) for g in gates if g.gid not in in_group]
        st1 = _PartitionState(range(n), pairs1, make_groups(lambda g, q: q), n_frag)
        best = None
        for _ in range(runs):
            perm = list(range(n))
            rng.shuffle(perm)
            labels = {q: i // max_per for i, q in enumerate(perm)}
            st1.bind(labels, _SizeWidths(n_frag, max_per, labels))
            cost = st1.optimize(max_passes, patience, swaps=True)
            if best is None or cost < best[0] - EPS:
                best = (cost, dict(st1.labels))
            if cost < EPS:
                break
        cost1, labels1 = best

    # ---- stage 2: circuit topology graph, adds wire cuts ----
    wires = defaultdict(list)
    for g in gates:
        for q in g.qubits:
            wires[q].append((g.gid, q))
    labels2 = {v: labels1[v[1]] for seq in wires.values() for v in seq}
    cost2 = cost1
    if use_wire_cuts and n_frag > 1 and cost1 > EPS:
        pairs2 = [((g.gid, g.qubits[0]), (g.gid, g.qubits[1]), g.logk) for g in gates if g.gid not in in_group]
        for seq in wires.values():
            pairs2.extend((a, b, LOG_WIRE) for a, b in zip(seq, seq[1:]))
        base = [0] * n_frag
        for q in range(n):
            if q not in wires:
                base[labels1[q]] += 1
        st2 = _PartitionState([v for s in wires.values() for v in s], pairs2,
                              make_groups(lambda g, q: (g.gid, q)), n_frag)
        st2.bind(labels2, _RunWidths(wires, base, n_frag, max_per, labels2))
        cost2 = st2.optimize(max_passes, patience, swaps=False)
        labels2 = st2.labels

    idle = {q: labels1[q] for q in range(n) if q not in wires}
    cut_qc, frag, final_q, n_wire_cuts = _build_cut_circuit(tqc, gates, labels2, wires, idle)

    crossing = [g for g in gates if labels2[(g.gid, g.qubits[0])] != labels2[(g.gid, g.qubits[1])]]
    info = dict(
        joint_stage1_log10_kappa=cost1,
        joint_stage2_log10_kappa=cost2,
        joint_est_log10_kappa=cost2,                                   # with joint cutting of groups
        joint_indiv_log10_kappa=sum(g.logk for g in crossing) + n_wire_cuts * LOG_WIRE,   # all cuts individual
        joint_gate_cuts=len(crossing),
        joint_wire_cuts=n_wire_cuts,
        joint_n_groups=len(group_specs),
        joint_grouped_gates=len(in_group),
    )
    return cut_qc, frag, final_q, info