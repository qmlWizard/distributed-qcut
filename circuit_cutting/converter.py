"""
converter.py: convert Cirq / PennyLane circuits into Qiskit QuantumCircuits.

    from converter import convert, from_cirq, from_pennylane, circuit_factory

Every converter returns a ConvertedCircuit:

    conv.circuit          Qiskit circuit (may contain free Parameters)
    conv.parameters       ORDERED list of Qiskit Parameters (this order defines bind() vectors)
    conv.bind(values)     -> fully bound circuit (list / array / {name: value})
    conv.bind_args(*a)    -> PennyLane only: bind using the same arguments as the quantum function
    conv.observables      list of SparsePauliOp (PennyLane expval(...) terms / Cirq `observables=`)
    conv.qubit_map        source qubit / wire -> Qiskit qubit index
    conv.to_cutting_circuit(values)   bound + measurement-free circuit for CircuitCutting

Parameterised circuits (VQE / QAOA)
-----------------------------------
* Cirq: sympy symbols become Qiskit Parameters (expressions such as 2*gamma*w are kept).
  Pass `param_values=` to bind some/all of them at conversion time.
* PennyLane has no symbolic parameters, so a *quantum function / QNode* plus example
  arguments is traced: every float / float-array argument becomes Parameters (named after
  the argument, e.g. gammas[0], betas[1]). Gate angles are recovered as affine functions of
  these inputs (after decomposition, so qml.qaoa layers, PauliRot, etc. work) and the result
  is validated at a random point. Non-affine dependence raises ConversionError; use
  parametrize=False to get a plain bound circuit instead.
* Circuit structure does not depend on the parameter values, so convert ONCE and call
  bind() inside the optimiser loop.

Not supported: classical control, mid-circuit measurement, noise channels, non-Pauli
observables. Observables are evaluated at the example arguments (their coefficients are not
traced). Circuits for CircuitCutting must be measurement-free.
"""

import cmath
import inspect
import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, SparsePauliOp


class ConversionError(Exception):
    pass


# ============================================================
# Result container
# ============================================================

@dataclass
class ConvertedCircuit:
    circuit: QuantumCircuit
    parameters: list = field(default_factory=list)
    source: str = "qiskit"
    qubit_map: dict = field(default_factory=dict)
    observables: list = field(default_factory=list)
    default_values: object = None            # np.ndarray or None (PennyLane example arguments)
    notes: list = field(default_factory=list)
    _flatten_args: object = field(default=None, repr=False)

    @property
    def is_parameterized(self):
        return bool(self.parameters)

    @property
    def parameter_names(self):
        return [p.name for p in self.parameters]

    @property
    def num_qubits(self):
        return self.circuit.num_qubits

    def _mapping(self, values):
        if isinstance(values, dict):
            by_name = {p.name: p for p in self.parameters}
            out = {}
            for key, val in values.items():
                if isinstance(key, Parameter):
                    p = key
                else:
                    if str(key) not in by_name:
                        raise ConversionError(f"Unknown parameter {str(key)!r}; known: {list(by_name)}")
                    p = by_name[str(key)]
                out[p] = float(val)
            return out
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size != len(self.parameters):
            raise ConversionError(f"Expected {len(self.parameters)} values (order: {self.parameter_names}), got {arr.size}.")
        if not np.all(np.isfinite(arr)):
            raise ConversionError("Parameter values must be finite.")
        return dict(zip(self.parameters, arr.tolist()))

    def bind(self, values=None, strict=True):
        """Return a new circuit with parameters assigned. values=None uses the example
        arguments (PennyLane) or requires an already-bound circuit. strict=False allows partial binding."""
        if values is None:
            if not self.parameters:
                return self.circuit.copy()
            if self.default_values is None:
                raise ConversionError(f"No values given for parameters {self.parameter_names}.")
            values = self.default_values
        qc = self.circuit.assign_parameters(self._mapping(values), inplace=False, strict=False)
        if strict and qc.parameters:
            raise ConversionError(f"Unbound parameters remain: {[p.name for p in qc.parameters]}")
        return qc

    def bind_args(self, *args):
        """PennyLane only: bind with the same positional arguments as the converted function."""
        if self._flatten_args is None:
            raise ConversionError("bind_args() is only available for circuits converted from a PennyLane function.")
        return self.bind(self._flatten_args(args))

    def to_cutting_circuit(self, values=None):
        """Bound, measurement-free circuit ready for CircuitCutting."""
        qc = self.bind(values)
        ops = qc.count_ops()
        if ops.get("measure") or ops.get("reset"):
            raise ConversionError("Circuit contains measure/reset; convert with measurements='drop' for circuit cutting.")
        return qc


# ============================================================
# Shared helpers
# ============================================================

def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _unique(names):
    seen, out = {}, []
    for n in names:
        k = seen.get(n, 0)
        seen[n] = k + 1
        out.append(n if k == 0 else f"{n}_{k}")
    return out


def _add_phase(qc, x):
    if isinstance(x, (int, float)) and abs(x) < 1e-15:
        return
    qc.global_phase = qc.global_phase + x


def _sympy_to_qiskit(expr, pmap):
    """sympy expression -> float or Qiskit ParameterExpression (symbols looked up in pmap by name)."""
    import sympy

    if isinstance(expr, (int, float, np.integer, np.floating)):
        return float(expr)
    if expr.is_number:
        v = complex(expr)
        if abs(v.imag) > 1e-12:
            raise ConversionError(f"Complex-valued parameter expression: {expr}")
        return v.real
    if isinstance(expr, sympy.Symbol):
        if expr.name not in pmap:
            raise ConversionError(f"Symbol {expr.name!r} is not a known parameter.")
        return pmap[expr.name]
    rec = lambda e: _sympy_to_qiskit(e, pmap)
    if isinstance(expr, sympy.Add):
        out = rec(expr.args[0])
        for a in expr.args[1:]:
            out = out + rec(a)
        return out
    if isinstance(expr, sympy.Mul):
        out = rec(expr.args[0])
        for a in expr.args[1:]:
            out = out * rec(a)
        return out
    if isinstance(expr, sympy.Pow):
        return rec(expr.args[0]) ** rec(expr.args[1])
    funcs = {"sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp", "log": "log",
             "asin": "arcsin", "acos": "arccos", "atan": "arctan", "Abs": "abs"}
    fname = type(expr).__name__
    if fname in funcs and len(expr.args) == 1:
        try:
            return getattr(rec(expr.args[0]), funcs[fname])()
        except AttributeError:
            raise ConversionError(f"Cannot convert symbolic function {fname} to a Qiskit expression.")
    raise ConversionError(f"Unsupported symbolic expression: {expr}")


def _pauli_op(terms, qidx, n):
    """terms: iterable of (coeff, {source_qubit: 'X'|'Y'|'Z'}) -> SparsePauliOp."""
    labels, coeffs = [], []
    for coeff, paulis in terms:
        label = ["I"] * n
        for q, p in paulis.items():
            if q not in qidx:
                raise ConversionError(f"Observable acts on {q!r}, which is not in the circuit.")
            label[n - 1 - qidx[q]] = p
        labels.append("".join(label))
        coeffs.append(complex(coeff))
    if not labels:
        labels, coeffs = ["I" * n], [0.0]
    return SparsePauliOp(labels, coeffs)


def _verify(conv, reference, max_qubits):
    """Compare the Qiskit unitary (bound at example + random values) against the source unitary
    (big-endian, hence reverse_qargs), up to global phase."""
    n = conv.circuit.num_qubits
    if n > max_qubits:
        conv.notes.append(f"verification skipped ({n} qubits > verify_max_qubits={max_qubits})")
        return
    if conv.parameters:
        rng = np.random.default_rng(7)
        points = ([np.asarray(conv.default_values, float)] if conv.default_values is not None else [])
        points.append(rng.uniform(-math.pi, math.pi, len(conv.parameters)))
    else:
        points = [np.zeros(0)]
    for v in points:
        got = Operator(conv.bind(v) if conv.parameters else conv.circuit)
        want = Operator(np.asarray(reference(v))).reverse_qargs()
        if not got.equiv(want, atol=1e-7):
            raise ConversionError(f"Verification failed: Qiskit circuit differs from the source circuit at values {v.tolist()}.")
    conv.notes.append(f"verified against source unitary at {len(points)} point(s)")


# ============================================================
# Cirq
# ============================================================

class _CirqEmitter:
    def __init__(self, qc, qidx, pmap, measurements):
        self.qc, self.qidx, self.pmap, self.measurements = qc, qidx, pmap, measurements
        self.dropped = 0
        self._cregs = {}

    def _exp(self, t):
        import sympy
        if isinstance(t, sympy.Basic) and not t.is_number:
            return True, _sympy_to_qiskit(t, self.pmap)
        return False, float(t)

    def emit(self, op, depth=0):
        import cirq

        if depth > 40:
            raise ConversionError("Operation decomposition too deep.")
        op = op.untagged
        gate = op.gate
        try:
            qs = [self.qidx[q] for q in op.qubits]
        except KeyError as exc:
            raise ConversionError(f"Operation {op!r} uses a qubit outside the circuit (ancilla?): {exc}")
        qc = self.qc

        if isinstance(gate, cirq.MeasurementGate):
            self._measure(op, qs)
            return
        if isinstance(gate, cirq.IdentityGate):
            return
        if isinstance(gate, cirq.GlobalPhaseGate):
            import sympy
            coeff = gate.coefficient
            if isinstance(coeff, sympy.Basic) and not coeff.is_number:
                raise ConversionError("Symbolic GlobalPhaseGate is not supported.")
            _add_phase(qc, cmath.phase(complex(coeff)))
            return
        if isinstance(gate, cirq.CSwapGate):
            qc.cswap(*qs)
            return

        t = getattr(gate, "exponent", None)
        if t is not None:
            sym, tv = self._exp(t)
            shift = float(getattr(gate, "_global_shift", 0.0))
            one = (not sym) and tv == 1 and shift == 0
            theta = math.pi * tv
            rot_phase = math.pi * (tv * (shift + 0.5))      # PowGate(t, g) = e^{i*pi*t*(g+1/2)} * R(pi*t)

            if isinstance(gate, cirq.XPowGate):
                qc.x(qs[0]) if one else (qc.rx(theta, qs[0]), _add_phase(qc, rot_phase))
                return
            if isinstance(gate, cirq.YPowGate):
                qc.y(qs[0]) if one else (qc.ry(theta, qs[0]), _add_phase(qc, rot_phase))
                return
            if isinstance(gate, cirq.ZPowGate):
                qc.z(qs[0]) if one else (qc.p(theta, qs[0]), _add_phase(qc, math.pi * (tv * shift)))
                return
            if isinstance(gate, cirq.CZPowGate):
                qc.cz(*qs) if one else (qc.cp(theta, *qs), _add_phase(qc, math.pi * (tv * shift)))
                return
            if isinstance(gate, cirq.ZZPowGate):
                qc.rzz(theta, *qs); _add_phase(qc, rot_phase)
                return
            if isinstance(gate, cirq.XXPowGate):
                qc.rxx(theta, *qs); _add_phase(qc, rot_phase)
                return
            if isinstance(gate, cirq.YYPowGate):
                qc.ryy(theta, *qs); _add_phase(qc, rot_phase)
                return
            if one:
                simple = [(cirq.HPowGate, qc.h), (cirq.CNotPowGate, qc.cx), (cirq.SwapPowGate, qc.swap),
                          (cirq.ISwapPowGate, qc.iswap), (cirq.CCXPowGate, qc.ccx), (cirq.CCZPowGate, qc.ccz)]
                for cls, fn in simple:
                    if isinstance(gate, cls):
                        fn(*qs)
                        return

        # ---- fallback: decompose, else dense unitary ----
        sub = cirq.decompose_once(op, default=None)
        if sub is not None:
            for s in sub:
                self.emit(s, depth + 1)
            return
        if cirq.has_unitary(op):
            qc.unitary(cirq.unitary(op), qs[::-1])          # Cirq is big-endian, Qiskit little-endian
            return
        raise ConversionError(f"Unsupported Cirq operation: {op!r}")

    def _measure(self, op, qs):
        if self.measurements == "error":
            raise ConversionError("Circuit contains measurements (measurements='error').")
        if self.measurements == "drop":
            self.dropped += 1
            return
        import cirq
        key = re.sub(r"\W", "_", cirq.measurement_key_name(op))
        name, k = f"c_{key}", 1
        while name in self._cregs:
            name, k = f"c_{key}_{k}", k + 1
        creg = ClassicalRegister(len(qs), name)
        self._cregs[name] = creg
        self.qc.add_register(creg)
        self.qc.measure(qs, creg)


def from_cirq(circuit, param_values=None, *, qubit_order=None, measurements="drop", observables=None,
              param_order=None, verify=False, verify_max_qubits=10, name=None):
    """
    circuit       cirq.Circuit / FrozenCircuit (sympy symbols -> Qiskit Parameters)
    param_values  {name_or_symbol: value}; bound at conversion (partial binding allowed)
    qubit_order   list of cirq qubits (default: sorted); position k -> Qiskit qubit k
    measurements  "drop" (default, needed for cutting) | "keep" | "error"
    observables   PauliString / PauliSum / list of them -> conv.observables (SparsePauliOps)
    param_order   explicit order of symbol names for conv.parameters (default: natural sort)
    verify        compare unitaries with Cirq (<= verify_max_qubits qubits)
    """
    import cirq

    if measurements not in ("drop", "keep", "error"):
        raise ValueError("measurements must be 'drop', 'keep' or 'error'")
    if isinstance(circuit, cirq.FrozenCircuit):
        circuit = circuit.unfreeze()
    if param_values:
        circuit = cirq.resolve_parameters(circuit, cirq.ParamResolver(dict(param_values)))

    order = cirq.QubitOrder.as_qubit_order(
        cirq.QubitOrder.DEFAULT if qubit_order is None else qubit_order).order_for(circuit.all_qubits())
    qidx = {q: i for i, q in enumerate(order)}

    found = set(cirq.parameter_names(circuit))
    names = list(param_order) if param_order is not None else sorted(found, key=_natural_key)
    if not found <= set(names):
        raise ConversionError(f"param_order is missing symbols: {sorted(found - set(names))}")
    thetas = [Parameter(n) for n in names]
    pmap = dict(zip(names, thetas))

    qc = QuantumCircuit(len(order), name=name or "cirq_circuit")
    em = _CirqEmitter(qc, qidx, pmap, measurements)
    for op in circuit.all_operations():
        em.emit(op)

    conv = ConvertedCircuit(circuit=qc, parameters=thetas, source="cirq", qubit_map=qidx)
    if em.dropped:
        conv.notes.append(f"dropped {em.dropped} measurement operation(s)")
        warnings.warn(f"Dropped {em.dropped} Cirq measurement(s) during conversion.")

    if observables is not None:
        items = observables if isinstance(observables, (list, tuple)) else [observables]
        sym = {cirq.X: "X", cirq.Y: "Y", cirq.Z: "Z"}
        for obs in items:
            terms = [(ps.coefficient, {q: sym[p] for q, p in ps.items()}) for ps in cirq.PauliSum.wrap(obs)]
            conv.observables.append(_pauli_op(terms, qidx, len(order)))

    if verify:
        if measurements == "keep" and em._cregs:
            conv.notes.append("verification skipped (circuit contains measurements)")
        else:
            nomeas = cirq.Circuit(o for o in circuit.all_operations() if not cirq.is_measurement(o))
            ref = lambda v: nomeas.unitary(qubit_order=order, qubits_that_should_be_present=order) if not names else \
                cirq.resolve_parameters(nomeas, dict(zip(names, v.tolist()))).unitary(
                    qubit_order=order, qubits_that_should_be_present=order)
            _verify(conv, ref, verify_max_qubits)
    return conv


# ============================================================
# PennyLane
# ============================================================

_PL_SUPPORTED = {
    "Hadamard", "PauliX", "PauliY", "PauliZ", "S", "T", "SX",
    "RX", "RY", "RZ", "PhaseShift", "Rot", "U1", "U2", "U3",
    "CNOT", "CZ", "CY", "SWAP", "ISWAP", "CRX", "CRY", "CRZ", "ControlledPhaseShift",
    "IsingXX", "IsingYY", "IsingZZ", "Toffoli", "CSWAP", "MultiRZ",
    "GlobalPhase", "QubitUnitary", "Identity", "Barrier",
}
_PL_SKIP = {"Identity", "Barrier"}


def _decompose_pl(tape, names):
    import pennylane as qml

    if all(op.name in names for op in tape.operations):
        return tape
    try:
        if hasattr(qml.transforms, "decompose"):
            tapes, _ = qml.transforms.decompose(tape, gate_set=set(names))
            return tapes[0]
        return tape.expand(depth=30, stop_at=lambda o: o.name in names)
    except Exception as exc:
        raise ConversionError(f"Could not decompose PennyLane operations into supported gates: {exc}")


def _as_param(p):
    arr = np.asarray(p)
    if arr.ndim == 0:
        v = complex(arr)
        if abs(v.imag) > 1e-12:
            raise ConversionError("Complex gate parameters are not supported.")
        return v.real
    return arr


def _pl_gates(tape):
    t = _decompose_pl(tape, _PL_SUPPORTED)
    gates = []
    for op in t.operations:
        if op.name in _PL_SKIP:
            continue
        if op.name not in _PL_SUPPORTED:
            raise ConversionError(f"Unsupported PennyLane operation after decomposition: {op.name}")
        gates.append((op.name, tuple(op.wires.tolist()), [_as_param(p) for p in op.parameters]))
    return gates


def _slots(gates):
    return np.array([p for _, _, ps in gates for p in ps if not isinstance(p, np.ndarray)], dtype=float)


def _same_structure(g0, g1):
    if len(g0) != len(g1):
        return False
    for (n0, w0, p0), (n1, w1, p1) in zip(g0, g1):
        if n0 != n1 or w0 != w1 or len(p0) != len(p1):
            return False
        for a, b in zip(p0, p1):
            if isinstance(a, np.ndarray) != isinstance(b, np.ndarray):
                return False
            if isinstance(a, np.ndarray) and not np.array_equal(a, b):
                return False
    return True


def _trace_affine(build, x0, gates0):
    """Recover every scalar gate angle as c + A @ x by finite differences, validated at a random point."""
    P0 = _slots(gates0)
    n, h = len(x0), 0.7
    A = np.zeros((len(P0), n))
    for k in range(n):
        x = x0.copy()
        x[k] += h
        g = _pl_gates(build(x))
        if not _same_structure(gates0, g):
            raise ConversionError("Circuit structure depends on the parameter values; use parametrize=False.")
        A[:, k] = (_slots(g) - P0) / h
    xr = x0 + np.random.default_rng(1234).uniform(-1.3, 1.3, n)
    gr = _pl_gates(build(xr))
    if not _same_structure(gates0, gr) or not np.allclose(_slots(gr), P0 + A @ (xr - x0), atol=1e-7):
        raise ConversionError("Gate angles are not affine in the input parameters; use parametrize=False "
                              "(bound circuit) or restructure the function.")
    return A, P0 - A @ x0


def _is_variable(a, i, variable_args):
    if variable_args is not None:
        return i in variable_args
    if getattr(a, "requires_grad", True) is False or isinstance(a, bool):
        return False
    if isinstance(a, (float, np.floating)):
        return True
    try:
        arr = np.asarray(a)
    except Exception:
        return False
    return arr.dtype.kind == "f" and arr.size > 0


def _pl_spec(args, arg_names, variable_args):
    flat, labels, spec = [], [], []
    for i, a in enumerate(args):
        nm = arg_names[i] if i < len(arg_names) else f"arg{i}"
        if _is_variable(a, i, variable_args):
            arr = np.asarray(a, dtype=float)
            if arr.ndim == 0:
                labels.append(nm)
            else:
                labels.extend(f"{nm}[{','.join(map(str, ix))}]" for ix in np.ndindex(arr.shape))
            spec.append(("var", i, arr.shape, len(flat), arr.size))
            flat.extend(arr.ravel().tolist())
        else:
            spec.append(("static", i, a))
    return np.array(flat, dtype=float), _unique(labels), spec


def _pl_rebuild(spec, x):
    out = []
    for s in spec:
        if s[0] == "static":
            out.append(s[2])
        else:
            _, _, shape, off, size = s
            seg = x[off:off + size]
            out.append(float(seg[0]) if shape == () else seg.reshape(shape).copy())
    return out


def _pl_flatten_values(spec, args):
    parts = []
    for s in spec:
        if s[0] == "var":
            arr = np.asarray(args[s[1]], dtype=float)
            if arr.shape != s[2]:
                raise ConversionError(f"Argument {s[1]} has shape {arr.shape}, expected {s[2]}.")
            parts.append(arr.ravel())
    return np.concatenate(parts) if parts else np.zeros(0)


def _emit_pl(qc, name, q, p):
    one = {"Hadamard": qc.h, "PauliX": qc.x, "PauliY": qc.y, "PauliZ": qc.z, "S": qc.s, "T": qc.t, "SX": qc.sx,
           "CNOT": qc.cx, "CZ": qc.cz, "CY": qc.cy, "SWAP": qc.swap, "ISWAP": qc.iswap,
           "Toffoli": qc.ccx, "CSWAP": qc.cswap}
    par = {"RX": qc.rx, "RY": qc.ry, "RZ": qc.rz, "PhaseShift": qc.p, "U1": qc.p,
           "CRX": qc.crx, "CRY": qc.cry, "CRZ": qc.crz, "ControlledPhaseShift": qc.cp,
           "IsingXX": qc.rxx, "IsingYY": qc.ryy, "IsingZZ": qc.rzz}
    if name in one:
        one[name](*q)
    elif name in par:
        par[name](p[0], *q)
    elif name == "Rot":                                   # Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)
        qc.rz(p[0], q[0]); qc.ry(p[1], q[0]); qc.rz(p[2], q[0])
    elif name == "U2":
        qc.u(math.pi / 2, p[0], p[1], q[0])
    elif name == "U3":
        qc.u(p[0], p[1], p[2], q[0])
    elif name == "MultiRZ":
        for a, b in zip(q[:-1], q[1:]):
            qc.cx(a, b)
        qc.rz(p[0], q[-1])
        for a, b in reversed(list(zip(q[:-1], q[1:]))):
            qc.cx(a, b)
    elif name == "GlobalPhase":                           # e^{-i phi}
        qc.global_phase = qc.global_phase - p[0]
    elif name == "QubitUnitary":                          # PennyLane is big-endian
        qc.unitary(p[0], q[::-1])
    else:
        raise ConversionError(f"No Qiskit mapping for {name}")


def _pl_observable(obs, widx, n):
    ps = obs.pauli_rep
    if ps is None:
        raise ConversionError(f"Observable {obs!r} is not a Pauli sum; only Pauli observables are supported.")
    return _pauli_op([(c, dict(pw.items())) for pw, c in ps.items()], widx, n)


def from_pennylane(obj, *args, parametrize=True, variable_args=None, wire_order=None,
                   verify=False, verify_max_qubits=10, name=None, **kwargs):
    """
    obj            QNode / quantum function (called as obj(*args, **kwargs)) or a QuantumTape/QuantumScript
    parametrize    True: float / float-array arguments become Qiskit Parameters (functions only)
                   False: plain bound circuit at the given arguments
    variable_args  indices of arguments to treat as parameters (default: float scalars/arrays,
                   skipping arguments with requires_grad=False)
    wire_order     list of PennyLane wires; position k -> Qiskit qubit k (default: sorted wires)
    kwargs         always static (passed to the function unchanged)
    """
    import pennylane as qml

    flatten_fn = None
    if isinstance(obj, qml.tape.QuantumScript):
        if args or kwargs:
            raise ConversionError("Arguments cannot be combined with a tape.")
        build, x0, labels, spec, parametrize = (lambda x: obj), np.zeros(0), [], [], False
    else:
        func = obj.func if isinstance(obj, qml.QNode) else obj
        try:
            arg_names = list(inspect.signature(func).parameters)
        except (TypeError, ValueError):
            arg_names = []
        x0, labels, spec = _pl_spec(args, arg_names, variable_args)
        if parametrize and len(x0):
            build = lambda x: qml.tape.make_qscript(func)(*_pl_rebuild(spec, x), **kwargs)
            flatten_fn = lambda a: _pl_flatten_values(spec, a)
        else:
            parametrize = False
            build = lambda x: qml.tape.make_qscript(func)(*args, **kwargs)

    tape0 = build(x0)
    wires = list(wire_order) if wire_order is not None else None
    if wires is None:
        wires = list(tape0.wires)
        try:
            wires = sorted(wires)
        except TypeError:
            pass
    missing = [w for w in tape0.wires if w not in wires]
    if missing:
        raise ConversionError(f"wire_order is missing wires: {missing}")
    widx = {w: i for i, w in enumerate(wires)}

    gates0 = _pl_gates(tape0)
    thetas, exprs = [], None
    if parametrize:
        A, c = _trace_affine(build, x0, gates0)
        thetas = [Parameter(n) for n in labels]
        exprs = []
        for a_row, c_k in zip(A, c):
            expr = None
            for a, th in zip(a_row, thetas):
                if abs(a) > 1e-12:
                    term = float(a) * th
                    expr = term if expr is None else expr + term
            if expr is None:
                exprs.append(float(c_k))
            else:
                exprs.append(expr + float(c_k) if abs(c_k) > 1e-12 else expr)

    qc = QuantumCircuit(len(wires), name=name or "pennylane_circuit")
    slot = 0
    for gname, gw, gp in gates0:
        params = []
        for p in gp:
            if isinstance(p, np.ndarray):
                params.append(p)
            else:
                params.append(exprs[slot] if exprs is not None else p)
                slot += 1
        _emit_pl(qc, gname, [widx[w] for w in gw], params)

    conv = ConvertedCircuit(circuit=qc, parameters=thetas, source="pennylane", qubit_map=widx,
                            default_values=x0 if parametrize else None, _flatten_args=flatten_fn)

    for m in tape0.measurements:
        if type(m).__name__ == "ExpectationMP" and getattr(m, "obs", None) is not None:
            conv.observables.append(_pl_observable(m.obs, widx, len(wires)))
        else:
            conv.notes.append(f"ignored measurement {type(m).__name__}")
    if conv.observables and parametrize:
        conv.notes.append("observables are evaluated at the example arguments (coefficients are not traced)")

    if verify:
        def ref(v):
            t = build(v if parametrize else x0)
            return qml.matrix(qml.tape.QuantumScript(t.operations), wire_order=wires)
        _verify(conv, ref, verify_max_qubits)
    return conv


# ============================================================
# Qiskit / QASM passthrough, dispatcher, factory
# ============================================================

def from_qiskit(circuit):
    return ConvertedCircuit(circuit=circuit, parameters=list(circuit.parameters), source="qiskit",
                            qubit_map={i: i for i in range(circuit.num_qubits)})


def from_qasm(text, name=None):
    if "OPENQASM 3" in text[:50]:
        from qiskit import qasm3
        qc = qasm3.loads(text)
    else:
        from qiskit import qasm2
        qc = qasm2.loads(text)
    if name:
        qc.name = name
    return from_qiskit(qc)


def _detect(obj):
    if isinstance(obj, QuantumCircuit):
        return "qiskit"
    if isinstance(obj, str):
        return "qasm"
    mod = type(obj).__module__.split(".")[0]
    if mod == "cirq":
        return "cirq"
    if mod == "pennylane" or callable(obj):
        return "pennylane"
    raise ConversionError(f"Cannot detect framework for object of type {type(obj)!r}")


def convert(obj, *args, framework=None, **kwargs):
    """Dispatch to the right converter. *args / **kwargs go to from_pennylane (function arguments)
    or **kwargs to from_cirq. framework: 'cirq' | 'pennylane' | 'qiskit' | 'qasm' (auto-detected)."""
    if isinstance(obj, ConvertedCircuit):
        return obj
    fw = framework or _detect(obj)
    if fw == "cirq":
        return from_cirq(obj, *args, **kwargs)
    if fw == "pennylane":
        return from_pennylane(obj, *args, **kwargs)
    if fw == "qiskit":
        return from_qiskit(obj)
    if fw == "qasm":
        return from_qasm(obj, **kwargs)
    raise ValueError(f"Unknown framework {fw!r}")


def circuit_factory(build, *, framework=None, param_values=None, **convert_kwargs):
    """Turn build(n) -> (Cirq circuit | PennyLane tape | QuantumCircuit | ConvertedCircuit) into the
    callable n -> bound, measurement-free QuantumCircuit that CircuitCutting.sweep() expects.
    param_values: values / dict, or callable n -> values (needed for parameterised circuits)."""
    def factory(n):
        conv = convert(build(n), framework=framework, **convert_kwargs)
        vals = param_values(n) if callable(param_values) else param_values
        return conv.to_cutting_circuit(vals)
    return factory