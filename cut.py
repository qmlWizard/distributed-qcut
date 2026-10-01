import contextlib
import copy
import math
import os
import time
from collections import Counter
from functools import partial

import numpy as np
import pennylane as qml
from pennylane.measurements import MidMeasureMP
from pennylane.wires import Wires

CIRCUIT_DIR = "circuits/bchmrk_data"
MAX_QUBITS = 8          # max width of each partition / fragment
MAX_CUTS = 30            # safety guard: classical cost grows roughly as 8^cuts
NUM_QUBITS = 25          # width of the generated circuit (qft / grover)
ALGORITHM = "grover"     # "qft" or "grover"

STRICT_EQUAL = False     # False: sizes differ by at most 1 (e.g. 25 -> 9,8,8)
                         # True : all partitions exactly equal (25 -> 5x5); needs a divisor of n
MAX_EXPANDED_TAPES = 200_000   # guard: abort before expanding absurdly large fragments
EXEC_CHUNK = 64          # tapes per qml.execute call (for progress reporting)
DRAW_LIMIT = 150         # don't ASCII-draw tapes with more ops than this (drawing is slow)


# --------------------------------------------------------------------------- #
# Timing / progress helpers
# --------------------------------------------------------------------------- #
_T0 = time.perf_counter()
STAGE_TIMES = []  # (stage name, seconds)


def ts():
    """Seconds since program start, for log lines."""
    return f"+{time.perf_counter() - _T0:8.1f}s"


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


@contextlib.contextmanager
def stage(name):
    """Print START/DONE lines and record the stage duration."""
    log(f">>> START: {name}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        STAGE_TIMES.append((name, dt))
        log(f"<<< DONE : {name}  ({dt:.2f}s)")


def print_timing_summary():
    print("\n" + "=" * 80)
    print("TIMING SUMMARY (slowest first)")
    print("=" * 80)
    total = sum(t for _, t in STAGE_TIMES) or 1e-9
    for name, dt in sorted(STAGE_TIMES, key=lambda x: -x[1]):
        print(f"{dt:10.2f}s  {100 * dt / total:5.1f}%  {name}")
    if STAGE_TIMES:
        worst = max(STAGE_TIMES, key=lambda x: x[1])
        print(f"\nBOTTLENECK: '{worst[0]}' ({worst[1]:.2f}s)")
    print(f"Total wall time: {time.perf_counter() - _T0:.2f}s")


# --------------------------------------------------------------------------- #
# Circuit generators
# --------------------------------------------------------------------------- #
def qft_circuit(num_qubits=NUM_QUBITS, approximation_degree=None, do_swaps=True,
                init_bitstring=None):
    """Quantum Fourier Transform on wires 0..num_qubits-1.

    Only queues operations (call it inside a QNode / make_qscript, or pass the
    function itself to `cut_circuit_horizontally`).

    Args:
        num_qubits: number of qubits.
        approximation_degree: if set, controlled-phase gates between qubits
            further apart than this are dropped (approximate QFT). This
            drastically reduces the number of wire cuts needed, because the
            exact QFT is all-to-all connected.
        do_swaps: apply the final qubit-reversal SWAPs.
        init_bitstring: optional bitstring (e.g. "0101") prepared before the QFT.
            Without it the QFT acts on |0...0> and returns a trivial uniform state.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")

    if init_bitstring is not None:
        if len(init_bitstring) != num_qubits:
            raise ValueError("init_bitstring length must equal num_qubits")
        for w, bit in enumerate(init_bitstring):
            if bit == "1":
                qml.PauliX(wires=w)

    for target in range(num_qubits):
        qml.Hadamard(wires=target)
        for control in range(target + 1, num_qubits):
            distance = control - target
            if approximation_degree is not None and distance > approximation_degree:
                continue
            qml.ControlledPhaseShift(np.pi / (2 ** distance), wires=[control, target])

    if do_swaps:
        for i in range(num_qubits // 2):
            qml.SWAP(wires=[i, num_qubits - i - 1])


def _multi_controlled_z(wires):
    """Z controlled on all-but-last wire (phase flip on |1...1>)."""
    wires = list(wires)
    if len(wires) == 1:
        qml.PauliZ(wires=wires[0])
    elif len(wires) == 2:
        qml.CZ(wires=wires)
    else:
        qml.ctrl(qml.PauliZ(wires=wires[-1]), control=wires[:-1])


def grover_circuit(num_qubits=NUM_QUBITS, marked_state=None, iterations=None):
    """Grover's search on wires 0..num_qubits-1.

    Args:
        num_qubits: number of search qubits.
        marked_state: bitstring of the marked item (wire 0 = first character).
            Defaults to all ones.
        iterations: number of Grover iterations. Defaults to the optimal
            floor(pi/4 * sqrt(2^n)).
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    marked_state = marked_state or "1" * num_qubits
    if len(marked_state) != num_qubits or set(marked_state) - {"0", "1"}:
        raise ValueError("marked_state must be a bitstring of length num_qubits")
    if iterations is None:
        iterations = max(1, int(math.floor(math.pi / 4 * math.sqrt(2 ** num_qubits))))

    wires = list(range(num_qubits))
    zero_wires = [w for w, b in enumerate(marked_state) if b == "0"]

    for w in wires:
        qml.Hadamard(wires=w)

    for _ in range(iterations):
        # Oracle: phase-flip the marked state
        for w in zero_wires:
            qml.PauliX(wires=w)
        _multi_controlled_z(wires)
        for w in zero_wires:
            qml.PauliX(wires=w)

        # Diffuser: reflection about the uniform superposition
        for w in wires:
            qml.Hadamard(wires=w)
            qml.PauliX(wires=w)
        _multi_controlled_z(wires)
        for w in wires:
            qml.PauliX(wires=w)
            qml.Hadamard(wires=w)


ALGORITHMS = {
    "qft": qft_circuit,
    "grover": grover_circuit,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_circuit(qasm_dir):
    files = sorted(os.listdir(qasm_dir))
    qasm_files = [f for f in files if f.endswith(".qasm") and "_transpiled" not in f]
    if not qasm_files:
        raise FileNotFoundError(f"No .qasm files found in {qasm_dir}")
    qasm_file = qasm_files[0]
    print(f"Loading: {os.path.join(qasm_dir, qasm_file)}")
    try:
        with open(os.path.join(qasm_dir, qasm_file), "r") as f:
            qasm = f.read()
        # measurements=[] drops the terminal `measure` statements of the QASM file
        # (qcut can't handle mid-circuit measurements; we add our own expval below).
        return qml.from_qasm(qasm, measurements=[])
    except Exception as e:
        print(f"Failed to load {qasm_file}: {type(e).__name__}: {e}")
        return None


def get_circuits(size="small", num=1):
    path = os.path.join(CIRCUIT_DIR, size)
    circuits = sorted(f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)))
    circuits = circuits[: min(num, len(circuits))]
    return [load_circuit(os.path.join(path, d)) for d in circuits]


def _decompose_to_two_qubit(tape):
    """Decompose every gate acting on >2 wires (e.g. multi-controlled Z in Grover).

    qcut can only cut wires *between* gates, so a k-qubit gate forces a fragment
    of at least k wires. Decomposing keeps every node at <= 2 wires.

    Each distinct big gate is decomposed ONCE and the result is reused (Grover
    repeats the same multi-controlled Z every iteration), with progress prints.
    """
    big = [op for op in tape.operations if len(op.wires) > 2]
    log(f"  gates before decomposition: {len(tape.operations)}  "
        f"(gates on >2 wires: {len(big)}, widest: {max((len(o.wires) for o in big), default=0)})")
    if not big:
        return tape

    two_qubit_set = lambda op: len(op.wires) <= 2
    cache = {}
    new_ops = []
    n_big_seen = 0
    for op in tape.operations:
        if len(op.wires) <= 2:
            new_ops.append(op)
            continue
        n_big_seen += 1
        key = (op.name, tuple(op.wires), repr(op.data), repr(op.hyperparameters))
        if key not in cache:
            t0 = time.perf_counter()
            log(f"  decomposing NEW big gate #{n_big_seen}/{len(big)}: "
                f"{op.name} on {len(op.wires)} wires ...")
            mini = qml.tape.QuantumTape([op], [qml.expval(qml.Z(op.wires[0]))])
            tapes, _ = qml.transforms.decompose(mini, gate_set=two_qubit_set)
            cache[key] = list(tapes[0].operations)
            log(f"    -> {len(cache[key])} gates in {time.perf_counter() - t0:.2f}s")
        else:
            log(f"  big gate #{n_big_seen}/{len(big)} ({op.name}): reusing cached decomposition")
        new_ops.extend(copy.copy(o) for o in cache[key])

    log(f"  gates after decomposition: {len(new_ops)}")
    return qml.tape.QuantumTape(new_ops, tape.measurements)


def get_tape(circuit, observable_wires=None, decompose=True):
    """Build a tape that qcut can work with.

    qcut needs (a) no mid-circuit measurements / barriers and (b) at least one
    expval of a Pauli word as the terminal measurement. Default observable: Z on
    every wire. `circuit` is any callable that queues operations.
    """
    with stage("queue circuit gates (make_qscript)"):
        qscript = qml.tape.make_qscript(circuit)()
        log(f"  queued operations: {len(qscript.operations)}")

    with stage("filter ops + build tape/observable"):
        ops = [op for op in qscript.operations
               if not isinstance(op, (MidMeasureMP, qml.Barrier))]
        all_wires = sorted(Wires.all_wires([op.wires for op in ops]).labels)
        obs_wires = list(observable_wires) if observable_wires is not None else all_wires
        obs = (qml.prod(*[qml.Z(w) for w in obs_wires])
               if len(obs_wires) > 1 else qml.Z(obs_wires[0]))
        tape = qml.tape.QuantumTape(ops, [qml.expval(obs)])
        log(f"  wires: {len(all_wires)}, ops: {len(ops)}")

    if decompose:
        with stage("decompose gates to <=2 qubits"):
            tape = _decompose_to_two_qubit(tape)
    return tape


def print_tape(tape, title="CIRCUIT", max_ops_listed=20):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if len(tape.operations) <= DRAW_LIMIT:
        try:
            print(qml.drawer.tape_text(tape, decimals=3))
        except Exception:
            print(tape)
    else:
        print(f"(drawing skipped: {len(tape.operations)} ops > DRAW_LIMIT={DRAW_LIMIT})")
    print(f"\nWires       : {list(tape.wires)}")
    print(f"Operations  : {len(tape.operations)}")
    print(f"Measurements: {len(tape.measurements)}")
    print("\nOperations:")
    for i, op in enumerate(tape.operations[:max_ops_listed]):
        print(f"{i:4d}: {op.name:<20} wires={list(op.wires)}")
    if len(tape.operations) > max_ops_listed:
        print(f"  ... ({len(tape.operations) - max_ops_listed} more)")


# --------------------------------------------------------------------------- #
# Partitioning
# --------------------------------------------------------------------------- #
def calculate_partitions(wires, max_qubits=MAX_QUBITS, strict_equal=STRICT_EQUAL):
    """Split wires into the fewest partitions of width <= max_qubits, as equal as possible.

    Default (balanced): sizes differ by at most 1, e.g. 25 wires, max 10 -> 9, 8, 8.
    strict_equal=True : every partition has exactly the same size; the number of
        partitions is raised until it divides len(wires) (25 wires, max 10 -> 5 x 5).
        For a prime number of wires this degenerates to size-1 partitions.
    """
    if max_qubits <= 0:
        raise ValueError("max_qubits must be greater than zero")
    wires = list(wires)
    n = len(wires)
    k = math.ceil(n / max_qubits)
    if strict_equal:
        while n % k:
            k += 1
    base, extra = divmod(n, k)

    partitions, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        partitions.append(wires[start : start + size])
        start += size
    boundaries = [p[-1] for p in partitions[:-1]]
    return partitions, boundaries


def print_partitions(partitions, boundaries):
    print("\n" + "=" * 80)
    print("HORIZONTAL PARTITIONS")
    print("=" * 80)
    for i, wires in enumerate(partitions):
        print(f"Partition {i} ({len(wires)} qubits): wires={wires}")
    print(f"Boundaries: {boundaries}")


def find_cross_partition_operations(tape, partitions, max_listed=30):
    wire_to_part = {w: i for i, ws in enumerate(partitions) for w in ws}
    crossing = []
    for index, op in enumerate(tape.operations):
        if len(op.wires) < 2:
            continue
        ids = sorted({wire_to_part[w] for w in op.wires})
        if len(ids) > 1:
            crossing.append((index, op, ids))
    print("\n" + "=" * 80)
    print("CROSS-PARTITION OPERATIONS")
    print("=" * 80)
    for index, op, frags in crossing[:max_listed]:
        print(f"{index:4d}: {op.name:<20} wires={list(op.wires)} partitions={frags}")
    if len(crossing) > max_listed:
        print(f"  ... ({len(crossing) - max_listed} more)")
    print(f"Total crossing operations: {len(crossing)}")
    return crossing


# --------------------------------------------------------------------------- #
# Cut placement
# --------------------------------------------------------------------------- #
def assign_nodes_to_partitions(graph, partitions):
    """Assign every graph node to a partition.

    A two-qubit gate between wires of different partitions cannot be cut, so the
    gate goes to ONE partition and the other qubit's wire is cut before (and
    after) it. Rules:
      * multi-qubit op  -> the partition where most of its qubits currently live
      * single-qubit op -> stays wherever its qubit currently lives (no extra cut)
      * measurement     -> stays where its qubit currently lives (qcut forbids
                           cutting a wire right before a measurement)
    """
    wire_to_part = {w: i for i, ws in enumerate(partitions) for w in ws}
    nodes = sorted(graph.nodes, key=lambda n: graph.nodes[n]["order"])
    assignment = {}
    step = max(1, len(nodes) // 10)

    for i, node in enumerate(nodes):
        if i % step == 0:
            log(f"  assigning nodes: {i}/{len(nodes)}")
        obj = node.obj
        pred_on_wire = {}
        for u, _, _, data in graph.in_edges(node, keys=True, data=True):
            pred_on_wire[data["wire"]] = u

        def current_location(w):
            return assignment[pred_on_wire[w]] if w in pred_on_wire else wire_to_part[w]

        locations = [current_location(w) for w in obj.wires]
        counts = Counter(locations)
        # most common location, ties -> lowest partition id
        assignment[node] = max(sorted(counts), key=lambda p: counts[p])

    return assignment


def find_wire_cut_edges(graph, partitions, max_listed=50):
    with stage("assign graph nodes to partitions"):
        assignment = assign_nodes_to_partitions(graph, partitions)

    with stage("collect cut edges"):
        # (node1, node2, key) triples, exactly what qml.qcut.place_wire_cuts expects
        cut_edges = [(u, v, k) for u, v, k in graph.edges(keys=True)
                     if assignment[u] != assignment[v]]

    print("\n" + "=" * 80)
    print("WIRE CUTS")
    print("=" * 80)
    name = lambda n: n.obj.name if hasattr(n.obj, "name") else type(n.obj).__name__
    for i, (u, v, k) in enumerate(cut_edges[:max_listed]):
        wire = graph.get_edge_data(u, v, k)["wire"]
        print(f"Cut {i}: wire={wire} | {name(u)} -> {name(v)}")
    if len(cut_edges) > max_listed:
        print(f"  ... ({len(cut_edges) - max_listed} more)")
    print(f"Total cuts: {len(cut_edges)}  (rough reconstruction cost ~ 8^{len(cut_edges)} "
          f"= {8 ** len(cut_edges):.3e})")

    return cut_edges


def apply_qcut(tape, partitions, strategy="manual"):
    with stage("tape -> graph (qcut.tape_to_graph)"):
        graph = qml.qcut.tape_to_graph(tape)
        log(f"  graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

    with stage("find cross-partition operations"):
        crossing_operations = find_cross_partition_operations(tape, partitions)

    if strategy == "auto":
        # KaHyPar-based automatic cutting (pip install kahypar); enforces the width limit.
        with stage("auto cut search (KaHyPar find_and_place_cuts)"):
            max_qubits = max(len(p) for p in partitions)
            cut_graph = qml.qcut.find_and_place_cuts(
                graph=graph,
                cut_strategy=qml.qcut.CutStrategy(max_free_wires=max_qubits),
                replace_wire_cuts=True,
            )
        return cut_graph, crossing_operations

    cut_edges = find_wire_cut_edges(graph, partitions)
    if len(cut_edges) > MAX_CUTS:
        raise RuntimeError(
            f"{len(cut_edges)} wire cuts needed (> MAX_CUTS={MAX_CUTS}); reconstruction "
            "cost grows exponentially. Try strategy='auto', fewer qubits, an approximate "
            "QFT (approximation_degree), fewer Grover iterations, or a different partitioning."
        )
    if not cut_edges:
        return None, crossing_operations

    with stage("place wire cuts"):
        cut_graph = qml.qcut.place_wire_cuts(graph, cut_edges)
    with stage("replace WireCut nodes with Measure/Prepare nodes"):
        qml.qcut.replace_wire_cut_nodes(cut_graph)  # in-place
    return cut_graph, crossing_operations


def create_fragments(cut_graph):
    with stage("fragment_graph"):
        fragments, communication_graph = qml.qcut.fragment_graph(cut_graph)
        log(f"  fragments: {len(fragments)}")
    with stage("fragments -> tapes (graph_to_tape)"):
        fragment_tapes = []
        for i, f in enumerate(fragments):
            fragment_tapes.append(qml.qcut.graph_to_tape(f))
            log(f"  fragment {i}: {len(fragment_tapes[-1].operations)} ops, "
                f"{len(fragment_tapes[-1].wires)} wires")
    return fragment_tapes, communication_graph


def print_fragments(fragment_tapes, max_qubits=MAX_QUBITS):
    print("\n" + "#" * 80)
    print("PENNYLANE FRAGMENTS")
    print("#" * 80)
    print(f"Number of fragments: {len(fragment_tapes)}")
    for i, fragment in enumerate(fragment_tapes):
        print_tape(fragment, f"FRAGMENT {i}")
        if len(fragment.wires) > max_qubits:
            print(f"WARNING: fragment {i} uses {len(fragment.wires)} wires (> {max_qubits}); "
                  "cut wires add extra wires on top of the partition width")


# --------------------------------------------------------------------------- #
# Expansion / execution / reconstruction
# --------------------------------------------------------------------------- #
def _count_cut_nodes(fragment):
    n_prep = sum(isinstance(op, qml.qcut.PrepareNode) for op in fragment.operations)
    n_meas = sum(isinstance(op, qml.qcut.MeasureNode) for op in fragment.operations)
    return n_prep, n_meas


def expand_fragments(fragment_tapes):
    expanded_tapes, prepare_nodes, measure_nodes = [], [], []

    print("\n" + "=" * 80)
    print("EXPANDING FRAGMENTS")
    print("=" * 80)

    # Predict sizes first: each fragment expands into 4^(#prepare) * 3^(#measure) tapes.
    predicted = []
    for i, fragment in enumerate(fragment_tapes):
        n_prep, n_meas = _count_cut_nodes(fragment)
        pred = (4 ** n_prep) * (3 ** n_meas)
        predicted.append(pred)
        log(f"  fragment {i}: {n_prep} prepare, {n_meas} measure -> predicted {pred:,} expanded tapes")
    log(f"  predicted total expanded tapes: {sum(predicted):,}")
    if max(predicted, default=0) > MAX_EXPANDED_TAPES:
        raise RuntimeError(
            f"A fragment would expand into {max(predicted):,} tapes "
            f"(> MAX_EXPANDED_TAPES={MAX_EXPANDED_TAPES:,}). Reduce the number of cuts per "
            "fragment (fewer iterations, approximate QFT, better partitioning)."
        )

    for fragment_id, fragment in enumerate(fragment_tapes):
        with stage(f"expand fragment {fragment_id}"):
            tapes, prep, meas = qml.qcut.expand_fragment_tape(fragment)
            log(f"  Fragment {fragment_id}: {len(tapes)} expanded tapes, "
                f"{len(prep)} prepare nodes, {len(meas)} measure nodes")
        if len(tapes) == 0:
            raise RuntimeError(f"Fragment {fragment_id} generated zero expanded tapes")
        expanded_tapes.append(tapes)
        prepare_nodes.append(prep)
        measure_nodes.append(meas)

    return expanded_tapes, prepare_nodes, measure_nodes


def execute_expanded_fragments(expanded_tapes, device_name="default.qubit", shots=None):
    all_results = []

    print("\n" + "=" * 80)
    print("EXECUTING EXPANDED FRAGMENTS")
    print("=" * 80)

    for fragment_id, tapes in enumerate(expanded_tapes):
        with stage(f"execute fragment {fragment_id} ({len(tapes)} tapes)"):
            # Fragment wire labels are NOT necessarily 0..n-1 (graph_to_tape allocates
            # new labels), so the device must be built from the actual labels.
            wires = Wires.all_wires([t.wires for t in tapes])
            log(f"  building device '{device_name}' with {len(wires)} wires")
            device = qml.device(device_name, wires=wires)

            if shots is not None:
                tapes = [t.copy(shots=shots) for t in tapes]

            results = []
            t0 = time.perf_counter()
            report_every = max(EXEC_CHUNK, len(tapes) // 20)
            next_report = report_every
            for i in range(0, len(tapes), EXEC_CHUNK):
                results.extend(qml.execute(tapes[i : i + EXEC_CHUNK], device, diff_method=None))
                done = len(results)
                if done >= next_report or done == len(tapes):
                    el = time.perf_counter() - t0
                    eta = el / done * (len(tapes) - done)
                    log(f"  fragment {fragment_id}: {done}/{len(tapes)} tapes "
                        f"({el:.1f}s elapsed, ETA {eta:.1f}s)")
                    next_report += report_every
            all_results.append(results)

    return all_results


def reconstruct(results, communication_graph, prepare_nodes, measure_nodes):
    print("\n" + "=" * 80)
    print("PENNYLANE RECONSTRUCTION")
    print("=" * 80)

    if len(results) == 0:
        raise RuntimeError("No fragment results were produced")

    # qcut_processing_fn expects ONE flat sequence: fragment 0's tapes, then
    # fragment 1's tapes, ... (not a list of lists).
    with stage("flatten fragment results"):
        flat_results = [r for fragment_results in results for r in fragment_results]
        log(f"  total results: {len(flat_results)}")

    with stage("qcut_processing_fn (tensor contraction / reconstruction)"):
        return qml.qcut.qcut_processing_fn(
            flat_results, communication_graph, prepare_nodes, measure_nodes
        )


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def cut_circuit_horizontally(circuit, max_qubits=MAX_QUBITS, strategy="manual",
                             strict_equal=STRICT_EQUAL, observable_wires=None):
    """`circuit` must be a callable that queues operations (NOT its return value)."""
    tape = get_tape(circuit, observable_wires=observable_wires)
    print_tape(tape, "ORIGINAL CIRCUIT")

    with stage("calculate partitions"):
        partitions, boundaries = calculate_partitions(list(tape.wires), max_qubits, strict_equal)
    print(f"\nTotal qubits : {len(tape.wires)}")
    print(f"MAX_QUBITS   : {max_qubits}")
    print(f"Partitions   : {len(partitions)} (sizes {[len(p) for p in partitions]})")
    print_partitions(partitions, boundaries)

    cut_graph, crossing_operations = apply_qcut(tape, partitions, strategy)

    if cut_graph is None:
        print("\nNo cuts needed; circuit already fits.")
        return {
            "original_tape": tape, "partitions": partitions, "boundaries": boundaries,
            "cut_graph": None, "fragments": [tape], "communication_graph": None,
            "crossing_operations": crossing_operations,
        }

    fragment_tapes, communication_graph = create_fragments(cut_graph)
    with stage("print fragments"):
        print_fragments(fragment_tapes, max_qubits)

    return {
        "original_tape": tape,
        "partitions": partitions,
        "boundaries": boundaries,
        "cut_graph": cut_graph,
        "fragments": fragment_tapes,
        "communication_graph": communication_graph,
        "crossing_operations": crossing_operations,
    }


def execute_and_reconstruct(result, device_name="default.qubit", shots=None):
    if result["communication_graph"] is None:  # nothing was cut
        with stage("execute uncut circuit"):
            device = qml.device(device_name, wires=result["original_tape"].wires)
            value = qml.execute([result["original_tape"]], device, diff_method=None)[0]
        return {"reconstructed": value}

    expanded_tapes, prepare_nodes, measure_nodes = expand_fragments(result["fragments"])
    results = execute_expanded_fragments(expanded_tapes, device_name, shots)
    reconstructed = reconstruct(
        results, result["communication_graph"], prepare_nodes, measure_nodes
    )
    return {
        "expanded_tapes": expanded_tapes,
        "prepare_nodes": prepare_nodes,
        "measure_nodes": measure_nodes,
        "fragment_results": results,
        "reconstructed": reconstructed,
    }


def uncut_reference(tape, device_name="default.qubit"):
    with stage("uncut reference simulation"):
        device = qml.device(device_name, wires=tape.wires)
        return qml.execute([tape], device, diff_method=None)[0]


if __name__ == "__main__":
    try:
        if ALGORITHM == "qft":
            # Approximate QFT keeps the number of wire cuts manageable.
            circuit = partial(
                qft_circuit,
                num_qubits=NUM_QUBITS,
                approximation_degree=2,
                do_swaps=True,
                init_bitstring="01" * (NUM_QUBITS // 2) + "0" * (NUM_QUBITS % 2),
            )
        elif ALGORITHM == "grover":
            circuit = partial(
                grover_circuit,
                num_qubits=NUM_QUBITS,
                marked_state="1" * NUM_QUBITS,
                iterations=10,  # optimal count; lower it to reduce cuts
            )
        else:
            raise ValueError(f"Unknown ALGORITHM: {ALGORITHM}")

        # Or load a QASM benchmark instead:
        # circuits = [c for c in get_circuits(size="medium", num=1) if c is not None]
        # circuit = circuits[0]

        log(f"Config: ALGORITHM={ALGORITHM}, NUM_QUBITS={NUM_QUBITS}, MAX_QUBITS={MAX_QUBITS}, "
            f"MAX_CUTS={MAX_CUTS}, STRICT_EQUAL={STRICT_EQUAL}")

        result = cut_circuit_horizontally(circuit, max_qubits=MAX_QUBITS, strategy="manual")
        print("Circuit is cut horizontally !!", flush=True)
        final_result = execute_and_reconstruct(result, device_name="default.qubit", shots=None)

        print("\n" + "=" * 80)
        print("FINAL RECONSTRUCTED RESULT")
        print("=" * 80)
        print(final_result["reconstructed"])

        # Sanity check against the uncut circuit (only feasible for small widths)
        if len(result["original_tape"].wires) <= 20:
            print("Uncut reference:", uncut_reference(result["original_tape"]))
    finally:
        # Printed even if a stage raises or you Ctrl-C, so you can see where time went.
        print_timing_summary()