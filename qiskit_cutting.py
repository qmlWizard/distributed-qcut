import math
from fractions import Fraction

import numpy as np
import ray
import gurobipy as gp

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.containers import PrimitiveResult
from qiskit_aer.primitives import SamplerV2

from qiskit_addon_cutting import (
    expand_observables,
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)


# ============================================================
# SETTINGS
# ============================================================

N_QUBITS = 4

ITERATIONS = 1

PREFIX_QUBITS = 4

SHOTS = 2**14

# Maximum number of ORIGINAL circuit qubits per fragment.
#
# For example:
#
#     50 qubits
#     QUBITS_PER_SUBCIRCUIT = 12
#
# requires at least ceil(50 / 12) = 5 fragments.
#
QUBITS_PER_SUBCIRCUIT = 2


# ============================================================
# GUROBI SETTINGS
# ============================================================

GUROBI_TIME_LIMIT = 300

# 0.0 = prove optimality
#
# For large circuits you may want:
#
# GUROBI_MIP_GAP = 0.01
#
# which allows a 1% optimality gap.
#
GUROBI_MIP_GAP = 0.0

GUROBI_OUTPUT = True


# ============================================================
# CUTTING / SAMPLING SETTINGS
# ============================================================

MC_SAMPLES = 1000

# If number of generated QPD samples is small enough,
# enumerate everything exactly.
ENUMERATE_BELOW = 10_000

MAX_USABLE_GAMMA = 1_000


# ============================================================
# GROVER BUILDING BLOCKS
# ============================================================

def mcz(qc, data, ancillas):
    """
    Multi-controlled Z using a linear V-chain of Toffolis.

    The circuit uses clean ancillas.
    """

    controls = list(data[:-1])
    target = data[-1]

    if len(controls) == 1:
        qc.cz(
            controls[0],
            target,
        )
        return

    qc.h(target)

    qc.mcx(
        controls,
        target,
        ancilla_qubits=list(
            ancillas[: len(controls) - 2]
        ),
        mode="v-chain",
    )

    qc.h(target)


def oracle(
    qc,
    n,
    marked,
    ancillas,
):
    """
    Grover oracle.
    """

    zeros = [
        q
        for q in range(n)
        if marked[n - 1 - q] == "0"
    ]

    if zeros:
        qc.x(zeros)

    mcz(
        qc,
        list(range(n)),
        ancillas,
    )

    if zeros:
        qc.x(zeros)


def diffuser(
    qc,
    n,
    ancillas,
):
    """
    Grover diffusion operator.
    """

    qc.h(range(n))
    qc.x(range(n))

    mcz(
        qc,
        list(range(n)),
        ancillas,
    )

    qc.x(range(n))
    qc.h(range(n))


def grover(
    n,
    marked,
    iterations,
):
    """
    Build Grover circuit.
    """

    n_anc = max(
        0,
        n - 3,
    )

    qc = QuantumCircuit(
        n + n_anc
    )

    data = list(
        range(n)
    )

    ancillas = list(
        range(
            n,
            n + n_anc,
        )
    )

    qc.h(data)

    for _ in range(iterations):

        oracle(
            qc,
            n,
            marked,
            ancillas,
        )

        diffuser(
            qc,
            n,
            ancillas,
        )

    return qc


def optimal_iterations(n):
    """
    Theoretical optimal Grover iteration count.
    """

    return math.floor(
        math.pi
        / 4
        * 2.0 ** (n / 2)
    )


# ============================================================
# OBSERVABLE
# ============================================================

def prefix_projector(
    marked,
    m,
    num_qubits,
):
    """
    Projector onto the first m data qubits matching
    the marked string.

    Returns a SparsePauliOp.
    """

    n = len(marked)

    op = SparsePauliOp.from_sparse_list(
        [
            ("I", [0], 1.0)
        ],
        num_qubits=num_qubits,
    )

    for q in range(m):

        sign = (
            1
            if marked[n - 1 - q] == "0"
            else -1
        )

        factor = SparsePauliOp.from_sparse_list(
            [
                ("I", [q], 0.5),
                (
                    "Z",
                    [q],
                    0.5 * sign,
                ),
            ],
            num_qubits=num_qubits,
        )

        op = (
            op
            .dot(factor)
            .simplify()
        )

    return op


def analytic_prefix_probability(
    n,
    m,
    iterations,
):
    """
    Exact analytical probability.
    """

    theta = math.asin(
        2.0 ** (-n / 2)
    )

    phi = (
        2 * iterations + 1
    ) * theta

    frac = float(
        Fraction(
            2 ** (n - m) - 1,
            2**n - 1,
        )
    )

    return (
        math.sin(phi) ** 2
        +
        math.cos(phi) ** 2
        * frac
    )


# ============================================================
# CIRCUIT GRAPH
# ============================================================

def build_interaction_graph(
    circuit,
):
    """
    Build the weighted two-qubit interaction graph.

    Each qubit is a graph vertex.

    Every two-qubit gate contributes an edge.

    If the same pair of qubits interacts multiple times,
    its edge weight increases.

    Returns
    -------
    edge_weights:
        {
            (q1, q2): number_of_interactions
        }

    gate_records:
        list containing information about every
        two-qubit gate.
    """

    edge_weights = {}

    gate_records = []

    for gate_index, instruction in enumerate(
        circuit.data
    ):

        operation = instruction.operation

        qargs = instruction.qubits

        if len(qargs) != 2:
            continue

        q1 = circuit.find_bit(
            qargs[0]
        ).index

        q2 = circuit.find_bit(
            qargs[1]
        ).index

        if q1 == q2:
            continue

        edge = tuple(
            sorted(
                (
                    q1,
                    q2,
                )
            )
        )

        edge_weights[edge] = (
            edge_weights.get(
                edge,
                0,
            )
            + 1
        )

        gate_records.append(
            {
                "index": gate_index,
                "name": operation.name,
                "q1": q1,
                "q2": q2,
            }
        )

    return (
        edge_weights,
        gate_records,
    )


# ============================================================
# PRINT INTERACTION GRAPH
# ============================================================

def print_interaction_graph(
    circuit,
):
    """
    Print the interaction graph.
    """

    (
        edge_weights,
        gate_records,
    ) = build_interaction_graph(
        circuit
    )

    print()
    print(
        "=" * 70
    )
    print(
        "TWO-QUBIT INTERACTION GRAPH"
    )
    print(
        "=" * 70
    )

    print(
        f"Vertices : "
        f"{circuit.num_qubits}"
    )

    print(
        f"Edges    : "
        f"{len(edge_weights)}"
    )

    print()

    for edge, weight in sorted(
        edge_weights.items()
    ):

        print(
            f"q{edge[0]} <-> "
            f"q{edge[1]} "
            f"weight={weight}"
        )

    print(
        "=" * 70
    )


# ============================================================
# GUROBI OPTIMIZATION
# ============================================================

def optimize_partition_gurobi(
    circuit,
    max_qubits,
    time_limit=300,
    mip_gap=0.0,
    verbose=True,
):
    """
    Solve the quantum circuit partitioning problem.

    MIP formulation
    ----------------

    y[q,f] = 1
        if qubit q belongs to fragment f.

    x[e] = 1
        if interaction edge e crosses fragments.

    Objective
    ---------

        minimize

            sum_e w_e * x_e

    where w_e is the number of times the two qubits
    interact.

    Constraints
    -----------

    Every qubit belongs to exactly one fragment.

        sum_f y[q,f] = 1

    Fragment capacity:

        sum_q y[q,f] <= max_qubits

    Cross-fragment edge:

        x[e] >= |y[q1,f] - y[q2,f]|

    The resulting partition is supplied directly to
    Qiskit Addon Cutting through partition_labels.

    """

    num_qubits = circuit.num_qubits

    (
        edge_weights,
        gate_records,
    ) = build_interaction_graph(
        circuit
    )

    # --------------------------------------------------------
    # Minimum number of fragments
    # --------------------------------------------------------

    num_fragments = math.ceil(
        num_qubits
        / max_qubits
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if max_qubits <= 0:
        raise ValueError(
            "max_qubits must be > 0"
        )

    if max_qubits >= num_qubits:

        # No cutting is necessary.
        return {
            q: 0
            for q in range(
                num_qubits
            )
        }, {
            "partition": {
                0: list(
                    range(
                        num_qubits
                    )
                )
            },
            "cut_edges": [],
            "cut_gate_indices": [],
            "objective": 0.0,
            "num_fragments": 1,
            "model": None,
        }

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = gp.Model(
        "Gurobi_Quantum_Circuit_Cutting"
    )

    model.Params.TimeLimit = (
        time_limit
    )

    model.Params.MIPGap = (
        mip_gap
    )

    model.Params.OutputFlag = (
        1 if verbose else 0
    )

    # --------------------------------------------------------
    # y[q,f]
    # --------------------------------------------------------

    y = {}

    for q in range(
        num_qubits
    ):

        for f in range(
            num_fragments
        ):

            y[q, f] = model.addVar(
                vtype=gp.GRB.BINARY,
                name=(
                    f"assign_q{q}_f{f}"
                ),
            )

    # --------------------------------------------------------
    # x[q1,q2]
    #
    # x = 1 when the interaction crosses fragments.
    # --------------------------------------------------------

    x = {}

    for q1, q2 in edge_weights:

        x[q1, q2] = model.addVar(
            vtype=gp.GRB.BINARY,
            name=(
                f"cut_q{q1}_q{q2}"
            ),
        )

    model.update()

    # ========================================================
    # CONSTRAINT 1
    #
    # Every qubit belongs to exactly one fragment.
    # ========================================================

    for q in range(
        num_qubits
    ):

        model.addConstr(
            gp.quicksum(
                y[q, f]
                for f in range(
                    num_fragments
                )
            )
            == 1,
            name=(
                f"one_fragment_q{q}"
            ),
        )

    # ========================================================
    # CONSTRAINT 2
    #
    # Fragment capacity.
    # ========================================================

    for f in range(
        num_fragments
    ):

        model.addConstr(
            gp.quicksum(
                y[q, f]
                for q in range(
                    num_qubits
                )
            )
            <= max_qubits,
            name=(
                f"capacity_f{f}"
            ),
        )

    # ========================================================
    # SYMMETRY BREAKING
    #
    # Without this, fragment labels can be permuted.
    #
    # For example:
    #
    #   [0,1] [2,3]
    #
    # and
    #
    #   [2,3] [0,1]
    #
    # are mathematically identical.
    # ========================================================

    model.addConstr(
        y[0, 0] == 1,
        name="symmetry_break_q0",
    )

    # ========================================================
    # CONSTRAINT 3
    #
    # If q1 and q2 are in different fragments,
    # x[q1,q2] must become 1.
    #
    # x >= y[q1,f] - y[q2,f]
    #
    # x >= y[q2,f] - y[q1,f]
    # ========================================================

    for q1, q2 in edge_weights:

        for f in range(
            num_fragments
        ):

            model.addConstr(
                x[q1, q2]
                >=
                y[q1, f]
                -
                y[q2, f],
                name=(
                    f"cross_a_"
                    f"{q1}_{q2}_{f}"
                ),
            )

            model.addConstr(
                x[q1, q2]
                >=
                y[q2, f]
                -
                y[q1, f],
                name=(
                    f"cross_b_"
                    f"{q1}_{q2}_{f}"
                ),
            )

    # ========================================================
    # OBJECTIVE
    # ========================================================
    #
    # Minimize the number/weight of non-local interactions.
    #
    # If q0 and q1 interact 20 times, separating them costs
    # 20 in the objective.
    #
    # This is much better than merely minimizing the number
    # of distinct edges.
    #
    # ========================================================

    objective = gp.quicksum(
        edge_weights[q1, q2]
        * x[q1, q2]
        for q1, q2 in edge_weights
    )

    model.setObjective(
        objective,
        gp.GRB.MINIMIZE,
    )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "GUROBI OPTIMIZATION"
    )
    print(
        "=" * 70
    )

    print(
        f"Qubits             : "
        f"{num_qubits}"
    )

    print(
        f"Max qubits/fragment: "
        f"{max_qubits}"
    )

    print(
        f"Fragments          : "
        f"{num_fragments}"
    )

    print(
        f"Interaction edges   : "
        f"{len(edge_weights)}"
    )

    print(
        f"Time limit          : "
        f"{time_limit}s"
    )

    print()

    model.optimize()

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if model.Status == gp.GRB.INFEASIBLE:

        raise RuntimeError(
            "Gurobi reports the circuit "
            "partitioning model as infeasible."
        )

    if model.SolCount == 0:

        raise RuntimeError(
            "Gurobi did not find a feasible "
            "partition."
        )

    # --------------------------------------------------------
    # Recover partition
    # --------------------------------------------------------

    partition = {
        f: []
        for f in range(
            num_fragments
        )
    }

    for q in range(
        num_qubits
    ):

        for f in range(
            num_fragments
        ):

            if (
                y[q, f].X
                > 0.5
            ):

                partition[f].append(
                    q
                )

    # Remove empty fragments.
    partition = {
        f: qs
        for f, qs in partition.items()
        if len(qs) > 0
    }

    # --------------------------------------------------------
    # Map qubit -> fragment
    # --------------------------------------------------------

    qubit_to_fragment = {}

    for f, qubits in (
        partition.items()
    ):

        for q in qubits:

            qubit_to_fragment[q] = f

    # --------------------------------------------------------
    # Find cut edges
    # --------------------------------------------------------

    cut_edges = []

    for q1, q2 in edge_weights:

        if (
            qubit_to_fragment[q1]
            !=
            qubit_to_fragment[q2]
        ):

            cut_edges.append(
                (
                    q1,
                    q2,
                )
            )

    # --------------------------------------------------------
    # Find actual gate indices crossing fragments
    # --------------------------------------------------------

    cut_gate_indices = []

    for record in gate_records:

        q1 = record["q1"]
        q2 = record["q2"]

        if (
            qubit_to_fragment[q1]
            !=
            qubit_to_fragment[q2]
        ):

            cut_gate_indices.append(
                record["index"]
            )

    # --------------------------------------------------------
    # Convert partition into labels
    #
    # Qiskit partition_problem accepts labels such as:
    #
    #     AABBC
    #
    # where each character identifies a fragment.
    #
    # --------------------------------------------------------

    labels = [
        None
        for _ in range(
            num_qubits
        )
    ]

    # Use letters first.
    # If there are >26 fragments, fall back to strings.
    alphabet = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
    )

    if len(partition) <= len(
        alphabet
    ):

        fragment_labels = {
            f: alphabet[i]
            for i, f in enumerate(
                sorted(
                    partition
                )
            )
        }

    else:

        fragment_labels = {
            f: f"F{f}"
            for f in partition
        }

    for f, qubits in (
        partition.items()
    ):

        label = fragment_labels[f]

        for q in qubits:

            labels[q] = label

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        "Gurobi status:",
        model.Status,
    )

    print(
        "Objective:",
        model.ObjVal,
    )

    print(
        "MIP gap:",
        model.MIPGap,
    )

    print()

    print(
        "OPTIMAL / BEST FOUND PARTITION"
    )

    for f in sorted(
        partition
    ):

        print(
            f"Fragment {fragment_labels[f]}: "
            f"{partition[f]} "
            f"-> {len(partition[f])} qubits"
        )

    print()

    print(
        "Cross-fragment edges:"
    )

    for edge in cut_edges:

        print(
            f"  q{edge[0]} <-> q{edge[1]}"
        )

    print()

    print(
        "Number of cross-fragment edges:",
        len(cut_edges),
    )

    print(
        "Number of cut two-qubit gates:",
        len(cut_gate_indices),
    )

    print(
        "Partition labels:",
        "".join(
            str(x)
            for x in labels
        ),
    )

    print(
        "=" * 70
    )

    return labels, {
        "partition": partition,
        "fragment_labels": fragment_labels,
        "qubit_to_fragment": qubit_to_fragment,
        "cut_edges": cut_edges,
        "cut_gate_indices": cut_gate_indices,
        "objective": model.ObjVal,
        "mip_gap": model.MIPGap,
        "num_fragments": len(
            partition
        ),
        "labels": labels,
        "model": model,
    }


# ============================================================
# ANALYZE RESULTING QISKIT CUTTING PROBLEM
# ============================================================

def calculate_cutting_overhead(
    problem,
):
    """
    Calculate the product of QPD basis overheads.

    For example, if the partition creates:

        3 CNOT cuts

    and each CNOT has overhead 9,

    then:

        Gamma = 9^3 = 729

    """

    if not hasattr(
        problem,
        "bases",
    ):

        return None

    bases = problem.bases

    if bases is None:
        return None

    overheads = []

    for label, basis_list in (
        bases.items()
    ):

        for basis in basis_list:

            if hasattr(
                basis,
                "overhead",
            ):

                overheads.append(
                    float(
                        basis.overhead
                    )
                )

    if not overheads:
        return None

    return float(
        np.prod(
            overheads
        )
    )


# ============================================================
# RAY WORKER
# ============================================================

@ray.remote(
    num_cpus=1
)
def run_chunk(
    circuits,
    shots,
):
    """
    Execute a batch of subexperiments.
    """

    sampler = SamplerV2(
        options={
            "backend_options": {
                "max_parallel_threads": 1
            }
        }
    )

    return list(
        sampler
        .run(
            circuits,
            shots=shots,
        )
        .result()
    )


def chunked(
    seq,
    size,
):

    return [
        seq[
            i : i + size
        ]
        for i in range(
            0,
            len(seq),
            size,
        )
    ]


def run_parallel(
    subexperiments,
    shots,
):
    """
    Execute each partition's subexperiments
    using Ray.
    """

    n_cpus = int(
        ray.cluster_resources()[
            "CPU"
        ]
    )

    futures = {}

    for label, circuits in (
        subexperiments.items()
    ):

        circuits = list(
            circuits
        )

        size = max(
            1,
            len(circuits)
            // (
                n_cpus * 4
            ),
        )

        futures[label] = [
            run_chunk.remote(
                chunk,
                shots,
            )
            for chunk in chunked(
                circuits,
                size,
            )
        ]

    results = {}

    for label, refs in (
        futures.items()
    ):

        chunks = ray.get(
            refs
        )

        flattened = [
            result
            for chunk in chunks
            for result in chunk
        ]

        results[label] = (
            PrimitiveResult(
                flattened
            )
        )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 1. BUILD GROVER CIRCUIT
    # ========================================================

    n = N_QUBITS

    marked = (
        ("10" * n)[:n]
    )

    circuit = grover(
        n,
        marked,
        ITERATIONS,
    )

    # ========================================================
    # 2. TRANSPILE
    # ========================================================

    circuit = transpile(
        circuit,
        basis_gates=[
            "cx",
            "u",
        ],
        optimization_level=0,
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"Grover n={n} data qubits"
    )

    print(
        f"Total circuit qubits: "
        f"{circuit.num_qubits}"
    )

    print(
        f"Grover iterations: "
        f"{ITERATIONS}"
    )

    print(
        f"Theoretical optimal iterations: "
        f"~{optimal_iterations(n):.3e}"
    )

    print(
        "CX count:",
        circuit.count_ops().get(
            "cx",
            0,
        ),
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 3. PRINT INTERACTION GRAPH
    # ========================================================

    print_interaction_graph(
        circuit
    )

    # ========================================================
    # 4. OBSERVABLE
    # ========================================================

    observable = prefix_projector(
        marked,
        PREFIX_QUBITS,
        circuit.num_qubits,
    )

    exact = (
        analytic_prefix_probability(
            n,
            PREFIX_QUBITS,
            ITERATIONS,
        )
    )

    print()
    print(
        f"Observable terms: "
        f"{len(observable)}"
    )

    print(
        "Analytic P(prefix match):",
        exact,
    )

    # ========================================================
    # 5. GUROBI
    # ========================================================

    (
        partition_labels,
        gurobi_metadata,
    ) = optimize_partition_gurobi(
        circuit=circuit,
        max_qubits=QUBITS_PER_SUBCIRCUIT,
        time_limit=GUROBI_TIME_LIMIT,
        mip_gap=GUROBI_MIP_GAP,
        verbose=GUROBI_OUTPUT,
    )

    # ========================================================
    # 6. VERIFY PARTITION SIZE
    # ========================================================

    partition = (
        gurobi_metadata[
            "partition"
        ]
    )

    for fragment, qubits in (
        partition.items()
    ):

        if (
            len(qubits)
            >
            QUBITS_PER_SUBCIRCUIT
        ):

            raise RuntimeError(
                f"Gurobi produced invalid "
                f"fragment {fragment} with "
                f"{len(qubits)} qubits."
            )

    # ========================================================
    # 7. EXPAND OBSERVABLES
    #
    # We do this AFTER obtaining the original partition.
    # ========================================================

    # At this point there are no WireCut instructions yet.
    #
    # partition_problem will create the gate-cut/QPD
    # representation from the non-local gates.

    # ========================================================
    # 8. CREATE PARTITIONED / CUT PROBLEM
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "APPLYING GUROBI PARTITION TO QISKIT CUTTING"
    )

    print(
        "=" * 70
    )

    print(
        "Partition labels:",
        "".join(
            str(x)
            for x in partition_labels
        ),
    )

    problem = partition_problem(
        circuit=circuit,
        partition_labels=partition_labels,
        observables=observable.paulis,
    )

    # ========================================================
    # 9. GET SUBCIRCUITS
    # ========================================================

    print()
    print(
        "SUBCIRCUITS"
    )

    print(
        "-" * 70
    )

    for label, subcircuit in (
        problem.subcircuits.items()
    ):

        print(
            f"Fragment {label}: "
            f"{subcircuit.num_qubits} qubits"
        )

        print(
            f"  Gates: "
            f"{len(subcircuit.data)}"
        )

        print(
            f"  CX: "
            f"{subcircuit.count_ops().get('cx', 0)}"
        )

    # ========================================================
    # 10. EXPAND OBSERVABLES
    #
    # IMPORTANT:
    #
    # partition_problem() already returns subobservables
    # in the current addon workflow.
    #
    # We use those directly.
    # ========================================================

    subobservables = (
        problem.subobservables
    )

    # ========================================================
    # 11. DISPLAY CUT INFORMATION
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "CUT INFORMATION"
    )

    print(
        "=" * 70
    )

    print(
        "Gurobi weighted cut objective:",
        gurobi_metadata[
            "objective"
        ],
    )

    print(
        "Gurobi cross-fragment edges:",
        len(
            gurobi_metadata[
                "cut_edges"
            ]
        ),
    )

    print(
        "Gurobi cross-fragment gates:",
        len(
            gurobi_metadata[
                "cut_gate_indices"
            ]
        ),
    )

    print(
        "Gate indices:",
        gurobi_metadata[
            "cut_gate_indices"
        ],
    )

    # ========================================================
    # 12. GENERATE CUTTING EXPERIMENTS
    # ========================================================

    number_of_gurobi_cuts = len(
        gurobi_metadata[
            "cut_gate_indices"
        ]
    )

    if (
        number_of_gurobi_cuts
        <= 10
    ):

        num_samples = np.inf

    else:

        num_samples = (
            MC_SAMPLES
        )

    print()
    print(
        "Generating cutting experiments..."
    )

    print(
        "num_samples:",
        num_samples,
    )

    (
        subexperiments,
        coefficients,
    ) = generate_cutting_experiments(
        circuits=problem.subcircuits,
        observables=subobservables,
        num_samples=num_samples,
    )

    # ========================================================
    # 13. EXPERIMENT COUNTS
    # ========================================================

    print()
    print(
        "SUBEXPERIMENT COUNTS"
    )

    print(
        "-" * 70
    )

    total_experiments = 0

    for label, experiments in (
        subexperiments.items()
    ):

        count = len(
            experiments
        )

        total_experiments += count

        print(
            f"Fragment {label}: "
            f"{count} experiments"
        )

    print(
        f"Total experiments: "
        f"{total_experiments}"
    )

    # ========================================================
    # 14. RAY EXECUTION
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "RAY EXECUTION"
    )

    print(
        "=" * 70
    )

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _metrics_export_port=None,
    )

    try:

        results = run_parallel(
            subexperiments,
            SHOTS,
        )

    finally:

        ray.shutdown()

    # ========================================================
    # 15. RECONSTRUCT
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "RECONSTRUCTION"
    )

    print(
        "=" * 70
    )

    reconstructed = (
        reconstruct_expectation_values(
            results,
            coefficients,
            subobservables,
        )
    )

    # Combine the reconstructed expectation values
    # with the coefficients of the original observable.

    knitted = (
        np.dot(
            reconstructed,
            observable.coeffs,
        ).real
    )

    # ========================================================
    # 16. FINAL RESULTS
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "Analytic P(prefix match):"
    )

    print(
        f"    {exact}"
    )

    print()

    print(
        "Knitted P(prefix match):"
    )

    print(
        f"    {knitted}"
    )

    print()

    print(
        "Absolute error:"
    )

    print(
        f"    {abs(exact - knitted)}"
    )

    print()

    print(
        "Gurobi objective:"
    )

    print(
        f"    "
        f"{gurobi_metadata['objective']}"
    )

    print()

    print(
        "Gurobi partition:"
    )

    for f, qubits in (
        gurobi_metadata[
            "partition"
        ].items()
    ):

        print(
            f"    Fragment {f}: "
            f"{qubits}"
        )

    print()

    print(
        "Partition labels:"
    )

    print(
        f"    "
        f"{''.join(str(x) for x in partition_labels)}"
    )

    print()

    print(
        "Gurobi cut gates:"
    )

    print(
        f"    "
        f"{gurobi_metadata['cut_gate_indices']}"
    )

    print()

    print(
        "=" * 70
    )