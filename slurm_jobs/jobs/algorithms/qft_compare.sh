#!/bin/bash
#SBATCH --job-name=qft-comparision
#SBATCH --partition=shiwalik
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cdacB/santhoshj/digvijay/distributed-qcut/slurm_jobs/output/%j.out
#SBATCH --error=/home/cdacB/santhoshj/digvijay/distributed-qcut/slurm_jobs/error/%j.err

# ============================================================
# Environment
# ============================================================

source ~/.bashrc

. /home/apps/spack/share/spack/setup-env.sh

spack load /oxmullt
spack load /gaily2jw
spack load /2wdhn7r

# ============================================================
# OpenBLAS
# ============================================================

export OPENBLAS_ROOT=$(spack location -i /geypa2)
export LD_LIBRARY_PATH=$OPENBLAS_ROOT/lib:$LD_LIBRARY_PATH

export MPI_LIB=/home/apps/spack/opt/spack/linux-cascadelake/openmpi-4.1.8-gaily2jwc3klcubp55unnlgiaglqu7mf/lib
export LD_LIBRARY_PATH=$MPI_LIB:$LD_LIBRARY_PATH


# ============================================================
# Python / Conda
# ============================================================

conda activate qiskit_dist

export RAY_AUTH_MODE=disabled

ulimit -n 65535

echo "=========================================="
echo "Environment"
echo "=========================================="

echo "Open file limit:"
ulimit -n

echo ""
echo "Python:"
which python

echo ""
echo "Python version:"
python --version

echo ""
echo "MPI library:"
ls -l "$MPI_HOME/lib/libmpi.so.40" 2>/dev/null || \
    echo "WARNING: libmpi.so.40 not found"

echo ""
echo "MPI executables in PATH:"
which mpirun 2>/dev/null || echo "mpirun not in PATH"
which mpiexec 2>/dev/null || echo "mpiexec not in PATH"

echo "=========================================="

# ============================================================
# Get allocated nodes
# ============================================================

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

HEAD_NODE=${nodes[0]}

echo ""
echo "=========================================="
echo "SLURM JOB INFORMATION"
echo "=========================================="
echo "SLURM JOB ID      : $SLURM_JOB_ID"
echo "SLURM JOB NAME    : $SLURM_JOB_NAME"
echo "NUMBER OF NODES   : $SLURM_JOB_NUM_NODES"
echo "CPUS PER NODE     : $SLURM_CPUS_PER_TASK"
echo "HEAD NODE         : $HEAD_NODE"
echo "=========================================="

# ============================================================
# Benchmark configuration
# ============================================================

qbits=(22 26 30 34 38 40 44 48)
max_qbits=(15 15 20 20 28 30 30 30)

# ============================================================
# Run QFT benchmarks
# ============================================================

echo ""
echo "=========================================="
echo "Starting QFT benchmark"
echo "=========================================="

for i in "${!qbits[@]}"
do

    echo ""
    echo "=========================================="
    echo "Iteration              : $i"
    echo "Qubits                 : ${qbits[$i]}"
    echo "Qubits per subcircuit : ${max_qbits[$i]}"
    echo "Head node              : $HEAD_NODE"
    echo "=========================================="

    python -m algorithms.qft \
            --qubits "${qbits[$i]}" \
            --qubits-per-subcircuit "${max_qbits[$i]}" \
            --iterations 10 \
            --samples 100000 \
            --cut-strategy joint \
            --max-iterations 100 \
            --sv-max 32 \
            --benchmark \
            --no-show \
            --plot "algorithms/results/qft_${qbits[$i]}q_${max_qbits[$i]}max.png" \
            --data "algorithms/results/qft_${qbits[$i]}q_${max_qbits[$i]}max.json"

    status=$?

    echo ""
    echo "QFT ${qbits[$i]} qubit benchmark exit code: $status"

    if [ $status -ne 0 ]; then
        echo "ERROR: QFT benchmark failed for ${qbits[$i]} qubits."
        exit $status
    fi

done

# ============================================================
# Completion
# ============================================================

echo ""
echo "=========================================="
echo "Execution Complete!!!"
echo "=========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Nodes       : $SLURM_JOB_NUM_NODES"
echo "Head node   : $HEAD_NODE"
echo "=========================================="