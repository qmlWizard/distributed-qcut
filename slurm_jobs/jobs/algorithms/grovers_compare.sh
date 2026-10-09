#!/bin/bash
#SBATCH --job-name=grover-comparision
#SBATCH --partition=shiwalik
#SBATCH --nodes=52
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
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

conda activate qiskit_dist
export RAY_AUTH_MODE=disabled
ulimit -n 65535

echo "Open file limit:"
ulimit -n

# ============================================================
# Get allocated nodes
# ============================================================

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

HEAD_NODE=${nodes[0]}

echo "=========================================="
echo "SLURM JOB ID      : $SLURM_JOB_ID"
echo "SLURM JOB NAME    : $SLURM_JOB_NAME"
echo "NUMBER OF NODES   : $SLURM_JOB_NUM_NODES"
echo "CPUS PER NODE     : $SLURM_CPUS_PER_TASK"
echo "HEAD NODE         : $HEAD_NODE"
echo "=========================================="

# ============================================================
# Run Grover benchmarks
# ============================================================

echo "=========================================="
echo "Starting Grover benchmark"
echo "=========================================="

qbits=(22 26 30 34 38 40 44 48)
max_qbits=(15 15 20 20 28 30 30 30)

for i in "${!qbits[@]}"
do
    echo "=========================================="
    echo "Iteration             : $i"
    echo "Qubits                : ${qbits[$i]}"
    echo "Qubits per subcircuit: ${max_qbits[$i]}"
    echo "=========================================="

    python -m algorithms.grovers \
            --qubits "${qbits[$i]}" \
            --qubits-per-subcircuit "${max_qbits[$i]}" \
            --iterations 10 \
            --samples 100000 \
            --cut-strategy joint \
            --max-iterations 100 \
            --sv-max 20 \
            --benchmark \
            --no-show \
            --plot "algorithms/results/grover_${qbits[$i]}q_${max_qbits[$i]}max.png" \
            --data "algorithms/results/grover_${qbits[$i]}q_${max_qbits[$i]}max.json"

done

echo "=========================================="
echo "Execution Complete!!!"
echo "=========================================="