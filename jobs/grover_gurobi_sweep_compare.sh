#!/bin/bash
#SBATCH --job-name=qiskit-cutting
#SBATCH --partition=terai
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=/home/cdacB/santhoshj/digvijay/distributed-qcut/output/%j.out
#SBATCH --error=/home/cdacB/santhoshj/digvijay/distributed-qcut/error/%j.err

# ============================================================
# Environment
# ============================================================

source ~/.bashrc

. /home/apps/spack/share/spack/setup-env.sh

spack load /oxmullt
spack load /gaily2jw
spack load /2wdhn7r

export OPENBLAS_ROOT=$(spack location -i /geypa2)
export LD_LIBRARY_PATH=$OPENBLAS_ROOT/lib:$LD_LIBRARY_PATH

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

echo "Allocated nodes:"
printf '%s\n' "${nodes[@]}"


# ============================================================
# Determine IP of Ray head
# ============================================================

HEAD_IP=$(srun \
    -N1 \
    -n1 \
    -w "$HEAD_NODE" \
    hostname -I | awk '{print $1}')

echo "Ray HEAD IP: $HEAD_IP"


# ============================================================
# Start Ray HEAD
# ============================================================

echo "Starting Ray head..."

srun --overlap -N1 -n1 -w "$HEAD_NODE" \
    ray start \
    --head \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --num-cpus=48 \
    --block &

RAY_HEAD_PID=$!

sleep 10


# ============================================================
# Ray address
# ============================================================

export RAY_ADDRESS="$HEAD_IP:6379"

echo "=========================================="
echo "RAY_ADDRESS = $RAY_ADDRESS"
echo "=========================================="


# ============================================================
# Start Ray workers
# ============================================================

echo "Starting Ray workers..."

for node in "${nodes[@]:1}"; do
    srun --overlap -N1 -n1 -w "$node" \
        ray start \
        --address="$RAY_ADDRESS" \
        --num-cpus=48 \
        --block &
done


# ============================================================
# Give Ray workers time to join
# ============================================================

echo "Waiting for Ray workers..."

sleep 20


# ============================================================
# Check Ray cluster
# ============================================================

echo "=========================================="
echo "Ray cluster status"
echo "=========================================="

ray status


# ============================================================
# Run ONE Python driver on a COMPUTE NODE
# ============================================================

echo "=========================================="
echo "Starting Qiskit circuit-cutting driver"
echo "=========================================="

srun \
    -N1 \
    -n1 \
    -w "$HEAD_NODE" \
    python qiskit_cutting.py \
            --min-qubits 10 \
            --max-qubits 30 \
            --step 2 \
            --qubits-per-subcircuit 10 \
            --samples 10000 


# ============================================================
# Cleanup
# ============================================================

echo "=========================================="
echo "Stopping Ray cluster"
echo "=========================================="

ray stop

echo "Execution Complete!!!"