#!/bin/bash
#SBATCH --job-name=qasmbench-benchmark
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=150G
#SBATCH --time=1:00:00
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

export MPI_HOME=/home/apps/spack/opt/spack/linux-cascadelake/openmpi-4.1.8-gaily2jwc3klcubp55unnlgiaglqu7mf
export PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
  
conda activate Qiskit-build-1.1.1
  
# ============================================================
# Thread configuration
# ============================================================
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
  
# ============================================================
# Debug information
# ============================================================
echo "=========================================="
echo "JOB ID:          ${SLURM_JOB_ID}"
echo "JOB NAME:        ${SLURM_JOB_NAME}"
echo "NODES:           ${SLURM_JOB_NUM_NODES}"
echo "TASKS:           ${SLURM_NTASKS}"
echo "CPUS/TASK:       ${SLURM_CPUS_PER_TASK}"
echo "TOTAL CPUS:      $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "NODE LIST:"
scontrol show hostnames ${SLURM_JOB_NODELIST}
echo "=========================================="
  
# ============================================================
# Run with WITHOUT MPI
# ============================================================
  
python compare_mpi_qiskit.py \
          --size medium \
          --num-circuits 25 \
          --precision single \
          --shots 1024 \
          --threads ${SLURM_CPUS_PER_TASK} \
          --output qasmbench_without_mpi_medium.csv
          

echo "=================================================="
echo " RUNS WITHOUT MPI COMPLETE"
echo "=================================================="
