#!/bin/bash
#SBATCH --job-name=dj-sweep-compare
#SBATCH --partition=himachal
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=90G
#SBATCH --time=12:00:00
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

# ------------------------------------------------
# Check MPI
# ------------------------------------------------
echo
echo "Checking MPI..."
echo "mpirun:"
which mpirun
mpirun --version
echo
echo "mpicc:"
which mpicc
mpicc --version
echo
echo "mpicxx:"
which mpicxx
mpicxx --version
  
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
echo "MPI ranks:       ${SLURM_NTASKS}"
echo "NODE LIST:"
scontrol show hostnames ${SLURM_JOB_NODELIST}
echo "=========================================="
  
# ============================================================
# Run
# ============================================================
  
mpirun \
  --mca pml ob1 \
  --mca btl self,tcp \
  -np ${SLURM_NTASKS} \
  python compare_dj.py \
          --algorithm dj \
          --qubits 40 41\
          --oracle balanced \
          --shots 2048 \
          --repeats 1 \
          --warmup 1 \
          --blocking_qubits 10 \
          --output dj_with_mpi_compare_36_40.csv \
          --threads ${SLURM_CPUS_PER_TASK}