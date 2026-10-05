#!/bin/bash
#SBATCH --job-name=qiskit-cutting-knitting
#SBATCH --partition=terai
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --output=/home/cdacB/santhoshj/digvijay/output/%j.out
#SBATCH --error=/home/cdacB/santhoshj/digvijay/error/%j.err

source ~/.bashrc

. /home/apps/spack/share/spack/setup-env.sh

spack load /oxmullt
spack load /gaily2jw
spack load /2wdhn7r

export OPENBLAS_ROOT=$(spack location -i /geypa2)
export LD_LIBRARY_PATH=$OPENBLAS_ROOT/lib:$LD_LIBRARY_PATH

conda activate qiskit_dist

# Launch 20 MPI ranks: 2 per node
srun python distributed_qcut/qiskit_cutting.py

echo "Execution Complete!!!"