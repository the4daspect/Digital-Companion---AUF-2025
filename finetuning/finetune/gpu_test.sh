#!/bin/bash
#SBATCH --job-name=gpu_driver_test
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=00:02:00
#SBATCH --output=%x-%j.out

module purge
module load 2023

# === NEW MODULES TO TRY ===
# Loading both CUDA and the corresponding cuDNN library
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

echo "--- Attempting to run nvidia-smi with loaded modules: ---"
module list
echo "--------------------------------------------------------"
srun nvidia-smi