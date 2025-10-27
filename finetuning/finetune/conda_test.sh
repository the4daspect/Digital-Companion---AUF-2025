#!/bin/bash
#SBATCH --job-name=CondaGpuTest
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

echo "--- 1. Loading Modules ---"
module purge
module load 2024
module load GCCcore/13.3.0
module load Anaconda3/2024.06-1
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

echo "--- Attempting to run nvidia-smi with loaded modules: ---"
module list
echo "--------------------------------------------------------"
srun nvidia-smi

echo "--- 2. Initializing and Activating Conda ---"
eval "$(conda shell.bash hook)"

conda activate finetune_env

echo "--- 3. Verifying the Environment ---"
echo "Active Conda environment:"
# The active environment will have a '*' next to it
conda info --envs
echo ""

echo "Path to Python executable:"
# This should point to the python inside your finetune_env
which python
echo ""

echo "--- 4. Checking GPU Availability with PyTorch ---"
# Run a short Python command to check if torch can see the CUDA device
srun python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'GPU available? {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

echo "--- Test Job Finished ---"