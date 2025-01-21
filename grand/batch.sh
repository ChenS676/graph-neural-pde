#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=accelerated
#SBATCH --job-name=dw_poc_enc

#SBATCH --nodes=1
#SBATCH --mem=50160mb
#SBATCH --output=log/deepwalk_pos_%j.output
#SBATCH --error=error/deepwalk_pos_%j.error
#SBATCH --gres=gpu # Ensure you are allowed to use these many GPUs, otherwise reduce the number here
#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/graph-neural-pde/

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cshao676@gmail.com


# Request GPU resources
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate EAsF

cd /hkfs/work/workspace/scratch/cc7738-rebuttal/graph-neural-pde/src

module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


python deepwalk_embeddings.py --dataset Cora --embedding_dim 64 --walk_length 80 --geom_gcn_splits
python deepwalk_embeddings.py --dataset Cora --embedding_dim 128 --walk_length 40 --geom_gcn_splits
python deepwalk_embeddings.py --dataset Cora --embedding_dim 256 --walk_length 40 --geom_gcn_splits

