#!/bin/bash
#SBATCH -J hf2.pre
#SBATCH --nodes=1
#SBATCH --time=00:29:59
#SBATCH --export=ALL
#SBATCH --mem=16G
#SBATCH --output=o/hf2.pre.o
#SBATCH --mail-type=all
#SBATCH --mail-user=chhahn@princeton.edu

now=$(date +"%T") 
echo "start time ... $now"

source ~/.bashrc 
conda activate gqp 

python /home/chhahn/projects/haloflow/bin/data/preprocess_hf2.py TNG50
python /home/chhahn/projects/haloflow/bin/data/preprocess_hf2.py TNG100
python /home/chhahn/projects/haloflow/bin/data/preprocess_hf2.py Eagle100
python /home/chhahn/projects/haloflow/bin/data/preprocess_hf2.py Simba100 

now=$(date +"%T")
echo "end time ... $now"
