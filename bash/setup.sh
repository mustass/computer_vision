echo
module load python3/3.10.11
module load cuda/11.7
source ../$PATH_TO_VENV/bin/activate
export CUDA_VISIBLE_DEVICES=0,1