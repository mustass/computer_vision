#!/bin/sh
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1"
#BSUB -J ObjectDetection
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo
module load python3/3.10.11
module load cuda/11.7
source ../dl4cv/bin/activate

wandb login fc0bfbd70fdd67da5b368b0cf28fff43dba2ede4
echo "Running script..."
python3 scripts/train.py -cn config_taco_training trainer.accelerator=gpu metric.metric.params.task=multiclass general.run_name=Resnet50_bigger_batch_higherlr_extra_layers model=resnet50_transfer 