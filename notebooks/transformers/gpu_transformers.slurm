#!/bin/bash

##SBATCH -C v100-32g  

#SBATCH --job-name=mronce_run

#SBATCH --qos=qos_gpu-t4 # change

#SBATCH --output=%x.out

#SBATCH --error=%x.err

#SBATCH --time=100:00:00 # change

#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1  

#SBATCH --cpus-per-task=10

#SBATCH --account=owt@v100

#SBATCH --ntasks=1

#SBATCH --hint=nomultithread

module purge

module load tensorflow-gpu/py3/2.11.0

set -x

cd /gpfswork/rech/owt/uwz81lo/2023-sfh-galaxy-classification/notebooks

python TF_SFH_example.py 
