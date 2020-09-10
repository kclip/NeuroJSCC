#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python /users/k1804053/NeuroJSCC/launch_experiment.py --where=rosalind --dataset=mnist_dvs \
--n_h=128 --n_output_enc=128 --num_samples_train=1000 \
--dt=25000 --snr=0  \
--test_period=500 --num_ite=1 --labels 1 7
