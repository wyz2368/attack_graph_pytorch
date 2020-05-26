#!/bin/sh
#PBS -N egta_attack_graph
# 
# User information:
#PBS -M max.olan.smith@gmail.com
#PBS -m abe
# 
# Number of cores (ppn=1), amount of memory, and walltime.
#PBS -l nodes=1:ppn=5,pmem=4gb,walltime=60:00:00
#PBS -j oe
#PBS -V
# 
# Flux allocation.
#PBS -A wellman_flux
#PBS -q flux
#PBS -l qos=flux
RUN_NAME=$1
CONFIG=$2

echo $RUN_NAME
echo $CONFIG

module load python-anaconda3/latest-3.6
cd ${PBS_O_WORKDIR}/../attackgraph/
python egta_main.py --run_name $RUN_NAME --config_files $CONFIG
