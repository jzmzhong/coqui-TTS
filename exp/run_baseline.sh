#!/bin/sh
# Grid Engine options (lines prefixed with #$)

#$ -N v00
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l h_vmem=512G
#$ -q gpu
#$ -pe gpu-a100 1
#$ -M s2526235@ed.ac.uk
#$ -m beas

#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit: -l h_rt
#  memory limit: -l h_vmem
#  GPU: -pe gpu

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Environment
module unload anaconda
module load anaconda
conda activate coqui-TTS

# Run the program
python3 -m recipes.vctk.yourtts.train_yourtts
