#!/bin/sh
# Grid Engine options (lines prefixed with #$)

#$ -N v11_acc-alpha100
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
DATA_DIR=/exports/eddie/scratch/s2526235

EXP_NAME=11_acc-alpha100
PREPROCESSED_DATASET_PATH=${DATA_DIR}/accented_TTS/01_preprocessed/VCTK_accent_split
OUT_PATH=${DATA_DIR}/accented_TTS/10_ckpts

python3 -m recipes.vctk.yourtts.train_yourtts_v11 \
    --exp-name $EXP_NAME \
    --preprocessed-dataset-path $PREPROCESSED_DATASET_PATH \
    --out-path $OUT_PATH \
    --batch-size 96 \
    --lang-emb-dim 4

    # --restore-path "" \
    