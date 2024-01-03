#!/bin/sh
# Grid Engine options (lines prefixed with #$)

#$ -N yourtts_v00_preprocess
#$ -cwd
#$ -l h_rt=01:00:00
#$ -l h_vmem=10G

#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit: -l h_rt
#  memory limit: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Environment
module unload anaconda
module load anaconda
conda activate coqui-TTS

# Run the program
DATA_DIR=/exports/eddie/scratch/s2526235

EXP_NAME=00_preprocess
RAW_DATASET_PATH=${DATA_DIR}/accented_TTS/00_raw/VCTK
PREPROCESSED_DATASET_PATH=${DATA_DIR}/accented_TTS/01_preprocessed/VCTK
OUT_PATH=${DATA_DIR}/accented_TTS/10_ckpts

python3 -m recipes.vctk.yourtts.train_yourtts_v00 \
    --exp-name $EXP_NAME \
    --raw-dataset-path $RAW_DATASET_PATH \
    --preprocessed-dataset-path $PREPROCESSED_DATASET_PATH \
    --out-path $OUT_PATH
