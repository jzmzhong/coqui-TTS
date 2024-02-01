#!/bin/sh

# Run the program
DATA_DIR=/Users/jeffzhong/Desktop/work/Transsion/04_AccentedTTS

EXP_NAME=00_preprocess
RAW_DATASET_PATH=${DATA_DIR}/accented_TTS/00_raw/VCTK
PREPROCESSED_DATASET_PATH=${DATA_DIR}/accented_TTS/01_preprocessed/VCTK
OUT_PATH=${DATA_DIR}/accented_TTS/10_ckpts_local

/Users/jeffzhong/opt/anaconda3/envs/coqui-TTS/bin/python3 -m recipes.vctk.yourtts.train_yourtts_v00 \
    --exp-name $EXP_NAME \
    --raw-dataset-path $RAW_DATASET_PATH \
    --preprocessed-dataset-path $PREPROCESSED_DATASET_PATH \
    --out-path $OUT_PATH
