#!/bin/sh

DATA_DIR=/Users/jeffzhong/Desktop/work/Transsion/04_AccentedTTS

EXP_NAME=01_baseline_debug
PREPROCESSED_DATASET_PATH=${DATA_DIR}/accented_TTS/01_preprocessed/VCTK
OUT_PATH=${DATA_DIR}/accented_TTS/10_ckpts

/Users/jeffzhong/opt/anaconda3/envs/coqui-TTS/bin/python3 -m recipes.vctk.yourtts.train_yourtts_v01 \
    --exp-name $EXP_NAME \
    --preprocessed-dataset-path $PREPROCESSED_DATASET_PATH \
    --out-path $OUT_PATH \
    --batch-size 128
    
    # --restore-path "" \
