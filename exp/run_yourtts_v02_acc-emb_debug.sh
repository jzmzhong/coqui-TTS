#!/bin/sh

DATA_DIR=/Users/jeffzhong/Desktop/work/Transsion/04_AccentedTTS

EXP_NAME=02_acc-emb_debug
PREPROCESSED_DATASET_PATH=${DATA_DIR}/accented_TTS/01_preprocessed/VCTK_accent_split
OUT_PATH=${DATA_DIR}/accented_TTS/10_ckpts

/Users/jeffzhong/opt/anaconda3/envs/coqui-TTS/bin/python3 -m recipes.vctk.yourtts.train_yourtts_v02 \
    --exp-name $EXP_NAME \
    --preprocessed-dataset-path $PREPROCESSED_DATASET_PATH \
    --out-path $OUT_PATH \
    --batch-size 8 \
    --lang-emb-dim 4

    # --restore-path "" \
    