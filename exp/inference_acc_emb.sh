#!/bin/sh
source ~/.bashrc
conda activate coquitts_py310torch23cu121
python -m recipes.vctk.accentbox.inference_acc_emb