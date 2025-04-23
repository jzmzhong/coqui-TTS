#!/bin/sh
conda init
conda activate coquitts_py310torch23cu121
python -m recipes.vctk.accentbox.train_yourtts_finetune_acc_emb > exp/yourtts_ft_vctk_acc_emb.out 2> exp/yourtts_ft_vctk_acc_emb.err