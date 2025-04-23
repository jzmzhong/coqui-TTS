#!/bin/sh
conda init
conda activate coquitts_py310torch23cu121
python -m recipes.vctk.accentbox.train_yourtts_finetune > exp/yourtts_ft_vctk_2.out 2> exp/yourtts_ft_vctk_2.err