#!/bin/sh
source ~/.bashrc
conda activate coquitts_py310torch23cu121
python -m recipes.vctk.accentbox.train_yourtts_base > exp/yourtts_base_librittsr.out 2> exp/yourtts_base_librittsr.err