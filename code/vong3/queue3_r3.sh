#!/bin/sh
S=/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad; cd $S
export NT=4 PYTHONPATH=$S/pylib_torch:$S/pylib_tabm
SEED=1 timeout 1400 python3 tabm_r3.py tabmB_r3 >> log_r3_tabmB.txt 2>&1
SEED=2 timeout 1400 python3 tabm_r3.py tabmC_r3 >> log_r3_tabmC.txt 2>&1
