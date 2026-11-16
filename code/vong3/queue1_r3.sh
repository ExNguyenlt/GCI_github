#!/bin/sh
# hang doi 1 (chay lai duoc: moi buoc bo qua phan da co)
S=/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad; cd $S
python3 -I run_r3.py train cat "+PROF~PCAT~LR0.1~NR800" 0,1 catscrA_r3 0,1 >> log_r3_catscrA.txt 2>&1
python3 -I run_r3.py train cat "+PROF+PQ~PCAT~LR0.1~NR800" 0,1 catscrB_r3 0,1 >> log_r3_catscrB.txt 2>&1
python3 -I run_r3.py train lgb "+PROF+PQ~TE" 0,1,2 lgbte_r3 >> log_r3_lgbte.txt 2>&1
python3 -I run_r3.py collect lgb "+PROF+PQ~TE" 0,1,2 lgbte_r3 >> log_r3_lgbte.txt 2>&1
[ -f oof_lrbin_r3.npy ] || python3 -I div_r3.py lrbin lrbin_r3 > log_r3_lrbin.txt 2>&1
