#!/bin/sh
# hang doi 2 (chay lai duoc). Cho hang doi 1 xong.
S=/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad; cd $S
while pgrep -f queue1_r3 >/dev/null; do sleep 3; done
PYTHONPATH=$S/pylib_torch:$S/pylib_tabm timeout 1300 python3 tabm_r3.py tabm_r3 0 >> log_r3_tabm.txt 2>&1
for s in 2 3 4; do python3 -I run_r3.py train cat "+PROF~PCAT" $s catn_r3 >> log_r3_cat.txt 2>&1; done
python3 -I run_r3.py collect cat "+PROF~PCAT" 2,3,4 catn_r3 >> log_r3_cat.txt 2>&1
python3 -I run_r3.py train lgb "+PROF+PQ~DART~LR0.1~NR1200" 0 dart_r3 >> log_r3_div.txt 2>&1
python3 -I run_r3.py collect lgb "+PROF+PQ~DART~LR0.1~NR1200" 0 dart_r3 >> log_r3_div.txt 2>&1
python3 -I run_r3.py train lgb "+PROF+PQ~ET~LR0.03~NR2500" 0 et_r3 >> log_r3_div.txt 2>&1
python3 -I run_r3.py collect lgb "+PROF+PQ~ET~LR0.03~NR2500" 0 et_r3 >> log_r3_div.txt 2>&1
python3 -I run_r3.py train lgb "+PROF+PQ~LT~LR0.03~NR1500" 0 lt_r3 >> log_r3_div.txt 2>&1
python3 -I run_r3.py collect lgb "+PROF+PQ~LT~LR0.03~NR1500" 0 lt_r3 >> log_r3_div.txt 2>&1
