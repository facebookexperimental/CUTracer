#!/bin/bash
# Run every case N times natively (buggy and fixed), one GPU per case, in parallel.
# usage: reliability.sh <bindir> <fixeddir> <N> <gpu,list> > out.txt
B=$1; F=$2; N=$3; IFS=, read -ra G <<< "$4"; i=0
for c in S1 S2 S3 S4 S5 S6 V1; do
  [ -x $B/$c ] || continue
  g=${G[$((i % ${#G[@]}))]}; i=$((i+1))
  ( h=0; ok=0
    for r in $(seq $N); do CUDA_VISIBLE_DEVICES=$g timeout -s INT 30 $B/$c > /dev/null 2>&1; [ $? = 124 ] && h=$((h+1)); done
    for r in $(seq $N); do CUDA_VISIBLE_DEVICES=$g timeout -s INT 120 $F/$c > /dev/null 2>&1 && ok=$((ok+1)); done
    echo "$c gpu=$g buggy_hung=$h/$N fixed_ok=$ok/$N" ) &
done; wait
