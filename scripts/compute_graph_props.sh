#!/bin/bash

WDIR="$1"
OUTDIR="$2"
FPTRN="$3"
NASBENCH="$4"

if [ $NASBENCH == 'nasbench101' ]; then
    NARCHS=423624
elif [ $NASBENCH == 'nasbench201' ]; then
    NARCHS=15625
elif [ $NASBENCH == 'transbench101' ]; then
    NARCHS=7352
else
    echo "NAS Benchmark '$NASBENCH' not supported"
    exit
fi

args=(
    --wdir $WDIR \
    --outdir $OUTDIR \
    --fptrn $FPTRN \
    --index-st 0 \
    --index-ed $NARCHS \
    --directed \
    --ncpus 12
)

python3 compute_props.py "${args[@]}"