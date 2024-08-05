#!/bin/bash

OUTDIR="$1"
NASBENCH="$2"
NASPATH="$3"

if [ $NASBENCH != 'nasbench101' ] && [ $NASBENCH != 'nasbench201' ] && [ $NASBENCH != 'transbench101' ] && [ $NASBENCH != 'nds' ] ; then
    echo "NAS Benchmark '$NASBENCH' not supported"
    exit 1
fi

if [ $NASBENCH == 'nasbench101' ]; then
    NARCHS=423624
elif [ $NASBENCH == 'nasbench201' ]; then
    NARCHS=15625
elif [ $NASBENCH == 'transbench101' ]; then
    NARCHS=7352
elif [ $NASBENCH == 'nds' ]; then
    NARCHS=6000
else
    echo "NAS Benchmark '$NASBENCH' not supported"
    exit 1
fi

args=(
    --sspace $NASBENCH \
    --naspath $NASPATH \
    --outdir $OUTDIR \
    --index-st 0 \
    --index-ed $NARCHS \
)

python3 extract_nasbench.py "${args[@]}"