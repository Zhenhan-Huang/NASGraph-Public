#!/bin/bash

OUTDIR="$1"
NASBENCH="$2"
NASPATH="$3"
HASHFILE="$4"

if [ $NASBENCH == 'nasbench101' ]; then
    NARCHS=423624
    IMGSIZE=32
elif [ $NASBENCH == 'nasbench201' ]; then
    NARCHS=15625
    IMGSIZE=32
elif [ $NASBENCH == 'transbench101' ]; then
    NARCHS=7352
    IMGSIZE=224
elif [ $NASBENCH == 'nds' ]; then
    NARCHS=6000
    IMGSIZE=32
else
    echo "NAS Benchmark '$NASBENCH' not supported"
    exit 1
fi

args=(
    --outdir $OUTDIR \
    --sspace $NASBENCH \
    --naspath $NASPATH \
    --hashfile $HASHFILE \
    --index-st 0 \
    --index-ed $NARCHS \
    --nmodules 3 \
    --index-multiple 1 \
    --ncells 1 \
    --img-size $IMGSIZE \
    --stemchannels 16 \
    --ncpus 1 \
    --init 'normal' \
    --prepw 'none' \
    --directed \
    --no-use-bn \
    --ncpus 1 \
)

python3 convert_graph.py "${args[@]}"