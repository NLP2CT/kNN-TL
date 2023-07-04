#!/bin/bash

if [ $# -ne 4 ]; then
    echo "usage: $0 TESTSET SRCLANG TGTLANG GEN"
    exit 1
fi


SRCLANG=$1
TGTLANG=$2
SUBSET=$3
GEN=$4

if ! command -v sacremoses &> /dev/null
then
    echo "sacremoses could not be found, please install with: pip install sacremoses"
    exit
fi

grep ^H $GEN \
| sed 's/^H\-//' \
| sort -n -k 1 \
| cut -f 3 \
| sacremoses detokenize \
> $GEN.sorted.detok

sacrebleu /home/user/yc27405/LowResource/ConsisitTLknn/GV_raw_data/$SRCLANG/$SUBSET.raw.en -i $GEN.sorted.detok -m bleu -w 4
# sacrebleu --test-set $TESTSET --language-pair "${SRCLANG}-${TGTLANG}" < $GEN.sorted.detok
