PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../kNN-TL
PARENT_MODEL=$1
PSEUDO_PARENT_DATA=$2
PARENT_DATASTORE=$3

BASE_MODEL=$PARENT_MODEL
DATA_PATH=$PSEUDO_PARENT_DATA

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate-syn.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 5 --lenpen 1 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset train \
--max-tokens 5000 \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch parent_knn_mt@transformer \
--knn-mode inference \
--knn-datastore-path PARENT_DATASTORE \
--knn-keytype last_ffn_input \
--knn-value-list 16 32 64 128 256 \
--quiet
