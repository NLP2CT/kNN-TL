PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../kNN-TL
PARENT_MODEL=$1
PARENT_DATA=$2
PARENT_DATASTORE=$3

BASE_MODEL=$PARENT_MODEL
DATA_PATH=$PARENT_DATA
ARCH=parent_knn_mt@transformer

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/validate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--dataset-impl mmap \
--valid-subset train \
--skip-invalid-size-inputs-valid-test \
--max-tokens 5000 \
--bpe fastbpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch $ARCH \
--knn-mode build_datastore \
--knn-datastore-path $PARENT_DATASTORE\
--knn-keytype last_ffn_input
