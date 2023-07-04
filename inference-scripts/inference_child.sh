PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../kNN-TL
CHILD_MODEL=$1
CHILD_DATA=$2
GEN_SUBSET=$3
DATA_STORE=$4
SUBSET_PATH=$5
RESULT_PATH=$6
src_lang=tr

BASE_MODEL=$PARENT_MODEL
DATA_PATH=$CHILD_DATA
GEN_SUBSET=$GEN_SUBSET
DATA_STORE=$DATA_STORE

if [ $GEN_SUBSET == 'valid' ] 
then
k_value=(8 16 20 24 28 32)
lambda_value=(0.2 0.25 0.3 0.35 0.4)
temperature_value=(10 30 50 70 100)
fi
if [ $GEN_SUBSET == 'test' ] 
then
k_value=(16)
lambda_value=(0.3)
temperature_value=(50)
fi

for k in ${k_value[@]}
do
    for lambda in ${lambda_value[@]}
    do
        for temperature in ${temperature_value[@]}
        do
            CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
            --task translation \
            --path ${src2model[$4]} \
            --dataset-impl mmap \
            --beam 5 --lenpen 1 --max-len-a 1.2 --max-len-b 10 --source-lang $src_lang --target-lang en \
            --gen-subset $GEN_SUBSET \
            --max-tokens 18000 \
            --scoring sacrebleu \
            --tokenizer moses \
            --remove-bpe \
            --user-dir $PROJECT_PATH/knnbox/models \
            --arch consistTL_knn_mt@transformer \
            --knn-mode inference \
            --knn-datastore-path $DATA_STORE \
            --knn-k $k \
            --knn-lambda $lambda \
            --knn-temperature $temperature \
            --results-path $RESULT_PATH \
            --knn-keytype last_ffn_input \
            --subset-path $SUBSET_PATH
            if [ $src_lang != tr ] 
            then
            bash $PROJECT_PATH/scripts/sacrebleu-ref.sh $src_lang en $gen_subset $RESULT_PATH/generate-$gen_subset.txt
            fi
            echo "src $src_lang subset-size $2 knn-k $k knn-lambda $lambda knn-temperature $temperature"
        done
    done
done
