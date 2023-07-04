:<<! 
[script description]: use vanilla-knn-mt to translate
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_MODEL=$PROJECT_PATH/checkpoints/bt/new-4-gpu-share-all-freq-1-lr-0.0003-bsz-4000-clip-base-clean-warmup-1000-max_epoch-200-seed-1-attn-dropout-0.1-act-dropout-0.1-w_decay-0.0-kd_weight-0.01-aug_weight-0.0-prior_tau-1.0-loss_type-js-retain-dropout/checkpoint_best.pt
DATA_PATH=$PROJECT_PATH/transfer-bin

# k_value=(8 4 12 16)
# lambda_value=(0.2 0.25 0.3 0.15 0.35)
# temperature_value=(5 10 20 40 60 80 100)

k_value=(12)
lambda_value=(0.15)
temperature_value=(5)

for k in ${k_value[@]}
do
    for lambda in ${lambda_value[@]}
    do
        for temperature in ${temperature_value[@]}
        do
            CUDA_VISIBLE_DEVICES=3 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
            --task translation \
            --path $BASE_MODEL \
            --dataset-impl mmap \
            --beam 5 --lenpen 1 --max-len-a 1.2 --max-len-b 10 --source-lang tr --target-lang en \
            --gen-subset test \
            --max-tokens 10000 \
            --scoring sacrebleu \
            --tokenizer moses \
            --remove-bpe \
            --user-dir $PROJECT_PATH/knnbox/models \
            --arch consistTL_knn_mt@transformer \
            --knn-mode inference \
            --knn-datastore-path $PROJECT_PATH/datastore/mse/synthesis \
            --knn-k $k \
            --knn-lambda $lambda \
            --knn-temperature $temperature 
            echo "--knn-k $k --knn-lambda $lambda --knn-temperature $temperature"
        done
    done
done
