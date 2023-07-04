PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../kNN-TL
warmup=1000
seed=1
lr=0.0003
bsz=4000
using_gpus=1
w_decay=0.0
max_epoch=200
attn_dropout=0.1
act_dropout=0.1
freq=1
kd_weight=0.01
aug_weight=0.0
prior_tau=1.0
loss_type=mse
kd_type=full_frozen
aux_src_lang=de

#source language
src_lang=tr
#INPUT
aux_src=$1
parent_model=$2
parent_data=$3
child_data=$4
child_model=$5

save_dir=$PROJECT_PATH/checkpoints/child
MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=0,1,2,3 python $PROJECT_PATH/train_transfer.py \
    ${child_data} \
    --arch transformer_from_pretrained_mt \
    --share-decoder-input-output-embed \
    --ddp-backend=no_c10d \
    --pretrained-mt-checkpoint $child_model \
    --encoder-embed-path ../../ConsistTLknn/de_en_avg5best.emb \
    --aux-src-lang $aux_src_lang --aux-src $aux_src \
    --teacher-data $parent_data \
    --kd-weight $kd_weight --prior-tau $prior_tau --teacher-dir $parent_model --loss-type $loss_type \
    --fp16 \
    --update-freq $freq \
    --max-tokens $bsz \
    --tensorboard-logdir clean_train \
    --save-dir $save_dir \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr $lr --lr-scheduler inverse_sqrt --warmup-updates $warmup --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay $w_decay \
    --attention-dropout $attn_dropout \
    --activation-dropout $act_dropout \
    --seed $seed \
    --eval-bleu \
    --max-epoch $max_epoch \
    --keep-best-checkpoints 5 \
    --keep-last-epochs 1 \
    --eval-bleu-args '{"beam": 5, "lenpen":1}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --criterion label_smoothed_cross_entropy_xkd --label-smoothing 0.1 2>&1


out_path=test.consist.lr$lr*$bsz*$using_gpus-$loss_type$kd_weight.out
fairseq-generate $PROJECT_PATH/transfer-bin/tr_bt_tl-bin --path $save_dir/checkpoint_best.pt --remove-bpe --beam 5 --lenpen 1 --batch-size 200 --gen-subset test > $PROJECT_PATH/test_out/rebuttal/$src_lang/bt/$out_path
if [ $src_lang == tr ] 
then
bash $PROJECT_PATH/scripts/sacrebleu.sh wmt17 $src_lang en $PROJECT_PATH/test_out/rebuttal/$src_lang/bt/$out_path
fi
if [ $src_lang != tr ] 
then
bash $PROJECT_PATH/scripts/sacrebleu-ref.sh $src_lang en test $PROJECT_PATH/test_out/fr/$src_lang/$out_path
fi

