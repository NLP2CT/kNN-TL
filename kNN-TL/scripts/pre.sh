TEXT=/home/user/yc27405/LowResource/ConsisitTLknn/data-bin/enfr
fairseq-preprocess \
    --source-lang en --target-lang fr \
    --trainpref $TEXT/new_train \
    --testpref $TEXT/test \
    --validpref $TEXT/valid \
    --destdir /home/user/yc27405/LowResource/ConsisitTLknn/data-bin/fr-en-580 \
    --workers 100 \
    --joined-dictionary \