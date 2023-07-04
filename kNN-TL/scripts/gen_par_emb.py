special_dict = ['<s>','<pad>','</s>','<unk>']
with open('/home/user/yc27405/LowResource/fairseq-0.12.0/data-bin/fr-en-580/dict.en.txt') as f:
    par_dict = f.readlines()
    par_dict = special_dict + par_dict

import torch
model = torch.load('/home/user/yc27405/LowResource/fairseq-0.12.0/checkpoints/en-fr/share-dec-fr-en-seed-1-lr-0.001-warmup-10000-dropout-0.1-max_update-80000-bsz-14375-freq-8-act_drop-0.1-attn_drop-0.1/checkpoint.avg5best.pt')
dict_emb = model['model']['encoder.embed_tokens.weight']
print(len(par_dict),len(dict_emb))
with open('/home/user/yc27405/LowResource/fairseq-0.12.0/checkpoints/en-fr/share-dec-fr-en-seed-1-lr-0.001-warmup-10000-dropout-0.1-max_update-80000-bsz-14375-freq-8-act_drop-0.1-attn_drop-0.1/fr_en_abg5best.emb','w') as fw:
    for i in range(len(par_dict)):
        fw.write(par_dict[i].strip().split(' ')[0]+' ')
        for value in dict_emb[i]:
            fw.write(str(value.item())+' ')
        fw.write('\n')
        