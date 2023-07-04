from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.models.transformer_from_pretrained_mt import (
    TransformerFromPretrainedMTModel,
    TransformerEncoderFromPretrainedMT,
    TransformerDecoderFromPretrainedMT,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)

import numpy as np
from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner


@register_model("consistTL_filter_mt")
class ConsistTLFILTERMT(TransformerFromPretrainedMTModel):
    r"""
    The consistTL knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "inference", "ensemble_inference"],
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
                            help="the directory of save or load datastore")
        parser.add_argument("--knn-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter k of vanilla knn-mt")
        parser.add_argument("--knn-lambda", type=float, metavar="D", default=0.25,
                            help="The hyper-parameter lambda of vanilla knn-mt")
        parser.add_argument("--knn-temperature", type=float, metavar="D", default=10,
                            help="The hyper-parameter temperature of vanilla knn-mt")
        parser.add_argument("--knn-keytype", choices= ["decoder_output", "last_ffn_input"], default="decoder_output",
                            help="The hyper-parameter temperature of vanilla knn-mt")
        # subset of datastore
        parser.add_argument("--subset-path", type=str, metavar="STR", default='',
                            help="the directory of the subset file")
        # ensemble
        parser.add_argument("--knn-child-datastore-path", type=str, metavar="STR", default='',
                            help="the directory to load a child datastore")
        parser.add_argument("--knn-syn-datastore-path", type=str, metavar="STR", default='',
                            help="the directory to load a syn datastore")
        parser.add_argument('--lambda-list', nargs='+', type=float, default=[0.2,0.05,0.05] ,
                            help="lambda of the parent/(child)/(syn)")
        parser.add_argument("--knn-k-list", nargs='+', type=int, metavar="N", default=[12,12,12],
                            help="The hyper-parameter of the parent/(child)/(syn)")
        parser.add_argument("--knn-temperature-list", nargs='+', type=float, metavar="D", default=[10,10,10],
                            help="The hyper-parameter temperature of vanilla knn-mt")
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with VanillaKNNMTDecoder
        """
        print("Build ConsistTLknnMT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return ConsistTLFILTERMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


class ConsistTLFILTERMTDecoder(TransformerDecoderFromPretrainedMT):
    r"""
    The vanilla knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        if args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,t
                # because we need access the same datastore in another 
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)  
            self.datastore = global_vars()["datastore"]

        elif args.knn_mode == "inference":
            # when inference, we don't load the keys, use its faiss index is enough
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=["vals"])
            self.datastore.load_faiss_index("keys")
            self.k_list = [128,256,512,1024,1536]
            global_vars()["k_list"] = self.k_list
            self.retriever = {}
            for kl in self.k_list:
                self.retriever[kl] = Retriever(datastore=self.datastore, k=kl)
            self.combiner = Combiner(lambda_=args.knn_lambda,
                     temperature=args.knn_temperature, probability_dim=len(dictionary))
            global_vars()["syn_retrivel_ind_dict"] = {}
            self.ind_dict = global_vars()["syn_retrivel_ind_dict"]
            for kl in self.k_list:
                self.ind_dict[kl] = {}

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        r"""
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if self.args.knn_mode == "build_datastore":
            if self.args.knn_keytype == "last_ffn_input":
                keys = select_keys_with_pad_mask(extra["last_ffn_input"], self.datastore.get_pad_mask())
            else:
                keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            # save half precision keys
            self.datastore["keys"].add(keys.half())
        
        elif self.args.knn_mode == "inference":
            ## query with x (x needn't to be half precision), 
            ## save retrieved `vals` and `distances`
            for kl in self.k_list:
                if self.args.knn_keytype == "last_ffn_input":
                    self.retriever[kl].retrieve(extra["last_ffn_input"], return_list=["indices"])
                else:
                    self.retriever[kl].retrieve(x, return_list=["indices"])
                # self.retriever[kl].retrieve(x, return_list=["indices"])
                inds = set(self.retriever[kl].results["indices"].cpu().numpy().reshape(-1).tolist())
                for i in inds:
                    self.ind_dict[kl][i] = 1

        if not features_only:
            x = self.output_layer(x)
        return x, extra
    

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == " ":
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


r""" Define some vanilla knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("consistTL_filter_mt", "consistTL_filter_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("consistTL_filter_mt", "consistTL_filter_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("consistTL_filter_mt", "consistTL_filter_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("consistTL_filter_mt", "consistTL_filter_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("consistTL_filter_mt", "consistTL_filter_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("consistTL_filter_mt", "consistTL_filter_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("consistTL_filter_mt", "consistTL_filter_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("consistTL_filter_mt", "consistTL_filter_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)

    
    

        

