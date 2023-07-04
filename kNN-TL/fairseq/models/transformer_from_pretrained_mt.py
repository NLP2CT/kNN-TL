# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)


@register_model("transformer_from_pretrained_mt")
class TransformerFromPretrainedMTModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-mt-checkpoint",
            type=str,
            metavar="STR",
            help="XLM model to use for initializing transformer encoder and/or decoder",
        )
        # parser.add_argument(
        #     "--init-encoder-only",
        #     action="store_true",
        #     help="if set, don't load the XLM weights and embeddings into decoder",
        # )
        # parser.add_argument(
        #     "--init-decoder-only",
        #     action="store_true",
        #     help="if set, don't load the XLM weights and embeddings into encoder",
        # )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderFromPretrainedMT(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderFromPretrainedMT(args, tgt_dict, embed_tokens)


def upgrade_state_dict_with_xlm_weights(
    state_dict: Dict[str, Any], pretrained_xlm_checkpoint: str
) -> Dict[str, Any]:
    """
    Load XLM weights into a Transformer encoder or decoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_xlm_checkpoint: checkpoint to load XLM weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_xlm_checkpoint
    """
    # print("state_dict",state_dict)
    # print("pretrained_xlm_checkpoint",pretrained_xlm_checkpoint)
    if not os.path.exists(pretrained_xlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_xlm_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_xlm_checkpoint)
    xlm_state_dict = state["model"]
    cnt = 0
    for key in xlm_state_dict.keys():

        for search_key in ["embed_tokens", "embed_positions", "layers"]:
            if search_key in key:
                subkey = key[key.find(search_key) :]
                # assert subkey in state_dict, (
                #     "{} Transformer encoder / decoder "
                #     "state_dict does not contain {}. Cannot "
                #     "load {} from pretrained XLM checkpoint "
                #     "{} into Transformer.".format(
                #         str(state_dict.keys()), subkey, key, pretrained_xlm_checkpoint
                #     )
                # )

                if (subkey in state_dict) and ('decoder' in key):
                    state_dict[subkey] = xlm_state_dict[key]
                    cnt = cnt + 1
    print(cnt)
    return state_dict

def upgrade_state_dict_with_enc_weights(
    state_dict: Dict[str, Any], pretrained_xlm_checkpoint: str
) -> Dict[str, Any]:
    """
    Load XLM weights into a Transformer encoder or decoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_xlm_checkpoint: checkpoint to load XLM weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_xlm_checkpoint
    """
    if not os.path.exists(pretrained_xlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_xlm_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_xlm_checkpoint)
    xlm_state_dict = state["model"]
    cnt = 0
    for key in xlm_state_dict.keys():
        for search_key in ["embed_positions","layers"]:
            if search_key in key:
                subkey = key[key.find(search_key) :]
                # assert subkey in state_dict, (
                #     "{} Transformer encoder / decoder "
                #     "state_dict does not contain {}. Cannot "
                #     "load {} from pretrained XLM checkpoint "
                #     "{} into Transformer.".format(a
                #         str(state_dict.keys()), subkey, key, pretrained_xlm_checkpoint
                #     )
                # )
                if (subkey in state_dict) and ('encoder' in key):
                    # print('bf, {}:'.format(subkey), state_dict[subkey])
                    state_dict[subkey] = xlm_state_dict[key]
                    cnt = cnt + 1
                    # print('af, {}:'.format(subkey), state_dict[subkey])
    return state_dict


class TransformerEncoderFromPretrainedMT(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "init_decoder_only", False):
            # Don't load XLM weights for encoder if --init-decoder-only
            return

        assert hasattr(args, "pretrained_mt_checkpoint"), (
            "--pretrained-mt-checkpoint must be specified to load Transformer "
            "encoder from pretrained mt"
        )
        xlm_loaded_state_dict = upgrade_state_dict_with_enc_weights(
            state_dict=self.state_dict(),
            pretrained_xlm_checkpoint=args.pretrained_mt_checkpoint,
        )
        
        self.load_state_dict(xlm_loaded_state_dict, strict=True)


class TransformerDecoderFromPretrainedMT(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        # if getattr(args, "init_encoder_only", False):
        #     # Don't load XLM weights for decoder if --init-encoder-only
        #     return
        # assert hasattr(args, "pretrained_xlm_checkpoint"), (
        #     "--pretrained-xlm-checkpoint must be specified to load Transformer "
        #     "decoder from pretrained XLM"
        # )

        xlm_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
            state_dict=self.state_dict(),
            pretrained_xlm_checkpoint=args.pretrained_mt_checkpoint,
        )
        self.load_state_dict(xlm_loaded_state_dict, strict=True)


@register_model_architecture(
    "transformer_from_pretrained_mt", "transformer_from_pretrained_mt"
)
def base_architecture(args):
    transformer_base_architecture(args)
