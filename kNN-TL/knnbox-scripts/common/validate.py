r""" 
This file is copied from fairseq_cli/validate.py.
knnbox made 2 major changes:

change 1. We modified the part of parsing args so that is
can parse the arch specified on the cli instead of directly
using the arch inside checkpoint.

change 2. we add codes about `saving datastore vals`, `dump datastore`, etc. 
"""

import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.utils import reset_logging

## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from knnbox.datastore import Datastore, GreedyMergeDatastore
from knnbox.common_utils import filter_pad_tokens, global_vars
import numpy as np
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(cfg: DictConfig, override_args=None, args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    reset_logging()

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]
    
    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    
    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    knn_type = cfg.model.arch.split("@")[0]
    if "datastore" not in global_vars():
        # create suitable datastore class if not exists
        if knn_type in ["consistTL_knn_mt_visual", "parent_knn_mt", "consistTL_knn_mt", "vanilla_knn_mt", "adaptive_knn_mt", "kernel_smoothed_knn_mt", "vanilla_knn_mt_visual"]:
            global_vars()["datastore"] = Datastore(path=args.knn_datastore_path)
        if knn_type == "greedy_merge_knn_mt":
            global_vars()["datastore"] = GreedyMergeDatastore(path=args.knn_datastore_path)
    datastore = global_vars()["datastore"]
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end

    for subset in cfg.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample

            ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if knn_type in ["parent_knn_mt", "consistTL_knn_mt", "vanilla_knn_mt", "adaptive_knn_mt", "greedy_merge_knn_mt", "kernel_smoothed_knn_mt"]:
                non_pad_tokens, mask = filter_pad_tokens(sample["target"])
                datastore["vals"].add(non_pad_tokens)
                datastore.set_pad_mask(mask)

            elif knn_type in ["consistTL_knn_mt_visual", "vanilla_knn_mt_visual"]:
                non_pad_tokens, mask = filter_pad_tokens(sample["target"])
                datastore["vals"].add(non_pad_tokens)
                datastore.set_pad_mask(mask)
                # get the key-value pair related sentence_ids 
                target_len = mask.sum(dim=-1)
                sentence_ids = []
                for idx, sentence_id in enumerate(sample["id"].cpu().numpy()):
                    sentence_ids += [sentence_id]*target_len[idx]
                sentence_ids = np.array(sentence_ids, dtype=int)
                # get the key-value pair related token_postions
                token_positions = []
                for len_ in target_len:
                    token_positions += [i for i in range(len_)]
                token_positions = np.array(token_positions, dtype=int)
                # add them to datastore
                datastore["sentence_ids"].add(sentence_ids)
                datastore["token_positions"].add(token_positions)
            ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end

            _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            progress.log(log_output, step=i)
            log_outputs.append(log_output)

        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        progress.print(log_output, tag=subset, step=i)
    

    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if knn_type in ["consistTL_knn_mt_visual", "parent_knn_mt", "consistTL_knn_mt", "vanilla_knn_mt", "adaptive_knn_mt", "kernel_smoothed_knn_mt", "vanilla_knn_mt_visual"]:
        datastore.dump()    # dump to disk
        datastore.build_faiss_index("keys")   # build faiss index
    elif knn_type == "greedy_merge_knn_mt:":
        datastore.dump() # dump the un-pruned datastore to disk
        datastore.build_faiss_index(use_pca=False) # build faiss index for un-pruned datastore
        datastore.prune(merge_neighbors=args.merge_neighbors_n) # prune the datastore. search k neighbors when do greedy merge
        datastore.dump() # dump the pruned datastore to disk
        datastore.build_faiss_index(use_pca=True, pca_dim=args.pca_dim) # build faiss index with pre-PCA operation
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end


## knnbox code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_build_datastore_parser(default_task=None):
    r"""
    very similar to options.get_validation_parser() but parse arch as well.

    Difference:
    - when validate, we don't need to specify --arch and model args, because they are
    recorded in .pt file.

    - when building datastore, we need to load the saved model parameter to a knn-mt arch,
    which is different from the checkpoint original arch.
    For example, I have a nmt checkpoint with arch `transformer_iwslt_de_en`, and now I want to
    load it's parameter to arch `vanilla@transformer_iwslt_de_en`, I must specify
    arch = "vanilla@transfromer_iwslt_de_en".
    """
    parser = options.get_parser("Validation", default_task)
    options.add_dataset_args(parser, train=True)
    options.add_distributed_training_args(parser, default_world_size=1)
    # knnbox add one line below to parse arch
    options.add_model_args(parser)
    group = parser.add_argument_group("Evaluation")
    # from fairseq.dataclass.data_class import CommonEvalParams
    # options.gen_parser_from_dataclass(group, CommonEvalParams())
    from fairseq.dataclass.configs import CommonEvalConfig
    options.gen_parser_from_dataclass(group, CommonEvalConfig())
    return parser
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end 

def cli_main():
    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # parser = options.get_validation_parser()
    parser = get_build_datastore_parser()
    args = options.parse_args_and_arch(parser)


    ## only override args that are explicitly given on the command line
    # override_parser = options.get_validation_parser()
    override_parser = get_build_datastore_parser()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main, override_args=override_args, args=args)




if __name__ == "__main__":
    cli_main()
    

