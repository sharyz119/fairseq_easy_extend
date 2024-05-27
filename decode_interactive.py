from fairseq_easy_extend_cli import interactive

if __name__ == "__main__":
    interactive.cli_main()

# import argparse
# import logging
# import os
# import sys
# import fileinput
# import numpy as np
# import torch
# import fairseq
# import ast
# from collections import namedtuple
# from fairseq import checkpoint_utils, distributed_utils, tasks, utils
# from fairseq_cli.generate import get_symbols_to_strip_from_output
# from fairseq_easy_extend.dataclass.utils import convert_namespace_to_omegaconf
# from fairseq_easy_extend import options
# import fairseq_easy_extend.tasks.utils as task_utils
# from fairseq_easy_extend.dataclass.configs import FEETextgenConfig

# # Set up logging
# logging.basicConfig(
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=os.environ.get("LOGLEVEL", "INFO").upper(),
#     stream=sys.stdout,
# )
# logger = logging.getLogger("fairseq_cli.interactive")

# Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
# Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

# def buffered_read(input, buffer_size):
#     buffer = []
#     with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
#         for src_str in h:
#             buffer.append(src_str.strip())
#             if len(buffer) >= buffer_size:
#                 yield buffer
#                 buffer = []
#     if len(buffer) > 0:
#         yield buffer

# def make_batches(lines, cfg, task, max_positions, encode_fn):
#     tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)
#     itr = task.get_batch_iterator(
#         dataset=task.build_dataset_for_inference(tokens, lengths),
#         max_tokens=cfg.dataset.max_tokens,
#         max_sentences=cfg.dataset.batch_size,
#         max_positions=max_positions,
#         ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
#     ).next_epoch_itr(shuffle=False)
#     for batch in itr:
#         ids = batch["id"]
#         src_tokens = batch["net_input"]["src_tokens"]
#         src_lengths = batch["net_input"]["src_lengths"]
#         yield ids, src_tokens, src_lengths

# def main(cfg: FEETextgenConfig):
#     if isinstance(cfg, argparse.Namespace):
#         cfg = convert_namespace_to_omegaconf(cfg)
#     utils.import_user_module(cfg.common)
#     use_cuda = torch.cuda.is_available() and not cfg.common.cpu
#     task = tasks.setup_task(cfg.task)
#     models, _model_args = checkpoint_utils.load_model_ensemble(
#         utils.split_paths(cfg.common_eval.path),
#         arg_overrides=ast.literal_eval(cfg.common_eval.model_overrides),
#         task=task,
#         suffix=cfg.checkpoint.checkpoint_suffix,
#         strict=(cfg.checkpoint.checkpoint_shard_count == 1),
#         num_shards=cfg.checkpoint.checkpoint_shard_count,
#     )
#     for model in models:
#         if model is None:
#             continue
#         if cfg.common.fp16:
#             model.half()
#         if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
#             model.cuda()
#         model.prepare_for_inference_(cfg)

#     generator = task.build_generator(models, cfg.generation)
#     tokenizer = task.build_tokenizer(cfg.tokenizer)
#     bpe = task.build_bpe(cfg.bpe)

#     def encode_fn(x):
#         if tokenizer is not None:
#             x = tokenizer.encode(x)
#         if bpe is not None:
#             x = bpe.encode(x)
#         return x

#     def decode_fn(x):
#         if bpe is not None:
#             x = bpe.decode(x)
#         if tokenizer is not None:
#             x = tokenizer.decode(x)
#         return x

#     max_positions = utils.resolve_max_positions(
#         task.max_positions(), *[model.max_positions() for model in models]
#     )

#     start_id = 0
#     with open(cfg.interactive.input, "r") as input_file, open(cfg.output_file, "w") as output_file:
#         for inputs in buffered_read(cfg.interactive.input, cfg.interactive.buffer_size):
#             results = []
#             for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
#                 ids, src_tokens, src_lengths = batch
#                 if use_cuda:
#                     src_tokens = src_tokens.cuda()
#                     src_lengths = src_lengths.cuda()
#                 sample = {"net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths}}
#                 translations = task.inference_step(generator, models, sample)
#                 for i, (id, hypos) in enumerate(zip(ids.tolist(), translations)):
#                     src_tokens_i = utils.strip_pad(src_tokens[i], task.source_dictionary.pad())
#                     results.append((start_id + id, src_tokens_i, hypos))
#             for id_, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
#                 src_str = task.source_dictionary.string(src_tokens, cfg.common_eval.post_process)
#                 for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
#                     hypo_tokens, hypo_str, _ = utils.post_process_prediction(
#                         hypo_tokens=hypo["tokens"].int().cpu(),
#                         src_str=src_str,
#                         align_dict=None,
#                         tgt_dict=task.target_dictionary,
#                         remove_bpe=cfg.common_eval.post_process,
#                     )
#                     output_file.write(f"{decode_fn(hypo_str)}\n")
#             start_id += len(inputs)

# def cli_main():
#     parser = options.get_interactive_generation_parser()
#     parser.add_argument('--output-file', required=True, help='file to write hypotheses')
#     args = options.parse_args_and_arch(parser)
#     distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)

if __name__ == "__main__":
    cli_main()


