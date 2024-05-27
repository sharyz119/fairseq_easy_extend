# from fairseq_easy_extend_cli import interactive

# if __name__ == "__main__":
#     interactive.cli_main()


import argparse
import logging
import fileinput
from collections import namedtuple
from fairseq import options, tasks, checkpoint_utils, utils, distributed_utils
import torch
import os
from fairseq_easy_extend.dataclass.utils import convert_namespace_to_omegaconf


def cli_main():
    parser = options.get_interactive_generation_parser()

    # Add new arguments to the parser, checking for conflicts
    existing_args = {action.option_strings[0]: action for action in parser._actions}

    if '--sampling' not in existing_args:
        parser.add_argument('--sampling', action='store_true', help='use multinomial sampling')
    if '--temperature' not in existing_args:
        parser.add_argument('--temperature', type=float, default =
        1.0, help = 'temperature for sampling')
        if '--output-file' not in existing_args:
            parser.add_argument('--output-file', type=str, required=False, help='path to save the output file')

        args = options.parse_args_and_arch(parser)

        distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)

    def buffered_read(input, buffer_size):
        buffer = []
        with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
            for src_str in h:
                buffer.append(src_str.strip())
                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []

        if len(buffer) > 0:
            yield buffer

    def make_batches(lines, cfg, task, max_positions, encode_fn):
        tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

        itr = task.get_batch_iterator(
            dataset=task.build_dataset_for_inference(
                tokens, lengths
            ),
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            ids = batch["id"]
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]

            yield Batch(
                ids=ids,
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                constraints=None,
            )

    Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")

    def main(cfg):
        if isinstance(cfg, argparse.Namespace):
            cfg = convert_namespace_to_omegaconf(cfg)

        # Set up logging
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
        logger = logging.getLogger("fairseq_cli.interactive")

        utils.import_user_module(cfg.common)

        use_cuda = torch.cuda.is_available() and not cfg.common.cpu

        # Setup task
        task = tasks.setup_task(cfg.task)

        # Load ensemble
        logger.info("loading model(s) from {}".format(cfg.common_eval.path))
        models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=eval(cfg.common_eval.model_overrides),
            task=task,
        )

        for model in models:
            if model is None:
                continue
            if cfg.common.fp16:
                model.half()
            if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)

        # Initialize generator
        generator = task.build_generator(models, cfg.generation)

        # Handle tokenization and BPE
        tokenizer = task.build_tokenizer(cfg.tokenizer)
        bpe = task.build_bpe(cfg.bpe)

        def encode_fn(x):
            if tokenizer is not None:
                x = tokenizer.encode(x)
            if bpe is not None:
                x = bpe.encode(x)
            return x

        def decode_fn(x):
            if bpe is not None:
                x = bpe.decode(x)
            if tokenizer is not None:
                x = tokenizer.decode(x)
            return x

        max_positions = utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        )

        all_hypotheses = []
        for inputs in buffered_read(cfg.interactive.input, cfg.interactive.buffer_size):
            results = []
            for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
                src_tokens = batch.src_tokens
                src_lengths = batch.src_lengths
                if use_cuda:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()

                sample = {
                    "net_input": {
                        "src_tokens": src_tokens,
                        "src_lengths": src_lengths,
                    },
                }
                with torch.no_grad():
                    translations = task.inference_step(generator, models, sample)

                for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    src_tokens_i = utils.strip_pad(src_tokens[i], task.source_dictionary.pad())
                    for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
                        hypo_tokens = hypo['tokens'].int().cpu()
                        hypo_str = task.target_dictionary.string(hypo_tokens, cfg.common_eval.post_process)
                        all_hypotheses.append(hypo_str)

        # Save all hypotheses to the output file if specified
        if cfg.interactive.output_file:
            with open(cfg.interactive.output_file, 'w') as f:
                for hyp in all_hypotheses:
                    f.write(f"{hyp}\n")
        else:
            for hyp in all_hypotheses:
                print(hyp)

    if __name__ == "__main__":
        cli_main()

