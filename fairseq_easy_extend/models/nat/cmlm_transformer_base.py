# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""
import argparse
import collections
from dataclasses import field, dataclass

import omegaconf
import torch
from fairseq.models import register_model
from fairseq.models.nat import CMLMNATransformerModel
from fairseq.models.transformer import TransformerConfig

from fairseq_easy_extend.dataclass.utils import gen_parser_from_dataclass
from fairseq_easy_extend.dataclass.utils import convert_omegaconf_to_namesapce


@dataclass
class CMLMTransformerConfig(TransformerConfig):
    # --- special arguments ---
    sg_length_pred: bool = field(
        default=False,
        metadata={
            "help": "stop gradients through length"
        }
    )
    pred_length_offset: bool = field(
        default=False,
        metadata={
            "help": "predict length offset"
        },
    )
    length_loss_factor: float = field(
        default=0.1,
        metadata={"help": "loss factor for length"},
    )
    ngram_predictor: int = field(
        default=1, metadata={"help": "maximum iterations for iterative refinement."},
    )
    src_embedding_copy: bool = field(
        default=False,
        metadata={
            "help": "copy source embeddings"
        },
    )
    label_smoothing: float = field(default=0.1, metadata={"help": "label smoothing"})

@register_model("cmlm_transformer_base", dataclass=CMLMTransformerConfig)

class BaseCMLMNATransformerModel(CMLMNATransformerModel):

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, CMLMTransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        if isinstance(cfg, omegaconf.DictConfig):
            cfg = convert_omegaconf_to_namesapce(cfg)
        model = super().build_model(cfg, task)
        return model


    def forward_decoder(self, decoder_out, encoder_out, temperature=1.0, sampling = False, **unused):
        x, extra = self.decoder(
            normalize=False,
            prev_output_tokens=decoder_out[0],
            encoder_out=encoder_out,
            temperature=temperature,
            sampling=sampling,
        )

        if sampling:
            # apply temperature scaling
            x = x / temperature
            # perform multinomial sampling
            x = torch.multinomial(torch.softmax(x, dim = 1), num_samples = 1).squeeze(-1)

        else:
            # use argmax for greedy decoding
            # x = torch.argmax(x, dim = -1)
            x = x.argmax(dim = -1)

        return x, extra

    # def forward_decoder(self, decoder_out, encoder_out, temperature=1.0, sampling=False, **unused):
    #     x, extra = self.decoder(normalize=False, prev_output_tokens=decoder_out[0], encoder_out=encoder_out,
    #                             temperature=temperature)
    #
    #     if sampling:
    #         x = torch.multinomial(torch.softmax(x / temperature, dim=-1), num_samples=1).squeeze(-1)
    #     else:
    #         x = x.argmax(dim=-1)
    #
    #     return x, extra
