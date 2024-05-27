import torch
from fairseq.models import register_model
from fairseq.models.nat import CMLMNATransformerModel
from fairseq.models.transformer import TransformerConfig
from dataclasses import field, dataclass

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
        from fairseq_easy_extend.dataclass.utils import gen_parser_from_dataclass
        gen_parser_from_dataclass(
            parser, CMLMTransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        from fairseq_easy_extend.dataclass.utils import convert_omegaconf_to_namesapce
        import omegaconf
        if isinstance(cfg, omegaconf.DictConfig):
            cfg = convert_omegaconf_to_namesapce(cfg)
        model = super().build_model(cfg, task)
        return model

    def forward_decoder(self, decoder_out, encoder_out, eos_penalty=None, max_iter=None, max_ratio=None, **kwargs):
        """
        Run the decoder forward pass for iterative refinement.

        Args:
            decoder_out (tuple): output from the decoder containing the output tokens and states
            encoder_out (Tensor): output from the encoder
            eos_penalty (float, optional): penalty for EOS token
            max_iter (int, optional): maximum number of iterations
            max_ratio (float, optional): maximum ratio of the target length to the source length

        Returns:
            tuple:
                - new_decoder_out (Tensor): updated decoder output
                - extra (dict): additional decoding results
        """
        for step in range(max_iter):
            x, extra = self.decoder(
                normalize=False,
                prev_output_tokens=decoder_out[0],
                encoder_out=encoder_out,
                **kwargs
            )
            if max_iter > 1:
                # apply length penalty
                eos_penalty = (torch.ones(x.size(0), x.size(1), device=x.device) * eos_penalty)
                eos_penalty = eos_penalty.masked_fill(decoder_out[0].ne(self.decoder.padding_idx), 0)
                x = x + eos_penalty
            # select top-1 (greedy decoding)
            x = x.argmax(-1)
            decoder_out = (x, extra)

        return decoder_out, extra
