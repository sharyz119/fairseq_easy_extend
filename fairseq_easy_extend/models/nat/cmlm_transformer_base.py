import torch
import torch.nn.functional as F
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

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.sampling = getattr(args, 'sampling', False)
        self.temperature = getattr(args, 'temperature', 1.0)
        
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

    def forward_decoder(self, decoder_out, encoder_out, eos_penalty=None, max_iter=10, max_ratio=None, sampling_method='max', **kwargs):
    """
    Run the decoder forward pass for iterative refinement.

    Args:
        decoder_out (tuple): output from the decoder containing the output tokens and states
        encoder_out (Tensor): output from the encoder
        eos_penalty (float, optional): penalty for EOS token
        max_iter (int, optional): maximum number of iterations (default 10)
        max_ratio (float, optional): maximum ratio of the target length to the source length
        sampling_method (str, optional): sampling method to use ('max' or 'multinomial')

    Returns:
        tuple:
            - new_decoder_out (Tensor): updated decoder output
            - extra (dict): additional decoding results
    """
    for step in range(max_iter):
        decoder_result = self.decoder(
            normalize=False,
            prev_output_tokens=decoder_out[0],
            encoder_out=encoder_out,
            **kwargs
        )
        if isinstance(decoder_result, tuple):
            x, extra = decoder_result
        else:
            x = decoder_result
            extra = {}

        if max_iter > 1 and eos_penalty is not None:
            # apply length penalty
            eos_penalty_tensor = torch.ones_like(x) * eos_penalty
            eos_penalty_tensor = eos_penalty_tensor.masked_fill(decoder_out[0].ne(self.decoder.padding_idx), 0)
            x = x + eos_penalty_tensor

        # if sampling_method == 'max':
        #     x = x.argmax(-1)
        # elif sampling_method == 'multinomial':
        #     x = F.softmax(x, dim=-1)
        #     x = torch.multinomial(x.view(-1, x.size(-1)), 1).view(x.size()[:-1])

        if sampling_method == 'max':
                x = x.argmax(-1)
            elif sampling_method == 'multinomial':
                x = torch.multinomial(F.softmax(x, dim=-1), 1).squeeze(-1)
            else:
                raise ValueError(f"Unsupported sampling method: {sampling_method}")

        
        decoder_out = (x, extra)

    return decoder_out, extra


   



