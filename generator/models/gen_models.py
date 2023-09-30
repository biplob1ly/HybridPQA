import torch
from torch import nn
from transformers import T5ForConditionalGeneration


class WrappedEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.main_input_name = encoder.main_input_name
        self.encoder = encoder

    # input_ids: B x (N x S)
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        bsz, total_length = input_ids.shape
        ctx_length = total_length // self.n_ctxs
        # input_ids: B x (N x S) -> (B x N) x S
        input_ids = input_ids.view(bsz * self.n_ctxs, ctx_length)
        attention_mask = attention_mask.view(bsz * self.n_ctxs, ctx_length)
        # (B x N) x S -> (B x N) x S x H
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        # (B x N) x S x H -> B x (N x S) x H
        outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_ctxs * ctx_length, -1)
        return outputs


class GENERATOR_T5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    # input_ids: B x N x S
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            if input_ids.dim() == 3:
                self.encoder.n_ctxs = input_ids.size(1)
            # input_ids: B x (N x S)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(self, input_ids=None, attention_mask=None, max_length=None, **kwargs):
        self.encoder.n_ctxs = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )

    def wrap_encoder(self):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = WrappedEncoder(self.encoder)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

