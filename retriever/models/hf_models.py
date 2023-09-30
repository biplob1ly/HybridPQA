import logging
from typing import Optional
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
logger = logging.getLogger()


class HFBertEncoderForMLM(nn.Module):
    def __init__(self, model_config, skip_mask):
        super().__init__()
        self.model_config = model_config
        self.skip_mask = skip_mask
        config = AutoConfig.from_pretrained(model_config.PRETRAINED_MODEL_TYPE, output_hidden_states=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_config.PRETRAINED_MODEL_TYPE, config=config)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # B x S x V
        logits = outputs.logits
        logits = logits * self.skip_mask
        if self.model_config.AGG == 'max':
            sparse, _ = torch.max(torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1), dim=1)
        else:
            sparse = torch.sum(torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1), dim=1)
        topk, indices = sparse.topk(k=self.model_config.TOP_K, dim=-1)
        top_sparse = sparse * torch.zeros_like(sparse).scatter(1, indices, torch.ones_like(topk))
        last_hidden_state = outputs.hidden_states[-1]
        pooled_dense = last_hidden_state[:, 0, :]
        return top_sparse, pooled_dense


class HFBertEncoder(nn.Module):
    def __init__(self, model_config, skip_list, special_tokens):
        super().__init__()
        self.model_config = model_config
        config = AutoConfig.from_pretrained(model_config.PRETRAINED_MODEL_TYPE, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_config.PRETRAINED_MODEL_TYPE, config=config)
        self.skip_list = skip_list
        self.special_tokens = special_tokens
        self.projection_layer = None
        if model_config.CROSS_INTERACTION and model_config.PROJECTION_DIM:
            self.projection_layer = nn.Linear(config.hidden_size, model_config.PROJECTION_DIM)

    def get_mask(self, input_ids):
        mask = [[(x not in self.skip_list) and (x not in self.special_tokens) for x in d] for d in input_ids.cpu().tolist()]
        return mask

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        if self.model_config.CROSS_INTERACTION:
            last_hidden_state = outputs.last_hidden_state
            if self.projection_layer:
                last_hidden_state = self.projection_layer(last_hidden_state)
            cur_mask = torch.tensor(self.get_mask(input_ids), device=last_hidden_state.device).unsqueeze(2).float()
            # B x S x D
            dense_repr = last_hidden_state * cur_mask
        else:
            # B x D
            dense_repr = outputs.last_hidden_state[:, 0, :]
        return dense_repr

