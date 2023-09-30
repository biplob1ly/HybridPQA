import collections

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


BiEncoderOutput = collections.namedtuple(
    'BiEncoderOutput',
    [
        'q_pooled',
        'q_seq',
        'ctx_pooled',
        'ctx_seq'
    ]
)


class Interaction:
    def __init__(self, level='pooled', broadcast='local', func='cosine'):
        self.level = level
        self.broadcast = broadcast
        self.func = func

    def compute_score(self, q_vectors, ctx_vectors):
        if self.level == 'pooled':
            Q, D = list(q_vectors.size())
            if self.broadcast == 'local':
                # q*c x d -> q x c x d
                ctx_vectors = ctx_vectors.view((Q, -1, D))
            if self.func == 'cosine':
                # For global: q_vector: q x d -> q x 1 x d, ctx_vectors:   q*c x d, result: q x q*c
                # For local : q_vector: q x d -> q x 1 x d, ctx_vectors: q x c x d, result: q x c
                return F.cosine_similarity(q_vectors.unsqueeze(1), ctx_vectors, dim=-1)
            else:
                # For global: q_vector: q x d -> q x 1 x d, ctx_vectors: q*c x d -> d x q*c, result: q x q*c
                # For local : q_vector: q x d -> q x 1 x d, ctx_vectors: q x c x d -> q x d x c, result: q x c
                return (q_vectors.unsqueeze(1) @ ctx_vectors.transpose(-1, -2)).squeeze(1)
        else:
            Q, _, D = list(q_vectors.size())
            if self.broadcast == 'local':
                W = ctx_vectors.size(1)
                # q*c x w x d -> q x c x w x d
                ctx_vectors = ctx_vectors.view((Q, -1, W, D))
            if self.func == 'cosine':
                # For global:
                # q_vector   :   q x w x d -> q x  1  x w x 1 x d
                # ctx_vectors: q*c x w x d ->     q*c x 1 x w x d
                # result:                     q x q*c x w x w     -> q x q*c x w -> q x q*c
                # For local :
                # q_vector   :     q x w x d -> q x 1 x w x 1 x d
                # ctx_vectors: q x c x w x d -> q x c x 1 x w x d
                # result:                       q x c x w x w     -> q x c x w -> q x c
                return F.cosine_similarity(
                    q_vectors.unsqueeze(2).unsqueeze(1),
                    ctx_vectors.unsqueeze(-3),
                    dim=-1
                ).max(-1).values.sum(-1)
            else:
                # global:
                # q_vector   :   q x w x d -> q x  1  x w x d
                # ctx_vectors: q*c x w x d ->     q*c x d x w
                # result:                     q x q*c x w x w -> q x q*c x w -> q x q*c
                # local:
                # q_vector   :     q x w x d -> q x 1 x w x d,
                # ctx_vectors: q x c x w x d -> q x c x d x w,
                # result:                       q x c x w x w -> q x c x w -> q x c
                return (q_vectors.unsqueeze(1) @ ctx_vectors.transpose(-1, -2)).max(-1).values.sum(-1)


HybridOutput = collections.namedtuple(
    'HybridOutput',
    [
        'q_sparse', 'q_dense', 'd_sparse', 'd_dense'
    ]
)


class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)


class RegWeightScheduler:
    """same scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return self.lambda_t


class ContrastLoss:
    def __init__(
            self,
            encoder_type='hybrid',
            level='pooled',
            broadcast='local',
            func='cosine',
            temperature=1.0,
            alpha=0.5,
            lambda_q=0.0003,
            lambda_d=0.0001,
            total_steps=2000
    ):
        self.encoder_type = encoder_type
        self.level = level
        self.broadcast = broadcast
        self.func = func
        self.temperature = temperature
        self.alpha = alpha
        self.q_reg_scheduler = RegWeightScheduler(lambda_q, total_steps) if lambda_q else None
        self.d_reg_scheduler = RegWeightScheduler(lambda_d, total_steps) if lambda_d else None
        self.interaction = Interaction(level=level, broadcast=broadcast, func=func)
        self.flops = FLOPS()

    def compute_loss(self, model_output, cids_per_qid, pos_cids_per_qid):
        pos_ctx_indices = []
        start = 0
        for cids, pos_cids in zip(cids_per_qid, pos_cids_per_qid):
            cum_pos = start + cids.index(pos_cids[0])
            pos_ctx_indices.append(cum_pos)
            if self.broadcast == 'global':
                start += len(cids)

        loss = None
        if model_output.q_sparse is not None:
            sparse_scores = self.interaction.compute_score(model_output.q_sparse, model_output.d_sparse) / self.temperature
            if len(model_output.q_sparse.size()) > 1:
                q_num = model_output.q_sparse.size(0)
                sparse_scores = sparse_scores.view(q_num, -1)
            sparse_softmax_scores = F.log_softmax(sparse_scores, dim=1)
            sparse_loss = F.nll_loss(
                sparse_softmax_scores,
                torch.tensor(pos_ctx_indices).to(sparse_softmax_scores.device),
                reduction='mean'
            )
            loss = sparse_loss

        if model_output.q_dense is not None:
            dense_scores = self.interaction.compute_score(model_output.q_dense, model_output.d_dense) / self.temperature
            if len(model_output.q_dense.size()) > 1:
                q_num = model_output.q_dense.size(0)
                dense_scores = dense_scores.view(q_num, -1)
            dense_softmax_scores = F.log_softmax(dense_scores, dim=1)
            dense_loss = F.nll_loss(
                dense_softmax_scores,
                torch.tensor(pos_ctx_indices).to(dense_softmax_scores.device),
                reduction='mean'
            )
            loss = loss + dense_loss if loss is not None else dense_loss

        # max_score, max_idxs = torch.max(softmax_scores, 1)
        # correct_predictions_count = (max_idxs == torch.tensor(pos_ctx_indices).to(max_idxs.device)).sum()
        if model_output.q_sparse is not None and self.q_reg_scheduler:
            lambda_q = self.q_reg_scheduler.step()
            loss += self.flops(model_output.q_sparse) * lambda_q
        if model_output.d_sparse is not None and self.d_reg_scheduler:
            lambda_d = self.d_reg_scheduler.step()
            loss += self.flops(model_output.d_sparse) * lambda_d
        return loss


class BiEncoder(nn.Module):
    def __init__(self, question_model, ctx_model, biencoder_type):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.biencoder_type = biencoder_type

    def encode_query(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None
    ):
        sparse, dense = None, None
        if input_ids is not None:
            if self.biencoder_type == 'hybrid':
                sparse, dense = self.question_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            elif self.biencoder_type == 'sparse':
                sparse, _ = self.question_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            else:
                dense = self.question_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
        return sparse, dense

    def encode_context(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None
    ):
        sparse, dense = None, None
        if input_ids is not None:
            if self.biencoder_type == 'hybrid':
                sparse, dense = self.ctx_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            elif self.biencoder_type == 'sparse':
                sparse, _ = self.ctx_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            else:
                dense = self.ctx_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
        return sparse, dense

    def forward(
            self,
            q_input_ids, q_attention_mask, q_token_type_ids,
            ctx_input_ids, ctx_attention_mask, ctx_token_type_ids
    ):
        q_sparse, q_dense = self.encode_query(q_input_ids, q_attention_mask, q_token_type_ids)
        d_sparse, d_dense = self.encode_context(ctx_input_ids, ctx_attention_mask, ctx_token_type_ids)
        return HybridOutput(q_sparse, q_dense, d_sparse, d_dense)