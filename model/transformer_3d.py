"""
Adapted from Point-E: https://github.com/openai/point-e/tree/main/point_e/models
"""


import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable, Union
import torch
import torch.nn as nn

# from .pretrained_clip import FrozenImageCLIP, ImageCLIP, ImageType

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def checkpoint(
    func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
    inputs: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    flag: bool,
):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, dtype=dtype)
        self.c_proj = nn.Linear(width, width, dtype=dtype)
        self.attention = QKVMultiheadAttention(dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, *,  dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *,  dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, dtype=dtype)
        self.mlp = MLP(dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# Core DiT Model
class ConditionEmbedder(nn.Module):
    """
    Embeds input condition. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        # self.embedder = nn.Linear(3, hidden_size, bias=True)
        self.embedder = nn.Linear(2, hidden_size, bias=True)
        self.dropout_prob = dropout_prob

    def token_drop(self, cond, force_drop_ids=None):
        """
        Drops condition to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(cond.shape[0], device=cond.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1 # force_drop_ids is a one dimension vector with size [batch_size] where 1 refers to drop the condition

        drop_ids = torch.tile(drop_ids.unsqueeze(1), [1, cond.shape[1]])
        cond = torch.where(drop_ids, torch.zeros_like(cond), cond)
        return cond

    def forward(self, cond, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond = self.token_drop(cond, force_drop_ids)
        embeddings = self.embedder(cond)
        return embeddings
    
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.cond_embed = ConditionEmbedder(width, 0.1)
        self.ln_pre = nn.LayerNorm(width, dtype=dtype)
        self.backbone = Transformer(
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param cond: an [N x C' x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        c_embed = self.cond_embed(cond, self.training)
        return self._forward_with_cond(x, [
                (t_embed, self.time_token_cond),
                (c_embed, True)
            ]
        )

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)