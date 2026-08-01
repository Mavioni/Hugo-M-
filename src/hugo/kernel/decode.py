"""CUDA-graph autoregressive decode engine.

``model.generate`` grows the KV cache every token, so its shapes change per
step and cannot be captured as a CUDA graph. This engine runs one decode
step with *static* shapes: KV caches are preallocated to ``max_len`` and
attention beyond the current position is masked to -inf, which is
numerically identical to a growing cache. The whole step (embedding →
layers → LM head → argmax → token buffer update) is then captured once and
replayed per token with near-zero launch overhead.

Works with any model whose layers are ``nn.Linear`` *or* kernel-backed
``TernaryLinear`` (Llama/SmolLM2/Qwen2-style GQA decoder, RMSNorm, SiLU
MLP), so it benchmarks fp16 vs packed-ternary on equal footing.
"""
from __future__ import annotations

import torch

_NEG_INF = float("-inf")


class GraphDecoder:
    def __init__(
        self,
        model: torch.nn.Module,
        max_len: int = 512,
        device: torch.device | str | None = None,
    ):
        cfg = model.config
        layers = model.model.layers
        device = device or next(model.parameters()).device
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.num_layers = len(layers)
        self.hidden = cfg.hidden_size
        self.q_heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.rep = self.q_heads // self.kv_heads
        self.vocab = cfg.vocab_size
        self.max_len = max_len
        self.layers = layers
        self.embed = model.model.embed_tokens
        self.norm = model.model.norm
        self.lm_head = model.lm_head

        self.input_ids = torch.zeros(1, dtype=torch.int64, device=device)
        self.pos = torch.zeros(1, dtype=torch.int32, device=device)
        self.k_cache = torch.zeros(
            self.num_layers, 1, self.kv_heads, max_len, self.head_dim,
            dtype=self.dtype, device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.pos_mask = torch.arange(max_len, device=device, dtype=torch.int32)
        self.scale = self.head_dim ** -0.5
        self.graph: torch.cuda.CUDAGraph | None = None

    def _step(self) -> None:
        pos = self.pos
        h = self.embed(self.input_ids)  # [1, H]
        for li in range(self.num_layers):
            layer = self.layers[li]
            attn = layer.self_attn
            h2 = layer.input_layernorm(h)

            qkv = getattr(attn, "qkv", None)
            if qkv is not None:
                q, k, v = qkv(h2)
            else:
                q = attn.q_proj(h2)
                k = attn.k_proj(h2)
                v = attn.v_proj(h2)
            q = q.view(1, self.q_heads, self.head_dim)
            k = k.view(1, self.kv_heads, self.head_dim)
            v = v.view(1, self.kv_heads, self.head_dim)
            self.k_cache[li][:, :, pos, :] = k.unsqueeze(2)
            self.v_cache[li][:, :, pos, :] = v.unsqueeze(2)

            q_e = q.view(1, self.kv_heads, self.rep, self.head_dim)
            kt = self.k_cache[li].transpose(2, 3)  # [1, kvh, hd, max]
            scores = torch.matmul(q_e, kt) * self.scale  # [1, kvh, rep, max]
            past = self.pos_mask > pos
            scores = scores + torch.where(past, _NEG_INF, 0.0)
            probs = torch.softmax(scores, dim=-1).to(self.dtype)
            attn_out = torch.matmul(probs, self.v_cache[li])  # [1, kvh, rep, hd]
            attn_out = attn_out.view(1, self.q_heads * self.head_dim)

            h = h + attn.o_proj(attn_out)

            h3 = layer.post_attention_layernorm(h)
            gate_up = getattr(layer.mlp, "gate_up", None)
            if gate_up is not None:
                gate, up = gate_up(h3)
                h = h + layer.mlp.down_proj(torch.nn.functional.silu(gate) * up)
            else:
                gate = torch.nn.functional.silu(layer.mlp.gate_proj(h3))
                h = h + layer.mlp.down_proj(gate * layer.mlp.up_proj(h3))

        logits = self.lm_head(self.norm(h))  # [1, V]
        self.input_ids.copy_(torch.argmax(logits, dim=-1))
        pos.add_(1)

    def _reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.pos.zero_()

    def capture(self) -> None:
        """Warm up (compile triton/cublas kernels) and capture the step graph."""
        for _ in range(3):
            self._step()
        self._reset()
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self._step()
        torch.cuda.synchronize()

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Eager prefill for the prompt, then graph-replayed decode."""
        assert input_ids.dim() == 2 and input_ids.shape[0] == 1
        if input_ids.shape[1] + max_new_tokens > self.max_len:
            raise ValueError(
                f"prompt ({input_ids.shape[1]}) + {max_new_tokens} new tokens exceeds "
                f"max_len={self.max_len}"
            )
        ids = input_ids.to(self.device)
        self._reset()
        out = [t.item() for t in ids[0]]
        for tok in out[:-1]:
            self.input_ids.copy_(torch.tensor([tok], device=self.device))
            self._step()
        self.input_ids.copy_(ids[0, -1:])
        if self.graph is None:
            self.capture()
        for _ in range(max_new_tokens):
            self.graph.replay()
            out.append(self.input_ids.item())
        self._reset()
        return torch.tensor([out], dtype=torch.int64, device=self.device)
