"""RNN-T prediction network, joint and frame-synchronous batched greedy decoding (NeMo greedy_batch semantics)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import NemotronLiteConfig


class PredictionNetwork(nn.Module):
    def __init__(self, cfg: NemotronLiteConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.pred_hidden)
        self.lstm = nn.LSTM(cfg.pred_hidden, cfg.pred_hidden, num_layers=cfg.pred_layers, batch_first=True)
        self.decoder_projector = nn.Linear(cfg.pred_hidden, cfg.pred_hidden)

    def forward(self, tokens: torch.Tensor, state):
        """tokens (B,) -> projected output (B, H), new (h, c)."""
        out, state = self.lstm(self.embedding(tokens)[:, None, :], state)
        return self.decoder_projector(out[:, 0]), state


class Joint(nn.Module):
    def __init__(self, cfg: NemotronLiteConfig):
        super().__init__()
        self.head = nn.Linear(cfg.pred_hidden, cfg.vocab_size)

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(enc + pred))


@dataclass
class DecoderState:
    """Greedy decoder state carried across encoder chunks (streaming) for a batch of streams."""

    pred_out: torch.Tensor  # (B, H) projected prediction-network output for the last label
    hc: tuple[torch.Tensor, torch.Tensor]
    tokens: list[list[int]]


class GreedyRNNT:
    def __init__(self, cfg: NemotronLiteConfig, decoder: PredictionNetwork, joint: Joint):
        self.cfg, self.decoder, self.joint = cfg, decoder, joint
        self.counters: dict | None = None  # set to {} to count frames / symbol loops / host syncs

    def init_state(self, batch: int, device, dtype) -> DecoderState:
        blank = torch.full((batch,), self.cfg.blank_id, dtype=torch.long, device=device)
        h0 = torch.zeros(self.cfg.pred_layers, batch, self.cfg.pred_hidden, device=device, dtype=dtype)
        pred_out, hc = self.decoder(blank, (h0, h0.clone()))
        return DecoderState(pred_out=pred_out, hc=hc, tokens=[[] for _ in range(batch)])

    @torch.no_grad()
    def decode(self, enc_proj: torch.Tensor, lengths: torch.Tensor, state: DecoderState | None = None) -> DecoderState:
        """enc_proj (B, T, H) already projected by `encoder_projector`; lengths (B,) valid frames."""
        b, t_max, _ = enc_proj.shape
        if state is None:
            state = self.init_state(b, enc_proj.device, enc_proj.dtype)
        pred_out, (h, c) = state.pred_out, state.hc
        blank, max_sym = self.cfg.blank_id, self.cfg.max_symbols_per_step
        lengths = lengths.to(enc_proj.device)
        cnt = self.counters
        for t in range(t_max):
            active = t < lengths
            if cnt is not None:
                cnt["frames"] = cnt.get("frames", 0) + 1
                cnt["syncs"] = cnt.get("syncs", 0) + 1
            if not bool(active.any()):
                break
            f = enc_proj[:, t]
            for _ in range(max_sym):
                k = self.joint(f, pred_out).argmax(-1)
                emit = active & (k != blank)
                if cnt is not None:
                    cnt["symbol_loops"] = cnt.get("symbol_loops", 0) + 1
                    cnt["syncs"] = cnt.get("syncs", 0) + 1
                if not bool(emit.any()):
                    break
                emitted = emit.nonzero().flatten().tolist()
                if cnt is not None:  # nonzero + tolist, then one int(k[i]) read per token
                    cnt["syncs"] = cnt.get("syncs", 0) + 2 + len(emitted)
                    cnt["tokens"] = cnt.get("tokens", 0) + len(emitted)
                for i in emitted:
                    state.tokens[i].append(int(k[i]))
                new_out, (nh, nc) = self.decoder(k.masked_fill(~emit, blank), (h, c))
                m = emit[:, None]
                pred_out = torch.where(m, new_out, pred_out)
                h = torch.where(m[None], nh, h)
                c = torch.where(m[None], nc, c)
                active = emit
        state.pred_out, state.hc = pred_out, (h, c)
        return state

    @torch.no_grad()
    def decode_full(self, enc_proj: torch.Tensor, state: DecoderState) -> DecoderState:
        """Same result as `decode` when every stream has all T frames valid (a streaming chunk), with
        fewer host syncs: emitted labels stay on the GPU and are read back once, so the only per-iteration
        sync is the `emit.any()` that ends a frame's symbol loop."""
        b, t_max, _ = enc_proj.shape
        pred_out, (h, c) = state.pred_out, state.hc
        blank, max_sym = self.cfg.blank_id, self.cfg.max_symbols_per_step
        out = torch.full((t_max, max_sym, b), -1, dtype=torch.long, device=enc_proj.device)
        cnt = self.counters
        for t in range(t_max):
            f = enc_proj[:, t]
            active = None
            for s in range(max_sym):
                k = self.joint(f, pred_out).argmax(-1)
                emit = k != blank if active is None else active & (k != blank)
                out[t, s] = torch.where(emit, k, -1)
                if cnt is not None:
                    cnt["symbol_loops"] = cnt.get("symbol_loops", 0) + 1
                    cnt["syncs"] = cnt.get("syncs", 0) + 1
                if not bool(emit.any()):
                    break
                new_out, (nh, nc) = self.decoder(k.masked_fill(~emit, blank), (h, c))
                m = emit[:, None]
                pred_out = torch.where(m, new_out, pred_out)
                h = torch.where(m[None], nh, h)
                c = torch.where(m[None], nc, c)
                active = emit
        labels = out.permute(2, 0, 1).reshape(b, -1).tolist()  # (B, T*max_sym) in emission order
        if cnt is not None:
            cnt["frames"] = cnt.get("frames", 0) + t_max
            cnt["syncs"] = cnt.get("syncs", 0) + 1
        for i, row in enumerate(labels):
            new = [x for x in row if x >= 0]
            state.tokens[i].extend(new)
            if cnt is not None:
                cnt["tokens"] = cnt.get("tokens", 0) + len(new)
        state.pred_out, state.hc = pred_out, (h, c)
        return state
