"""NemotronLite: pure-PyTorch inference for nvidia/nemotron-3.5-asr-streaming-0.6b (offline + streaming)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import NemotronLiteConfig
from .encoder import Encoder, EncoderStreamState, _conv_out_len
from .features import LogMelFeatures, StreamingFeatures
from .rnnt import DecoderState, GreedyRNNT, Joint, PredictionNetwork

MODEL_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_FILES = ("config.json", "processor_config.json", "tokenizer.json", "model.safetensors")
LANG_TAG_AT_END = re.compile(r"\s*<[a-z]{2,3}(?:-[A-Z]{2})?>\s*$")


def strip_language_tag(text: str) -> str:
    return LANG_TAG_AT_END.sub("", text).strip()


class PromptProjector(nn.Module):
    def __init__(self, cfg: NemotronLiteConfig):
        super().__init__()
        self.linear_1 = nn.Linear(cfg.d_model + cfg.num_prompts, cfg.prompt_hidden)
        self.linear_2 = nn.Linear(cfg.prompt_hidden, cfg.d_model)

    def forward(self, x: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, one_hot[:, None, :].expand(-1, x.shape[1], -1)], dim=-1)
        return self.linear_2(torch.relu(self.linear_1(h)))


class Tokenizer:
    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as HFTokenizer

        self.tok = HFTokenizer.from_file(str(path))

    def decode(self, ids: list[int]) -> str:
        # SentencePiece (NeMo) renders the unknown piece as "⁇", HF tokenizers as "<unk>"
        return self.tok.decode(ids, skip_special_tokens=False).replace("<unk>", "⁇ ")


class NemotronLite(nn.Module):
    def __init__(self, cfg: NemotronLiteConfig):
        super().__init__()
        self.cfg = cfg
        self.preprocessor = LogMelFeatures(cfg)
        self.encoder = Encoder(cfg)
        self.prompt_projector = PromptProjector(cfg)
        self.encoder_projector = nn.Linear(cfg.d_model, cfg.pred_hidden)
        self.decoder = PredictionNetwork(cfg)
        self.joint = Joint(cfg)
        self.greedy = GreedyRNNT(cfg, self.decoder, self.joint)
        self.tokenizer: Tokenizer | None = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, model_dir: str | Path | None = None, device: str = "cuda", dtype=torch.float32):
        if model_dir is None:
            from huggingface_hub import hf_hub_download

            model_dir = Path(hf_hub_download(MODEL_REPO, "model.safetensors")).parent
            for name in MODEL_FILES:
                hf_hub_download(MODEL_REPO, name)
        model_dir = Path(model_dir)
        from safetensors.torch import load_file

        model = cls(NemotronLiteConfig.from_dir(model_dir))
        state = load_file(str(model_dir / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        # buffers computed at init (mel filterbank, window, inv_freq) are not in the checkpoint
        if unexpected or missing:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        model.tokenizer = Tokenizer(model_dir / "tokenizer.json")
        model = model.to(device=device, dtype=dtype).eval()
        model.preprocessor.float()  # front-end and positions always run in float32
        model.encoder.inv_freq = model.encoder.inv_freq.float()
        return model

    @property
    def device(self) -> torch.device:
        return self.encoder_projector.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder_projector.weight.dtype

    def _one_hot(self, language, batch: int) -> torch.Tensor:
        idx = torch.full((batch,), self.cfg.prompt_id(language), dtype=torch.long, device=self.device)
        return nn.functional.one_hot(idx, self.cfg.num_prompts).to(self.dtype)

    def _resolve_lookahead(self, lookahead: int | None) -> int:
        return self.cfg.default_lookahead if lookahead is None else lookahead

    def _text(self, tokens: list[int], strip_lang_tag: bool) -> str:
        text = self.tokenizer.decode(tokens)
        return strip_language_tag(text) if strip_lang_tag else text

    # ------------------------------------------------------------------ offline
    @torch.no_grad()
    def encode(self, audio: torch.Tensor, lengths: torch.Tensor, language="auto", lookahead: int | None = None):
        """audio (B, S) zero padded -> projected encoder output (B, T, H), lengths."""
        feats, feat_lens = self.preprocessor(audio.to(self.device).float(), lengths.to(self.device))
        enc, enc_lens = self.encoder(feats.to(self.dtype), feat_lens, self._resolve_lookahead(lookahead))
        enc = self.prompt_projector(enc, self._one_hot(language, enc.shape[0]))
        return self.encoder_projector(enc), enc_lens

    @torch.no_grad()
    def transcribe_batch(self, audios: list[np.ndarray | torch.Tensor], language="auto",
                         lookahead: int | None = None, strip_lang_tag: bool = True) -> list[str]:
        tensors = [torch.as_tensor(a, dtype=torch.float32).flatten() for a in audios]
        lengths = torch.tensor([t.numel() for t in tensors])
        batch = torch.zeros(len(tensors), int(lengths.max()))
        for i, t in enumerate(tensors):
            batch[i, : t.numel()] = t
        enc, enc_lens = self.encode(batch, lengths, language, lookahead)
        state = self.greedy.decode(enc, enc_lens)
        return [self._text(toks, strip_lang_tag) for toks in state.tokens]

    def transcribe(self, paths: list[str], batch_size: int = 8, **kwargs) -> list[str]:
        import soundfile as sf

        out = []
        for i in range(0, len(paths), batch_size):
            audios = [load_audio(p, self.cfg.sample_rate, sf) for p in paths[i : i + batch_size]]
            out.extend(self.transcribe_batch(audios, **kwargs))
        return out

    # ---------------------------------------------------------------- streaming
    def stream(self, language="auto", lookahead: int | None = None, strip_lang_tag: bool = True) -> "StreamingSession":
        return StreamingSession(self, language, self._resolve_lookahead(lookahead), strip_lang_tag)


def load_audio(path: str, sample_rate: int, sf=None) -> np.ndarray:
    if sf is None:
        import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        raise ValueError(f"{path}: expected {sample_rate} Hz audio, got {sr} Hz (resample first)")
    return audio


class StreamingSession:
    """Cache-aware streaming for one stream: feed audio, get text as soon as each chunk is decoded."""

    def __init__(self, model: NemotronLite, language, lookahead: int, strip_lang_tag: bool):
        self.model = model
        self.lookahead = lookahead
        self.strip_lang_tag = strip_lang_tag
        self.one_hot = model._one_hot(language, 1)
        self.features = StreamingFeatures(model.preprocessor)
        self.enc_state: EncoderStreamState = model.encoder.init_stream(lookahead)
        self.dec_state: DecoderState = model.greedy.init_state(1, model.device, model.dtype)
        self.done = False

    @property
    def latency_ms(self) -> float:
        cfg = self.model.cfg
        return (self.lookahead + 1) * cfg.subsampling_factor * cfg.hop_length * 1000 / cfg.sample_rate

    @torch.no_grad()
    def _step(self, total_mel: int | None) -> None:
        enc = self.model.encoder
        n = enc.chunk_mel_frames(self.enc_state)
        feats = self.features.take(n, device=self.model.device).to(self.model.dtype)
        x, n_valid = enc.stream_step(self.enc_state, feats[None], total_mel)
        x = self.model.prompt_projector(x[:, :n_valid], self.one_hot)
        x = self.model.encoder_projector(x)
        self.model.greedy.decode(x, torch.tensor([n_valid]), self.dec_state)

    def accept_audio(self, samples) -> str:
        """Add samples (16 kHz float) and decode every complete chunk. Returns the text so far."""
        self.features.accept(torch.as_tensor(samples, dtype=torch.float32))
        enc = self.model.encoder
        while self.features.available_frames() - self.features.frames_done >= enc.chunk_mel_frames(self.enc_state):
            self._step(None)
        return self.text

    def finish(self) -> str:
        """Flush the tail of the stream (zero padded like the offline front-end) and return the final text."""
        if self.done:
            return self.text
        self.features.finish()
        total_mel = self.features.num_valid_frames()
        sub = self.model.encoder.subsampling
        total_enc = torch.tensor(total_mel)
        for _ in sub.strided_convs():
            total_enc = _conv_out_len(total_enc, sub.k, sub.s)
        while self.enc_state.frames_out < int(total_enc):
            self._step(total_mel)
        self.done = True
        return self.text

    @property
    def tokens(self) -> list[int]:
        return self.dec_state.tokens[0]

    @property
    def text(self) -> str:
        return self.model._text(self.tokens, self.strip_lang_tag)
