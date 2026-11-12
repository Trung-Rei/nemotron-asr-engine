"""Model hyper-parameters, read from the HF checkpoint's config.json / processor_config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NemotronLiteConfig:
    # preprocessor
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    preemphasis: float = 0.97
    # encoder
    d_model: int = 1024
    n_heads: int = 8
    n_layers: int = 24
    ff_dim: int = 4096
    conv_kernel: int = 9
    subsampling_factor: int = 8
    subsampling_channels: int = 256
    subsampling_kernel: int = 3
    subsampling_stride: int = 2
    left_context: int = 56
    max_positions: int = 5000
    default_lookahead: int = 3
    supported_lookaheads: tuple[int, ...] = (0, 3, 6, 13)
    # prompt
    num_prompts: int = 128
    prompt_hidden: int = 2048
    default_prompt_id: int = 101
    prompt_dictionary: dict[str, int] = field(default_factory=dict)
    # transducer
    pred_hidden: int = 640
    pred_layers: int = 2
    vocab_size: int = 13088  # includes blank
    blank_id: int = 13087
    max_symbols_per_step: int = 10

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "NemotronLiteConfig":
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        proc = json.loads((model_dir / "processor_config.json").read_text())
        enc = cfg["encoder_config"]
        fe = proc["feature_extractor"]
        return cls(
            sample_rate=fe["sampling_rate"],
            n_mels=fe["feature_size"],
            n_fft=fe["n_fft"],
            win_length=fe["win_length"],
            hop_length=fe["hop_length"],
            preemphasis=fe["preemphasis"],
            d_model=enc["hidden_size"],
            n_heads=enc["num_attention_heads"],
            n_layers=enc["num_hidden_layers"],
            ff_dim=enc["intermediate_size"],
            conv_kernel=enc["conv_kernel_size"],
            subsampling_factor=enc["subsampling_factor"],
            subsampling_channels=enc["subsampling_conv_channels"],
            subsampling_kernel=enc["subsampling_conv_kernel_size"],
            subsampling_stride=enc["subsampling_conv_stride"],
            left_context=enc["sliding_window"] - 1,
            max_positions=enc["max_position_embeddings"],
            default_lookahead=enc["default_num_lookahead_tokens"],
            supported_lookaheads=tuple(sorted(enc["supported_num_lookahead_tokens"])),
            num_prompts=cfg["num_prompts"],
            prompt_hidden=cfg["prompt_intermediate_size"],
            default_prompt_id=cfg["default_prompt_id"],
            prompt_dictionary=dict(proc["prompt_dictionary"]),
            pred_hidden=cfg["decoder_hidden_size"],
            pred_layers=cfg["num_decoder_layers"],
            vocab_size=cfg["vocab_size"],
            blank_id=cfg["blank_token_id"],
            max_symbols_per_step=cfg["max_symbols_per_step"],
        )

    def prompt_id(self, language: str | int | None) -> int:
        if language is None:
            return self.default_prompt_id
        if isinstance(language, int):
            return language
        if language not in self.prompt_dictionary:
            raise ValueError(f"Unknown language {language!r}; known: {sorted(self.prompt_dictionary)}")
        return self.prompt_dictionary[language]
