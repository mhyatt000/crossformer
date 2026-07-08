from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crossformer.model.components.multiview import (
    StackedDinoTokenizer,
    StackedTipsTokenizer,
    StackedVitTokenizer,
    TIPS_VARIANT_DEFAULT,
)
from crossformer.model.components.vit_encoders import vit_encoder_configs
from crossformer.model.components.tokenizers import ImageTokenizer, LanguageTokenizer, LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.components.xstate import XStateEncoder
from crossformer.model.crossformer_module import CrossFormerModule
from crossformer.utils.spec import ModuleSpec


@dataclass
class StackedViewTokenizerCfg:
    """Config for the single stacked-multiview image tokenizer.

    Consumes observation["image"] (B, T, V, H, W, C) and emits one token group
    with per-token view ids (see components/multiview.py).
    """

    name: str = "image"
    encoder: str = "tips"  # tips | dino
    tips_variant: str = TIPS_VARIANT_DEFAULT
    dino_model_id: str | None = None
    freeze: bool = True
    permute_views: bool = True
    key_drop_prob: float = 0.1
    patch_drop_prob: float = 0.1

    def create(self) -> ModuleSpec:
        common = {
            "permute_views": self.permute_views,
            "key_drop_prob": self.key_drop_prob,
            "patch_drop_prob": self.patch_drop_prob,
        }
        if self.encoder == "tips":
            return ModuleSpec.create(StackedTipsTokenizer, variant=self.tips_variant, freeze=self.freeze, **common)
        if self.encoder == "dino":
            dino = {"model_id": self.dino_model_id} if self.dino_model_id else {}
            return ModuleSpec.create(StackedDinoTokenizer, **dino, **common)
        raise ValueError(f"unknown stacked encoder: {self.encoder!r}")


@dataclass
class ImageTokenizerCfg:
    """Config for one image tokenizer."""

    name: str
    obs_stack_keys: tuple[str, ...]
    encoder: ModuleSpec
    task_stack_keys: tuple[str, ...] = ()
    task_film_keys: tuple[str, ...] = ()
    use_obs_mask: bool = True
    use_token_learner: bool = False
    num_tokens: int = 8

    def create(self) -> ModuleSpec:
        return ModuleSpec.create(
            ImageTokenizer,
            encoder=self.encoder,
            obs_stack_keys=self.obs_stack_keys,
            task_stack_keys=self.task_stack_keys,
            task_film_keys=self.task_film_keys,
            use_obs_mask=self.use_obs_mask,
            use_token_learner=self.use_token_learner,
            num_tokens=self.num_tokens,
        )


@dataclass
class LowdimTokenizerCfg:
    """Config for one lowdim observation tokenizer."""

    name: str
    obs_keys: tuple[str, ...]
    use_obs_mask: bool = True
    discretize: bool = False
    n_bins: int = 256
    low: float = 0.0
    high: float = 1.0
    dropout_rate: float = 0.0
    token_drop: float = 0.0

    def create(self) -> ModuleSpec:
        return ModuleSpec.create(
            LowdimObsTokenizer,
            obs_keys=self.obs_keys,
            use_obs_mask=self.use_obs_mask,
            discretize=self.discretize,
            n_bins=self.n_bins,
            low=self.low,
            high=self.high,
            dropout_rate=self.dropout_rate,
            token_drop=self.token_drop,
        )


@dataclass
class LanguageTokenizerCfg:
    """Config for one language tokenizer."""

    name: str = "language"
    encoder: str | None = None
    finetune_encoder: bool = False
    use_task_mask: bool = True

    def create(self) -> ModuleSpec:
        return ModuleSpec.create(
            LanguageTokenizer,
            encoder=self.encoder,
            finetune_encoder=self.finetune_encoder,
            use_task_mask=self.use_task_mask,
        )


@dataclass
class TransformerCfg:
    """Config for CrossFormerTransformer size and horizon."""

    token_embedding_size: int
    transformer_kwargs: dict[str, Any]
    max_horizon: int
    repeat_task_tokens: bool = False

    @classmethod
    def from_size(cls, size: str, max_horizon: int, *, repeat_task_tokens: bool = False) -> TransformerCfg:
        token_embedding_size, transformer_kwargs = common_transformer_sizes(size)
        return cls(
            token_embedding_size=token_embedding_size,
            transformer_kwargs=transformer_kwargs,
            max_horizon=max_horizon,
            repeat_task_tokens=repeat_task_tokens,
        )


@dataclass
class ModelCfg:
    """Thin config wrapper for explicit CrossFormer module construction."""

    observation_tokenizers: list[ImageTokenizerCfg | LowdimTokenizerCfg | StackedViewTokenizerCfg] = field(
        default_factory=list
    )
    task_tokenizers: list[ImageTokenizerCfg | LanguageTokenizerCfg] = field(default_factory=list)
    heads: dict[str, ModuleSpec] = field(default_factory=dict)
    readouts: dict[str, int] = field(default_factory=dict)
    transformer: TransformerCfg | None = None

    def create(self) -> dict[str, Any]:
        assert self.transformer is not None, "transformer config is required"
        return {
            "observation_tokenizers": {cfg.name: cfg.create() for cfg in self.observation_tokenizers},
            "task_tokenizers": {cfg.name: cfg.create() for cfg in self.task_tokenizers},
            "heads": self.heads,
            "readouts": self.readouts,
            "token_embedding_size": self.transformer.token_embedding_size,
            "transformer_kwargs": self.transformer.transformer_kwargs,
            "max_horizon": self.transformer.max_horizon,
            "repeat_task_tokens": self.transformer.repeat_task_tokens,
        }

    def build(self) -> CrossFormerModule:
        """Instantiate the module described by this config."""
        return CrossFormerModule.create(**self.create())
