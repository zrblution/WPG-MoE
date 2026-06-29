"""Shared post encoder."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.utils.schemas import META_TOKEN, make_post_marker


class PostEncoder(nn.Module):
    """Post encoder that supports both bidirectional and decoder-style backbones."""

    def __init__(
        self,
        model_name: str = "/path/to/Qwen3.5-2B",
        *,
        max_post_tokens: int = 512,
        num_post_markers: int = 2048,
        pooling_strategy: str = "auto",
        torch_dtype: Union[str, torch.dtype, None] = "auto",
        gradient_checkpointing: bool = False,
        use_lora: bool = False,
        trust_remote_code: bool = False,
        device_map: str | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        backbone_model_class: str = "auto",
        encoder_forward_batch_size: int | None = None,
        use_fast_tokenizer: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_post_tokens = max_post_tokens
        self.num_post_markers = num_post_markers
        self.encoder_forward_batch_size = (
            int(encoder_forward_batch_size)
            if encoder_forward_batch_size is not None and int(encoder_forward_batch_size) > 0
            else None
        )
        self.backbone_model_class = backbone_model_class
        self.use_lora = bool(use_lora)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)

        tokenizer_kwargs = {
            "trust_remote_code": trust_remote_code,
            "use_fast": bool(use_fast_tokenizer),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.cls_token is not None:
                self.tokenizer.pad_token = self.tokenizer.cls_token
            elif self.tokenizer.sep_token is not None:
                self.tokenizer.pad_token = self.tokenizer.sep_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token

        # We only need the final hidden states for pooling. Returning every layer
        # materially increases memory traffic for large decoder backbones.
        model_kwargs = {"trust_remote_code": trust_remote_code}
        resolved_dtype = self._resolve_torch_dtype(torch_dtype)
        if resolved_dtype is None:
            resolved_dtype = torch.float32
        model_kwargs["torch_dtype"] = resolved_dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        loader_kind = self._resolve_loader_kind(config, backbone_model_class)
        loading_report_logger = logging.getLogger("transformers.utils.loading_report")
        previous_loading_report_level = loading_report_logger.level
        loading_report_logger.setLevel(logging.ERROR)
        try:
            if loader_kind == "causal_lm":
                loaded = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                backbone = getattr(loaded, "model", loaded)
            else:
                backbone = AutoModel.from_pretrained(model_name, **model_kwargs)
        finally:
            loading_report_logger.setLevel(previous_loading_report_level)

        post_tokens = [make_post_marker(idx) for idx in range(1, num_post_markers + 1)]
        special_tokens = {"additional_special_tokens": post_tokens + [META_TOKEN]}
        added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        if added_tokens:
            backbone.resize_token_embeddings(len(self.tokenizer))

        self.post_token_ids = {token: self.tokenizer.convert_tokens_to_ids(token) for token in post_tokens}
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            try:
                backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                backbone.gradient_checkpointing_enable()
            if hasattr(backbone, "enable_input_require_grads"):
                backbone.enable_input_require_grads()
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
        self.backbone = backbone
        self.cls_token_id = self.tokenizer.cls_token_id
        self.meta_token_id = self.tokenizer.convert_tokens_to_ids(META_TOKEN)
        self.hidden_dim = self._resolve_hidden_size(backbone)
        self.pooling_strategy = self._resolve_pooling_strategy(backbone, pooling_strategy)
        if self.pooling_strategy == "last_token":
            self.tokenizer.padding_side = "left"

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def format_post_text(self, post_marker: str, text: str, meta_info: Optional[Dict[str, object]] = None) -> str:
        formatted = f"{post_marker} {text}"
        if meta_info:
            meta_text = " ".join(f"{key}={value}" for key, value in meta_info.items())
            formatted = f"{formatted} {META_TOKEN} {meta_text}"
        return formatted

    def encode_posts(self, formatted_texts: List[str], post_markers: List[str]) -> torch.Tensor:
        if len(formatted_texts) != len(post_markers):
            raise ValueError("formatted_texts and post_markers must be aligned")
        if not formatted_texts:
            return torch.empty((0, self.hidden_dim), device=self.device)

        max_chunk_size = self.encoder_forward_batch_size or len(formatted_texts)
        representations = []
        start = 0
        chunk_size = max_chunk_size
        while start < len(formatted_texts):
            effective_chunk_size = min(chunk_size, len(formatted_texts) - start)
            chunk_texts = formatted_texts[start : start + effective_chunk_size]
            chunk_markers = post_markers[start : start + effective_chunk_size]
            try:
                encodings = self.tokenizer(
                    chunk_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_post_tokens,
                    return_tensors="pt",
                )
                encodings = {key: value.to(self.device) for key, value in encodings.items()}
                last_hidden = self._forward_last_hidden(encodings)
                for batch_idx, marker in enumerate(chunk_markers):
                    if self.pooling_strategy == "last_token":
                        position = self._resolve_last_token_position(encodings["attention_mask"][batch_idx])
                    else:
                        marker_id = self.post_token_ids.get(marker)
                        token_ids = encodings["input_ids"][batch_idx]
                        position = self._resolve_marker_position(token_ids, marker_id)
                    representations.append(last_hidden[batch_idx, position, :])
                start += effective_chunk_size
                if effective_chunk_size < max_chunk_size:
                    chunk_size = min(max_chunk_size, effective_chunk_size * 2)
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc):
                    raise
                if effective_chunk_size <= 1:
                    raise RuntimeError(
                        "CUDA OOM while encoding a single post; reduce max_post_tokens or use a smaller backbone."
                    ) from exc
                self._clear_cuda_cache()
                reduced_chunk_size = max(1, effective_chunk_size // 2)
                logging.warning(
                    "PostEncoder OOM at chunk_size=%s max_post_tokens=%s; retrying with chunk_size=%s",
                    effective_chunk_size,
                    self.max_post_tokens,
                    reduced_chunk_size,
                )
                chunk_size = reduced_chunk_size
        return torch.stack(representations, dim=0)

    def _forward_last_hidden(self, encodings: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(**encodings, return_dict=True)
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]
        raise RuntimeError("backbone did not return hidden states")

    def _resolve_marker_position(self, token_ids: torch.Tensor, marker_id: Optional[int]) -> int:
        if marker_id is not None:
            positions = (token_ids == marker_id).nonzero(as_tuple=True)[0]
            if len(positions):
                return int(positions[0].item())
        if self.cls_token_id is not None:
            positions = (token_ids == self.cls_token_id).nonzero(as_tuple=True)[0]
            if len(positions):
                return int(positions[0].item())
        return 0

    @staticmethod
    def _resolve_last_token_position(attention_mask: torch.Tensor) -> int:
        positions = attention_mask.nonzero(as_tuple=True)[0]
        if len(positions):
            return int(positions[-1].item())
        return 0

    @staticmethod
    def _is_cuda_oom(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "out of memory" in message and "cuda" in message

    def _clear_cuda_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
        if torch_dtype is None or torch_dtype == "auto":
            return None
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        if isinstance(torch_dtype, str):
            candidate = torch_dtype.replace("torch.", "")
            resolved = getattr(torch, candidate, None)
            if isinstance(resolved, torch.dtype):
                return resolved
        raise ValueError(f"Unsupported torch_dtype value: {torch_dtype!r}")

    @staticmethod
    def _resolve_hidden_size(backbone: nn.Module) -> int:
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(backbone.config, "d_model", None)
        if hidden_size is None and hasattr(backbone.config, "text_config"):
            hidden_size = getattr(backbone.config.text_config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(backbone.config.text_config, "d_model", None)
        if hidden_size is None:
            raise ValueError(f"Unable to determine hidden size for backbone {backbone.__class__.__name__}")
        return int(hidden_size)

    @staticmethod
    def _resolve_loader_kind(config: object, requested: str) -> str:
        if requested in {"auto", "encoder", "causal_lm"}:
            if requested != "auto":
                return requested
        else:
            raise ValueError(f"Unsupported backbone_model_class value: {requested!r}")

        model_type = str(getattr(config, "model_type", "")).lower()
        architectures = " ".join(getattr(config, "architectures", []) or []).lower()
        if model_type == "qwen3_5":
            return "causal_lm"
        if "causallm" in architectures or "conditionalgeneration" in architectures:
            return "causal_lm"
        if bool(getattr(config, "is_decoder", False)):
            return "causal_lm"
        return "encoder"

    @staticmethod
    def _resolve_pooling_strategy(backbone: nn.Module, requested: str) -> str:
        if requested in {"marker", "last_token"}:
            return requested
        if requested != "auto":
            raise ValueError(f"Unsupported pooling_strategy value: {requested!r}")

        model_type = str(getattr(backbone.config, "model_type", "")).lower()
        architectures = " ".join(getattr(backbone.config, "architectures", []) or []).lower()
        class_name = backbone.__class__.__name__.lower()
        causal_like_signature = " ".join([model_type, architectures, class_name])
        causal_like_tokens = ("qwen", "llama", "mistral", "gpt", "gemma", "phi")
        if bool(getattr(backbone.config, "is_decoder", False)) or any(
            token in causal_like_signature for token in causal_like_tokens
        ):
            return "last_token"
        return "marker"
