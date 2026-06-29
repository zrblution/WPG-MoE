"""Stage-D expert warm start."""

from __future__ import annotations

import random
from typing import Callable, Dict, List

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from src.training.dataset import format_user_sample
from src.training.distributed import (
    all_reduce_scalar,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)


class _WarmStartDataset(Dataset):
    def __init__(self, samples: List[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def _sync_gradients(parameters: List[torch.nn.Parameter]) -> None:
    if not is_distributed():
        return
    world_size = float(get_world_size())
    for parameter in parameters:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _warm_start_loss(forced_logits: torch.Tensor, label: torch.Tensor, *, pos_weight: float) -> torch.Tensor:
    logits = forced_logits if forced_logits.dim() > 1 else forced_logits.view(1, -1)
    if logits.shape[-1] == 2:
        class_weight = torch.tensor([1.0, pos_weight], dtype=logits.dtype, device=logits.device)
        return torch.nn.functional.cross_entropy(logits, label.view(-1).long(), weight=class_weight)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits.view(-1),
        label.view(-1),
        pos_weight=torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device),
    )


def _fraction_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    if ratio >= 1.0:
        return total
    return max(min(int(total * ratio), total), 1)


def _take_top_fraction(samples: List[dict], ratio: float, key_fn: Callable[[dict], float]) -> List[dict]:
    count = _fraction_count(len(samples), ratio)
    if count <= 0:
        return []
    return sorted(samples, key=key_fn, reverse=True)[:count]


def _sample_fraction(samples: List[dict], ratio: float, rng: random.Random) -> List[dict]:
    count = _fraction_count(len(samples), ratio)
    if count <= 0:
        return []
    if count >= len(samples):
        return list(samples)
    return rng.sample(samples, k=count)


def _mixed_expert_subset(
    train_samples: List[dict],
    depressed: List[dict],
    *,
    depressed_ratio: float,
    normal_to_depressed_ratio: float,
    rng: random.Random,
) -> List[dict]:
    depressed_subset = _sample_fraction(depressed, depressed_ratio, rng)
    if not depressed_subset:
        return []
    normal_samples = [sample for sample in train_samples if int(sample["label"]) == 0]
    if normal_to_depressed_ratio < 0:
        normal_subset = list(normal_samples)
    else:
        target_normal_count = 0
        if normal_to_depressed_ratio > 0:
            target_normal_count = max(int(len(depressed_subset) * normal_to_depressed_ratio), 1)
        normal_count = min(len(normal_samples), target_normal_count)
        normal_subset = normal_samples if normal_count >= len(normal_samples) else rng.sample(normal_samples, k=normal_count)
    mixed = list(depressed_subset) + list(normal_subset)
    rng.shuffle(mixed)
    return mixed


def _expert_subsets(train_samples: List[dict], config: dict) -> Dict[int, List[dict]]:
    depressed = [sample for sample in train_samples if int(sample["label"]) == 1]
    seed = int(config.get("seed", 42))
    top_ratio = float(config.get("warm_start_top_ratio", 0.30))
    expert3_ratio = float(config.get("warm_start_expert3_ratio", 1.0))
    expert4_ratio = float(config.get("warm_start_expert4_ratio", 1.0))
    expert4_normal_ratio = float(config.get("warm_start_expert4_normal_to_depressed_ratio", -1.0))
    return {
        0: _take_top_fraction(depressed, top_ratio, lambda item: float(item["priors"]["self_disclosure"])),
        1: _take_top_fraction(depressed, top_ratio, lambda item: float(item["priors"]["episode_supported"])),
        2: _take_top_fraction(depressed, top_ratio, lambda item: float(item["priors"]["sparse_evidence"])),
        3: _sample_fraction(depressed, expert3_ratio, random.Random(seed + 3)),
        4: _mixed_expert_subset(
            train_samples,
            depressed,
            depressed_ratio=expert4_ratio,
            normal_to_depressed_ratio=expert4_normal_ratio,
            rng=random.Random(seed + 4),
        ),
    }


def _local_subset(samples: List[dict]) -> List[dict]:
    if not samples or not is_distributed():
        return list(samples)
    dataset = _WarmStartDataset(samples)
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=False,
    )
    return [samples[index] for index in sampler]


def _project_expert_inputs(base_model, sample: dict, device: torch.device) -> List[torch.Tensor]:
    target_dtype = base_model.user_representation.attn_sd.attention.weight.dtype
    risk_post_reps, block_post_reps, segment_reps = base_model._encode_user_posts(
        sample["risk_texts"],
        sample["risk_markers"],
        sample["block_texts"],
        sample["block_markers"],
        sample["global_segment_texts"],
        sample["global_segment_markers"],
    )
    risk_post_reps = risk_post_reps.to(dtype=target_dtype)
    if block_post_reps is not None:
        block_post_reps = block_post_reps.to(dtype=target_dtype)
    segment_reps = [segment.to(dtype=target_dtype) for segment in segment_reps]
    pi_u = sample["pi_u"].to(device=device, dtype=target_dtype)
    crisis = sample["crisis"].to(device=device, dtype=target_dtype)
    stats = sample["stats"].to(device=device, dtype=target_dtype)
    meta_vector = sample["meta_vector"].to(device=device, dtype=target_dtype)
    if getattr(base_model, "disable_weak_prior_inputs", False):
        pi_u = torch.zeros_like(pi_u)
        crisis = torch.zeros_like(crisis)
        meta_vector = meta_vector.clone()
        meta_vector[:4] = 0
    stacked_segments = torch.stack(segment_reps, dim=0).to(dtype=target_dtype)
    z_dict = base_model.user_representation(
        risk_post_reps=risk_post_reps,
        block_post_reps=block_post_reps,
        segment_reps=stacked_segments,
        global_stats=stats,
    )
    projected_meta = base_model.experts.meta_proj(meta_vector)
    z_list = [z_dict["z_sd"], z_dict["z_ep"], z_dict["z_sp"], z_dict["z_mix"], z_dict["z_g"]]
    return [torch.cat([z_item, projected_meta], dim=0).detach() for z_item in z_list]


def warm_start_experts(model, train_samples: List[dict], config: dict) -> Dict[str, List[dict]]:
    device = next(model.parameters()).device
    base_model = model
    history: Dict[str, List[dict]] = {"experts": []}
    subsets = _expert_subsets(train_samples, config)
    max_samples_per_expert = config.get("warm_start_max_samples_per_expert")
    seed = int(config.get("seed", 42))
    use_augmentation = bool(config.get("warm_start_use_augmentation", False))
    force_risk_source = str(config.get("warm_start_force_risk_source", "llm"))
    classifier = base_model.moe_head.classifier
    pos_weight = float(config.get("pos_weight", 1.0))
    epochs = int(config.get("warm_start_epochs", 3))
    warm_start_batch_size = max(int(config.get("warm_start_batch_size", 1)), 1)
    local_subsets: Dict[int, List[dict]] = {}
    cached_samples_by_user: Dict[str, dict] = {}
    cacheable_samples: Dict[str, dict] = {}

    for expert_idx, subset in subsets.items():
        if max_samples_per_expert is not None and len(subset) > int(max_samples_per_expert):
            rng = random.Random(seed + expert_idx)
            subset = rng.sample(subset, k=int(max_samples_per_expert))
        subsets[expert_idx] = subset
        local_subset = _local_subset(subset)
        local_subsets[expert_idx] = local_subset
        for raw_sample in local_subset:
            cacheable_samples[raw_sample["user_id"]] = raw_sample

    base_model.eval()
    cache_progress = tqdm(
        cacheable_samples.values(),
        desc="Stage D Cache",
        leave=False,
        disable=not is_main_process(),
    )
    with torch.no_grad():
        for raw_sample in cache_progress:
            formatted_sample = format_user_sample(
                raw_sample,
                is_training=use_augmentation,
                max_risk_posts=config.get("max_risk_posts"),
                max_global_posts_per_segment=config.get("global_history_max_per_segment"),
                force_risk_source=force_risk_source,
            )
            cached_samples_by_user[raw_sample["user_id"]] = {
                "expert_inputs": _project_expert_inputs(base_model, formatted_sample, device),
                "label": formatted_sample["label"].to(device),
            }

    for expert_idx, subset in subsets.items():
        local_subset = local_subsets[expert_idx]
        if not local_subset:
            history["experts"].append({"expert_idx": expert_idx, "steps": 0, "loss": 0.0})
            continue
        if is_main_process():
            print(
                f"[Stage D] expert={expert_idx} subset_size={len(subset)} local_subset_size={len(local_subset)} "
                f"epochs={epochs} batch_size={warm_start_batch_size}",
                flush=True,
            )
        for param in base_model.parameters():
            param.requires_grad = False
        for param in base_model.experts.experts[expert_idx].parameters():
            param.requires_grad = True
        for param in classifier.parameters():
            param.requires_grad = True
        base_model.experts.experts[expert_idx].train()
        classifier.train()
        trainable_params = list(base_model.experts.experts[expert_idx].parameters()) + list(classifier.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(config.get("warm_start_lr", 1e-4)),
        )
        expert_inputs = torch.stack(
            [cached_samples_by_user[sample["user_id"]]["expert_inputs"][expert_idx] for sample in local_subset],
            dim=0,
        )
        labels = torch.cat([cached_samples_by_user[sample["user_id"]]["label"] for sample in local_subset], dim=0)
        running_loss = 0.0
        steps = 0
        for epoch in range(epochs):
            progress = tqdm(
                range(0, expert_inputs.shape[0], warm_start_batch_size),
                desc=f"Stage D Expert {expert_idx} Epoch {epoch + 1}",
                leave=False,
                disable=not is_main_process(),
            )
            for batch_start in progress:
                batch_end = min(batch_start + warm_start_batch_size, expert_inputs.shape[0])
                batch_inputs = expert_inputs[batch_start:batch_end]
                batch_labels = labels[batch_start:batch_end]
                expert_output = base_model.experts.experts[expert_idx](batch_inputs)
                forced_logit = classifier(expert_output)
                loss = _warm_start_loss(forced_logit, batch_labels, pos_weight=pos_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _sync_gradients(trainable_params)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                running_loss += float(loss.item())
                steps += 1
                progress.set_postfix(loss=f"{running_loss / max(steps, 1):.4f}")
            logged_loss = running_loss
            logged_steps = float(steps)
            if is_distributed():
                logged_loss = all_reduce_scalar(running_loss, device=str(device))
                logged_steps = all_reduce_scalar(float(steps), device=str(device))
            if is_main_process():
                print(
                    f"[Stage D] expert={expert_idx} epoch={epoch + 1} avg_loss={logged_loss / max(logged_steps, 1.0):.4f}",
                    flush=True,
                )
        history["experts"].append(
            {
                "expert_idx": expert_idx,
                "subset_size": len(subset),
                "local_subset_size": len(local_subset),
                "steps": int(logged_steps) if is_distributed() else steps,
                "loss": logged_loss / max(logged_steps, 1.0),
            }
        )
        del expert_inputs, labels
    for param in base_model.parameters():
        param.requires_grad = True
    return history
