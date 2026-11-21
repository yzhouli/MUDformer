from __future__ import annotations

import torch


def product_of_experts(
    means: torch.Tensor,
    log_vars: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    include_prior: bool = True,
    min_log_var: float = -12.0,
    max_log_var: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse diagonal Gaussian experts using precision-weighted PoE.

    Args:
        means: ``[..., num_experts, latent_dim]`` expert means.
        log_vars: Tensor with the same shape containing log variances.
        expert_mask: Optional boolean/float tensor ``[..., num_experts]``.
            A false value removes that expert, which is used for missing images.
        include_prior: Include the standard normal prior as an additional expert.

    Returns:
        Joint mean and joint log variance, both ``[..., latent_dim]``.
    """
    if means.shape != log_vars.shape:
        raise ValueError(f"means/log_vars shape mismatch: {means.shape} vs {log_vars.shape}")
    if means.ndim < 2:
        raise ValueError("PoE inputs must have expert and latent dimensions")

    log_vars = log_vars.clamp(min=min_log_var, max=max_log_var)
    precisions = torch.exp(-log_vars)
    if expert_mask is not None:
        expected = means.shape[:-1]
        if expert_mask.shape != expected:
            raise ValueError(f"expert_mask must have shape {expected}, got {expert_mask.shape}")
        precisions = precisions * expert_mask.to(precisions.dtype).unsqueeze(-1)

    joint_precision = precisions.sum(dim=-2)
    if include_prior:
        joint_precision = joint_precision + 1.0
    # Padding positions can have no active expert when the prior expert is ablated.
    # Give only those positions a neutral unit precision; the caller masks them out.
    joint_precision = torch.where(
        joint_precision > 0, joint_precision, torch.ones_like(joint_precision)
    )

    joint_var = joint_precision.reciprocal()
    joint_mean = joint_var * (means * precisions).sum(dim=-2)
    joint_log_var = torch.log(joint_var)
    return joint_mean, joint_log_var
