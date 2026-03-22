import torch


def _clamp_noisy_positions(dcls_layer, noisy_positions):
    if len(dcls_layer.dilated_kernel_size) == 1:
        lim = dcls_layer.dilated_kernel_size[0] // 2
        return noisy_positions.clamp(-lim, lim)

    clamped = []
    for i, dilated_size in enumerate(dcls_layer.dilated_kernel_size):
        lim = dilated_size // 2
        clamped.append(noisy_positions.select(0, i).clamp(-lim, lim))
    return torch.stack(clamped, dim=0)


def forward_with_delay_dropout(dcls_layer, layer_input, training, sigma_drop):
    if (not training) or sigma_drop <= 0:
        return dcls_layer(layer_input)

    noisy_positions = dcls_layer.P + torch.randn_like(dcls_layer.P) * sigma_drop
    noisy_positions = _clamp_noisy_positions(dcls_layer, noisy_positions)
    return dcls_layer._conv_forward(
        layer_input, dcls_layer.weight, dcls_layer.bias, noisy_positions, dcls_layer.SIG
    )


def delay_stats(dcls_layer):
    """Return summary stats for learned delay positions P in one DCLS layer."""
    P = dcls_layer.P.detach().float().flatten()
    lim = dcls_layer.dilated_kernel_size[0] // 2

    counts = torch.histc(P, bins=64, min=-lim, max=lim)
    probs = (counts + 1e-8) / (counts.sum() + 64 * 1e-8)
    entropy = -(probs * probs.log2()).sum().item()

    return {
        "delay_mean": P.mean().item(),
        "delay_std": P.std().item(),
        "delay_min": P.min().item(),
        "delay_max": P.max().item(),
        "delay_near_peak_frac": (P.abs() <= 1.0).float().mean().item(),
        "delay_entropy_bits": entropy,
    }


def all_layer_delay_stats(blocks):
    """Return per-layer delay stats as a flat dict for wandb logging."""
    from DCLS.construct.modules import Dcls1d

    logs = {}
    for i, block in enumerate(blocks):
        if isinstance(block[0][0], Dcls1d):
            for k, v in delay_stats(block[0][0]).items():
                logs[f"layer{i}_{k}"] = v
    return logs
