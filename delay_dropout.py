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
