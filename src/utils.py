import torch

def pad_or_trim(mel: torch.Tensor, T: int):
    # mel: (T0, M) -> (T, M)
    T0 = mel.size(0)
    if T0 == T:
        return mel
    if T0 > T:
        return mel[:T]
    pad = torch.zeros(T-T0, mel.size(1), device=mel.device)
    return torch.cat([mel, pad], dim=0)
