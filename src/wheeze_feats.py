import torch

def wheeze_index(mag: torch.Tensor, sr: int, hop_length: int, band=(100,2000)):
    """mag: (F,T) amplitude spectrum (lin). Retourne indice par frame (T,)"""
    f = torch.linspace(0, sr//2, steps=mag.size(0), device=mag.device)
    lo, hi = band
    band_mask = (f>=lo) & (f<=hi)
    band_pow = (mag[band_mask]**2).sum(dim=0)
    tot_pow  = (mag**2).sum(dim=0).clamp_min(1e-9)
    tonal = (mag.max(dim=0).values / (mag.mean(dim=0)+1e-9))
    return (band_pow/tot_pow) * tonal
