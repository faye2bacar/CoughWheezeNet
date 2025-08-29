import torch
import torchaudio as ta

def wav_to_logmel(x: torch.Tensor, sr: int, n_fft: int, win_length: int,
                   hop_length: int, n_mels: int, fmin: int, fmax: int) -> torch.Tensor:
    """x: (samples,) float32 -1..1 -> (T, M) log-mel"""
    # Use torchaudio's MelSpectrogram for broad version compatibility
    mel_spec = ta.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=fmin,
        f_max=fmax,
        n_mels=n_mels,
        center=True,
        power=2.0,
        norm=None,
    )
    # Input expects shape (B, T)
    X = x.unsqueeze(0)
    mel = mel_spec(X)  # (1, n_mels, T)
    mel = mel.squeeze(0).transpose(0, 1).clamp_min_(1e-10)  # (T, n_mels)
    logmel = torch.log1p(mel)
    return logmel
