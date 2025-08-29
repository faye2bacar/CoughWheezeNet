import torch
import torchaudio

def wav_to_logmel(x: torch.Tensor, sr: int, n_fft: int, win_length: int,
                   hop_length: int, n_mels: int, fmin: int, fmax: int) -> torch.Tensor:
    """x: (samples,) float32 -1..1 -> (T, M) log-mel"""
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=torch.hann_window(win_length, device=x.device),
                      return_complex=True)
    mag = spec.abs() ** 2
    mel_fb = torchaudio.functional.create_fb_matrix(n_fft//2+1, f_min=fmin, f_max=fmax,
                                                    n_mels=n_mels, sample_rate=sr, norm=None,)
    mel = torch.matmul(mag.transpose(0,1), mel_fb.to(mag.device)).clamp_min_(1e-10)
    logmel = torch.log1p(mel)
    return logmel  # (T, M)
