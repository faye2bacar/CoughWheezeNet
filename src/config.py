from dataclasses import dataclass

@dataclass
class TrainConfig:
    sr: int = 16000
    n_fft: int = 512  # pad freq; window=400 pour 25ms
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64
    fmin: int = 20
    fmax: int = 7600
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 50
    num_workers: int = 4
    segment_sec: float = 3.0
    segment_overlap: float = 0.5
    num_classes: int = 3  # {asthme, copd, sain}
