from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import soundfile as sf
from .config import TrainConfig
from .features import wav_to_logmel

class SegmentDataset(Dataset):
    def __init__(self, csv_path: str, cfg: TrainConfig):
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        x, sr = sf.read(row.filepath, dtype='float32')
        if sr != self.cfg.sr:
            raise ValueError(f"SR mismatch: {sr} != {self.cfg.sr}")
        start = int(row.start_s*self.cfg.sr)
        end   = int(row.end_s*self.cfg.sr)
        seg = torch.from_numpy(x[start:end])
        mel = wav_to_logmel(seg, self.cfg.sr, self.cfg.n_fft, self.cfg.win_length,
                            self.cfg.hop_length, self.cfg.n_mels, self.cfg.fmin, self.cfg.fmax)
        mel = mel.T.unsqueeze(0)  # (1, T, M) -> (C,T,M)
        y = torch.tensor(row.label, dtype=torch.long)
        exac = torch.tensor(row.get('exac', 0.0), dtype=torch.float32)
        return mel, y, exac

# Optional LightningDataModule omitted here in favor of inline in train.py for brevity
