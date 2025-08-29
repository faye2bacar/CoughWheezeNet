import torch
import torch.nn as nn
try:
    import pytorch_lightning as pl
except Exception:
    pl = None
from .config import TrainConfig

class CRNN(nn.Module):
    def __init__(self, n_mels: int, n_classes: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2,2)),
        )
        self.gru = nn.GRU(input_size=(n_mels//8)*128, hidden_size=128,
                           num_layers=1, bidirectional=True, batch_first=True)
        self.head_cls = nn.Linear(256, n_classes)
        self.head_exa = nn.Linear(256, 1)
    def forward(self, x):
        # x: (B,1,T,M)
        z = self.feat(x)  # (B,C,T/8,M/8)
        B,C,T,M = z.shape
        z = z.permute(0,2,1,3).contiguous().view(B,T,C*M)  # (B,T, C*M)
        z,_ = self.gru(z)  # (B,T,256)
        z_avg = z.mean(dim=1)
        logits = self.head_cls(z_avg)
        exa_seq = torch.sigmoid(self.head_exa(z).squeeze(-1))  # (B,T)
        return logits, exa_seq

if pl is not None:
    class LitCRNN(pl.LightningModule):
        def __init__(self, cfg: TrainConfig):
            super().__init__()
            self.save_hyperparameters()
            self.cfg = cfg
            self.model = CRNN(cfg.n_mels, cfg.num_classes)
            self.ce = nn.CrossEntropyLoss()
            self.mse = nn.MSELoss()
            self.lr = cfg.lr
        def training_step(self, batch, batch_idx):
            x,y,ex = batch
            logits, exa = self.model(x)
            loss = self.ce(logits, y) + 0.2*self.mse(exa.mean(dim=1), ex)
            acc = (logits.argmax(1)==y).float().mean()
            self.log_dict({"train/loss":loss, "train/acc":acc}, prog_bar=True)
            return loss
        def validation_step(self, batch, batch_idx):
            x,y,ex = batch
            logits, exa = self.model(x)
            loss = self.ce(logits, y) + 0.2*self.mse(exa.mean(dim=1), ex)
            acc = (logits.argmax(1)==y).float().mean()
            self.log_dict({"val/loss":loss, "val/acc":acc}, prog_bar=True)
        def configure_optimizers(self):
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
            return {"optimizer": opt, "lr_scheduler": sch}
else:
    class LitCRNN(object):
        def __init__(self, *args, **kwargs):
            raise ImportError("pytorch_lightning is required for LitCRNN but is not installed.")
