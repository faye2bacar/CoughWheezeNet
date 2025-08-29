import pytorch_lightning as pl
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from .config import TrainConfig
from .model_crnn import LitCRNN
from .data import SegmentDataset

parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', type=str, required=False)
parser.add_argument('--label_csv', type=str, required=True)
parser.add_argument('--val_csv', type=str, default=None)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--mels', type=int, default=64)
parser.add_argument('--sr', type=int, default=16000)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--mlflow', action='store_true', help='Enable MLflow logging if available')
args = parser.parse_args()

cfg = TrainConfig(sr=args.sr, n_mels=args.mels, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)

if __name__ == "__main__":
    train_ds = SegmentDataset(args.label_csv, cfg)
    val_ds = SegmentDataset(args.val_csv or args.label_csv, cfg)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=args.num_workers)

    model = LitCRNN(cfg)
    ckpt = ModelCheckpoint(dirpath='artifacts', save_last=True, save_top_k=1, monitor='val/acc', mode='max')
    lrmon = LearningRateMonitor(logging_interval='epoch')
    logger = False
    if args.mlflow:
        try:
            from pytorch_lightning.loggers import MLFlowLogger
            logger = MLFlowLogger(experiment_name='coughwheeze_train', tracking_uri='file:./mlruns')
        except Exception:
            logger = False
    trainer = pl.Trainer(max_epochs=cfg.epochs, logger=logger, callbacks=[ckpt, lrmon], accelerator='cpu')
    trainer.fit(model, train_dl, val_dl)
