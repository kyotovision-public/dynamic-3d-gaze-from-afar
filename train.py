import argparse
import random
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from dataloader.gafa import create_gafa_dataset
from models.gazenet import GazeNet


def train(opt):
    model = GazeNet(n_frames=opt.n_frames)

    # default training dataset
    train_exp_names = [
        'library/1026_3',
        'library/1028_2',
        'library/1028_5',
        'lab/1013_1',
        'lab/1014_1',
        'kitchen/1022_4',
        'kitchen/1015_4',
        'living_room/004',
        'living_room/005',
        'courtyard/004',
        'courtyard/005',
    ]

    random.shuffle(train_exp_names)
    dset = create_gafa_dataset(n_frames=opt.n_frames, exp_names=train_exp_names, interval=1)
    train_idx, val_idx = np.arange(0, int(len(dset)*0.9)), np.arange(int(len(dset)*0.9), len(dset))
    train_dset = Subset(dset, train_idx)
    validation_dset = Subset(dset, val_idx)

    checkpoint_collback = ModelCheckpoint(monitor="val_loss", save_top_k=1)

    trainer = Trainer(
        default_root_dir=opt.checkpoint,
        callbacks=[checkpoint_collback],
        benchmark=True,
        min_epochs=opt.epoch,
        max_epochs=opt.epoch,
        gpus=opt.gpus,
        strategy="ddp",
        precision=16,
    )

    train_loader = DataLoader(
        train_dset, batch_size=32, num_workers=4, pin_memory=True, shuffle=True
    )
    val_loader = DataLoader(
        validation_dset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--n_frames", type=int, default=7)
    parser.add_argument("--checkpoint", type=str,
                        default="output/")
    parser.add_argument("--gpus", type=int, default=1)
    opt = parser.parse_args()

    train(opt)
