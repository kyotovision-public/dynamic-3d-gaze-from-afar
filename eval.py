import argparse

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from dataloader.gafa import create_gafa_dataset
from models.gazenet import GazeNet


# Settings #####################################

# settings for dataset
test_exp_names = [
        'library/1029_2',
        'lab/1013_2',
        'kitchen/1022_2',
        'living_room/006',
        'courtyard/002',
        'courtyard/003',
        ]

checkpoint = './models/weights/gazenet_GAFA.pth'
# checkpoint = './models/weights/gazenet_AGORA_GAFA.pth'
###############################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=7)
    parser.add_argument("--gpus", type=int, default=1)
    opt = parser.parse_args()

    # load model
    model = GazeNet(n_frames=opt.n_frames)
    model.load_state_dict(torch.load(
        checkpoint, map_location=torch.device("cpu"))['state_dict'])

    # make dataloader
    test_dset = create_gafa_dataset(
        n_frames=opt.n_frames,
        exp_names=test_exp_names,
        interval=7,
    )
    test_loader = DataLoader(test_dset, batch_size=32,
                             num_workers=4, shuffle=False)

    trainer = Trainer(
        benchmark=True,
        gpus=opt.gpus,
        precision=16,
        accelerator='ddp'
    )

    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
