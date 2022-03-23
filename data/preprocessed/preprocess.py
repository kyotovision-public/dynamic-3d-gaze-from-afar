import os
import shutil
import pickle
from glob import glob
from PIL import Image
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import Compose as ComposeTransform

from transforms import (
    ImageTransform,
    ExpandBB,
    ExpandBBRect,
    CropBB,
    ReshapeBBRect,
    KeypointsToBB,
)


class GazeSeqDataset(Dataset):
    def __init__(self, video_path, annotation_path, target_width=720):
        self.video_path = video_path
        self.target_width = target_width
        annotation_path = annotation_path

        # load annotation
        with open(annotation_path, "rb") as f:
            anno_data = pickle.load(f)
        self.keypoints = anno_data["keypoints2d"]
        self.bodys = anno_data["chest"]["direction"].astype(np.float32)
        self.heads = anno_data["head"]["direction"].astype(np.float32)
        self.body_pos = anno_data["chest"]["position"].astype(np.float32)
        self.head_pos = anno_data["head"]["position"].astype(np.float32)
        self.gazes = anno_data["gaze"]["direction"].astype(np.float32)
        self.gazes = anno_data["gaze"]["direction"].astype(np.float32)
        self.R_cam = anno_data["R_cam2w"].astype(np.float32)
        self.t_cam = anno_data["t_cam2w"].astype(np.float32)

        # extract valid frames
        self.valid_index = np.where(~np.all(
            self.keypoints[:, [0, 1, 15, 16, 17, 18], :2] == 0, axis=(1,2)))[0]

        self.head_transform = ComposeTransform(
            [
                KeypointsToBB((0, 1, 15, 16, 17, 18)),
                ExpandBB(0.85, -0.2, 0.1, 0.1, "bb"),
                ExpandBBRect("bb"),
            ]
        )


        # define transform for body images
        self.body_transform = ComposeTransform(
            [
                KeypointsToBB(slice(None)),
                ExpandBB(0.15, 0.05, 0.2, 0.2, "bb"),
                ExpandBBRect("bb"),
                ReshapeBBRect((256, 192)),
                CropBB(bb_key="bb"),
                ImageTransform(
                    "image",
                    T.Compose(
                        [
                            T.Resize((256, 192)),
                        ]
                    ),
                ),
            ]
        )

    def __len__(self):
        return len(self.valid_index)

    def run_preprocessing(self, target_dir):
        head_bb_list = []
        body_bb_list = []
        image_dict = dict()
        for idx in tqdm(self.valid_index):
            # read image
            img_path = os.path.join(self.video_path, f"{idx:06}.jpg")
            img = Image.open(img_path)

            # downsample images
            width, height = img.size
            target_height = int(height * (self.target_width / width))
            img = img.resize((self.target_width, target_height))
            img = img.resize((width, height))

            item = {
                "image": img,
                "keypoints": self.keypoints[idx, :, :2],
            }

            # apply image transform
            head_trans = self.head_transform(item)
            head_bb = head_trans['bb']
            head_bb = np.array([head_bb['u'], head_bb['v'], head_bb['w'], head_bb['h']])
            head_bb_list.append(head_bb)

            body_trans = self.body_transform(item) 
            body_bb = body_trans['bb']
            body_bb = np.array([body_bb['u'], body_bb['v'], body_bb['w'], body_bb['h']])
            body_image = body_trans['image']
            body_bb_list.append(body_bb)

            # save images
            image_dict[f"{idx:06}.jpg"] = body_image

        # save images
        with open(os.path.join(target_dir, 'image.pickle'), 'wb') as f:
            pickle.dump(image_dict, f)

        # save annotations
        with open(os.path.join(target_dir, 'annotation.pickle'), 'wb') as f:
            data = {
                'keypoins': self.keypoints[self.valid_index],
                'bodys': self.bodys[self.valid_index],
                'heads': self.heads[self.valid_index],
                'gazes': self.gazes[self.valid_index],
                'body_pos': self.body_pos[self.valid_index],
                'head_pos': self.head_pos[self.valid_index],
                'head_bb': head_bb_list,
                'body_bb': body_bb_list,
                'R_cam': self.R_cam,
                't_cam': self.t_cam,
                'index': self.valid_index
            }

            pickle.dump(data, f)


def main(root_dir='../raw_data/', target_dir='.'):

    # all dataset
    exp_names = [
        'library/1026_3',
        'library/1028_2',
        'library/1028_5',
        'lab/1013_1',
        'lab/1014_1',
        'kitchen/1022_4',
        'kitchen/1015_4',
        'library/1029_2',
        'lab/1013_2',
        'kitchen/1022_2',
        'living_room/004',
        'living_room/005',
        'living_room/006',
        'courtyard/002',
        'courtyard/003',
        'courtyard/004',
        'courtyard/005',
    ]

    exp_dirs = [os.path.join(root_dir, en) for en in exp_names]
    for ed in exp_dirs:
        annot_files = sorted(glob(os.path.join(ed, "*.pkl")))
        video_dirs = [af.replace(".pkl", "") for af in annot_files]

        for vd, af in zip(video_dirs, annot_files):
            dset = GazeSeqDataset(vd, af)
            dirname = os.path.join(target_dir, '/'.join(os.path.normpath(vd).split(os.sep)[-3:]))
            if os.path.exists(dirname):
                print('Skipped', vd)
                continue
            else:
                try:
                    os.makedirs(dirname)
                except FileExistsError:
                    continue

            print(vd)
            try:
                dset.run_preprocessing(target_dir=dirname)
            except:
                shutil.rmtree(dirname)
                continue

if __name__=='__main__':
    main()
