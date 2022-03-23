import os
import pickle
from collections import defaultdict
import cv2

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


class GazeSeqDataset(Dataset):
    def __init__(self, video_path, n_frames, interval):
        self.video_path = video_path
        self.n_frames = n_frames

        # load annotation
        with open(os.path.join(video_path, 'annotation.pickle'), "rb") as f:
            anno_data = pickle.load(f)
        self.bodys = anno_data["bodys"]
        self.heads = anno_data["heads"]
        self.gazes = anno_data["gazes"]
        self.R_cam = anno_data["R_cam"]
        self.t_cam = anno_data["t_cam"]
        self.body_pos = anno_data["body_pos"]
        self.head_pos = anno_data["head_pos"]
        self.img_index = anno_data['index']
        self.keypoints = anno_data['keypoins']

        # abort if no data
        if len(self.gazes) < n_frames:
            self.valid_index = []
            return

        # extract successive frames
        self.valid_index = []
        for i in range(0, len(self.img_index) - self.n_frames, interval):
            if self.img_index[i] + self.n_frames - 1 == self.img_index[i + self.n_frames - 1] and i < len(self.gazes):
                self.valid_index.append(i)
        self.valid_index = np.array(self.valid_index)

        # Head boundig box changed to relative to chest
        self.head_bb = np.vstack(anno_data['head_bb']).astype(np.float32)
        self.body_bb = np.vstack(anno_data['body_bb']).astype(np.float32)
        self.height = self.body_bb[:, 3]
        self.head_bb[:, 0] -= self.body_bb[:, 0]
        self.head_bb[:, 1] -= self.body_bb[:, 1]
        self.head_bb[:, [0, 2]] /= self.body_bb[:, 2][:, None]
        self.head_bb[:, [1, 3]] /= self.body_bb[:, 3][:, None]

        # calculate body velocity
        self.body_bb_center = (self.body_bb[:, [0, 1]] + self.body_bb[:, [2, 3]] / 2)
        norm_body_center = self.body_bb_center / self.body_bb[:, [2, 3]]
        self.body_dv = norm_body_center - np.roll(norm_body_center, shift=1, axis=0)

        # calculate 3D body velocity
        self.body_dv_3d = self.body_pos - np.roll(self.body_pos, shift=1, axis=0)

        # image transform for body image
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.valid_index)

    def transform(self, item_allframe):
        image = torch.stack(item_allframe['image'])
        head_dir = np.stack(item_allframe['head_dir']).copy()
        body_dir = np.stack(item_allframe['body_dir']).copy()
        gaze_dir = np.stack(item_allframe['gaze_dir']).copy()
        head_bb = np.stack(item_allframe['head_bb']).copy()
        body_dv = np.stack(item_allframe['body_dv']).copy()
        body_dv_3d = np.stack(item_allframe['body_dv_3d']).copy()
        height = np.stack(item_allframe['height']).copy()

        # create mask of head bounding box
        head_mask = torch.zeros(image.shape[0], 1, image.shape[2], image.shape[3])
        head_bb_int = head_bb.copy()
        head_bb_int[:, [0, 2]] *= image.shape[3]
        head_bb_int[:, [1, 3]] *= image.shape[2]
        head_bb_int[:, 2] += head_bb_int[:, 0]
        head_bb_int[:, 3] += head_bb_int[:, 1]
        head_bb_int = head_bb_int.astype(np.int64)
        head_bb_int[head_bb_int < 0] = 0
        for i_f in range(head_mask.shape[0]):
            head_mask[i_f, :, head_bb_int[i_f, 1]:head_bb_int[i_f, 3], head_bb_int[i_f, 0]:head_bb_int[i_f, 2]] = 1

        ret_item = {
            'image': image,
            'head_dir': torch.from_numpy(head_dir),
            'body_dir': torch.from_numpy(body_dir),
            'gaze_dir': torch.from_numpy(gaze_dir),
            'head_bb': torch.from_numpy(head_bb),
            'head_mask': head_mask,
            'body_dv': torch.from_numpy(body_dv),
            'body_dv_3d': torch.from_numpy(body_dv_3d),
            'R': torch.from_numpy(item_allframe['R'][0]),
            't': torch.from_numpy(item_allframe['t'][0]),
            'height': torch.from_numpy(height),
        }

        return ret_item

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"index {idx} >= len { len(self) }")

        idx = self.valid_index[idx]

        item_allframe = defaultdict(list)
        for j in range(idx, idx+self.n_frames):

            # read image
            img = cv2.imread(os.path.join(self.video_path, f"{self.img_index[j]:06}.jpg"))[:, :, ::-1]

            # transform images
            img = self.normalize(image=img)['image']
            img = torch.from_numpy(img.transpose(2, 0, 1))

            item = {
                "image": img,
                "head_dir": self.heads[j],
                "body_dir": self.bodys[j],
                "gaze_dir": self.gazes[j],
                "head_bb": self.head_bb[j],
                "body_dv": self.body_dv[j],
                "body_dv_3d": self.body_dv_3d[j],
                "R": self.R_cam,
                "t": self.t_cam,
                "height": self.height[j]
            }

            for k, v in item.items():
                item_allframe[k].append(v)

        item_allframe = self.transform(item_allframe)

        return item_allframe


def create_gafa_dataset(n_frames, exp_names, root_dir='./data/preprocessed', interval=1):
    exp_dirs = [os.path.join(root_dir, en) for en in exp_names]

    dset_list = []
    for ed in exp_dirs:
        cameras = sorted(os.listdir(ed))
        for cm in cameras:
            if not os.path.exists(os.path.join(ed, cm, 'annotation.pickle')):
                continue

            dset = GazeSeqDataset(os.path.join(ed, cm), n_frames, interval)

            if len(dset) == 0:
                continue

            dset_list.append(dset)

    return ConcatDataset(dset_list)
