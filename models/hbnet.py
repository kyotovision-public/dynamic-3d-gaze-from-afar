from itertools import chain
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from efficientnet_pytorch import EfficientNet
from models.utils import compute_mae, vMFLayer
from models.loss import compute_basic_cos_loss, compute_kappa_vMF3_loss


class SharedFeatureNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        model = EfficientNet.from_name('efficientnet-b0', include_top=False)
        self.stem = nn.Sequential(*list(model.children())[:2])

    def forward(self, image):
        return self.stem(image)


class BodyNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.body_branch = EfficientNet.from_name('efficientnet-b0', include_top=False)
        self.body_branch._conv_stem = nn.Identity()
        self.body_branch._bn0 = nn.Identity()

    def forward(self, feature):
        return self.body_branch(feature)


class HeadNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.head_branch = EfficientNet.from_name('efficientnet-b0', include_top=False)
        self.head_branch._conv_stem = nn.Identity()
        self.head_branch._bn0 = nn.Identity()

    def forward(self, feature):
        return self.head_branch(feature)


class AttentionFromHeadBB(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 128 * 96),
        )

    def forward(self, bb):
        x = self.fc(bb)
        x = x.reshape(x.shape[0], 1, 128, 96)
        return x


class TrajNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
        )

    def forward(self, dv):
        x = self.fc(dv)
        return x


class TemporalAlignmentNet(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.temporal_align_net = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.temporal_align_net.flatten_parameters()

    def forward(self, feature_list):
        feature = torch.cat(feature_list, dim=-1)

        return self.temporal_align_net(feature)[0]


class HBNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.featurenet = SharedFeatureNet()
        self.headnet = HeadNet()
        self.bodynet = BodyNet()
        self.head_vmf = vMFLayer(in_dim=64, hidden_dim=16)
        self.body_vmf = vMFLayer(in_dim=64, hidden_dim=16)
        self.trajnet = TrajNet()
        self.temporal_alignment = TemporalAlignmentNet(1280*2 + 32, 64)

        self.automatic_optimization = False

    def forward(self, image, head_mask, body_dv):
        n_batch, n_frame, n_channel, h, w = image.shape

        image = image.reshape(n_batch * n_frame, n_channel, h, w)
        head_mask = head_mask.reshape(n_batch * n_frame, head_mask.shape[2], h, w)

        # extract low-level features
        shared_feature = self.featurenet(image)

        # get attention map from head bounding box
        attention_map = F.avg_pool2d(head_mask, 2)

        # apply attention for head feature
        head_feature = shared_feature * attention_map

        # estimate head and body orientations
        head_feature = self.headnet(head_feature)
        body_feature = self.bodynet(shared_feature)

        head_feature = head_feature.reshape(n_batch, n_frame, 1280)
        body_feature = body_feature.reshape(n_batch, n_frame, 1280)

        # extract features from TrajeNet
        traj_feature = self.trajnet(body_dv)

        # temporaly refine the estimates
        feature = self.temporal_alignment([head_feature, body_feature, traj_feature])
        head_dir, head_kappa = self.head_vmf(feature)
        body_dir, body_kappa = self.body_vmf(feature)

        head_output = {
            'direction': head_dir,
            'kappa': head_kappa
        }
        body_output = {
            'direction': body_dir,
            'kappa': body_kappa,
        }

        return head_output, body_output
