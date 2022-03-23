import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.hbnet import HBNet
from models.utils import get_rotation, compute_mae
from models.loss import compute_basic_cos_loss, compute_kappa_vMF3_loss


class GazeModule(pl.LightningModule):
    def __init__(self, n_frames, n_hidden=128):
        super().__init__()
        assert n_frames % 2 == 1
        self.n_frames = n_frames

        # LSTM
        self.lstm = nn.LSTM(3 * 2, n_hidden, bidirectional=True, num_layers=2)
        self.direction_layer = nn.Sequential(
            nn.Linear(2 * n_hidden * n_frames, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * n_frames),
        )
        self.kappa_layer = nn.Sequential(
            nn.Linear(2 * n_hidden * n_frames, 64),
            nn.ReLU(),
            nn.Linear(64, n_frames),
            nn.Softplus()
        )

    def forward(self, x):
        # LSTM
        fc_out, _ = self.lstm(x)
        fc_out = F.relu(fc_out).view(fc_out.shape[0], -1)

        # estimate mean of vMF
        direction = self.direction_layer(fc_out)
        direction = direction.reshape(x.shape[0], x.shape[1], 3)
        direction /= torch.norm(direction, dim=-1, keepdim=True)
        kappa = self.kappa_layer(fc_out).reshape(x.shape[0], x.shape[1], 1)

        output = {
            'direction': direction,
            'kappa': kappa
        }

        return output


class GazeNet(pl.LightningModule):
    def __init__(self, n_frames):
        super().__init__()
        self.n_frames = n_frames
        self.hbnet = HBNet()
        self.gazemodule = GazeModule(n_frames)

        self.automatic_optimization = False

    def forward(self, img, head_mask, body_dv):
        # get head and body orientation estimates
        head_outputs, body_outputs = self.hbnet(img, head_mask, body_dv)

        reference_rad = head_outputs['direction'][:, self.n_frames // 2]
        dst_rad = torch.zeros_like(reference_rad)
        dst_rad[:, 2] = -1
        R = get_rotation(reference_rad, dst_rad)
        head_dir = torch.einsum('bij,bfj->bfi', R, head_outputs['direction'])
        body_dir = torch.einsum('bij,bfj->bfi', R, body_outputs['direction'])

        # weight direction with kappa
        head_dir = head_dir * head_outputs['kappa']
        body_dir = body_dir * body_outputs['kappa']

        # get gaze direction
        input_value = torch.cat(
            [body_dir, head_dir], dim=2)
        gaze_res = self.gazemodule(input_value)

        gaze_res['direction'] = torch.einsum(
            'bij,bfj->bfi', R.transpose(1, 2), gaze_res['direction'])

        return gaze_res, head_outputs, body_outputs

    def configure_optimizers(self):
        opt_direction = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4)
        opt_kappa = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gazemodule.kappa_layer.parameters()), lr=1e-4)

        return opt_direction, opt_kappa

    def training_step(self, batch, batch_idx):
        image = batch['image']
        head_mask = batch['head_mask']
        body_dv = batch['body_dv']

        gaze_res, head_res, body_res = self.forward(image, head_mask, body_dv)

        # take loss for gaze, head and body orientations
        opt_direction, opt_kappa = self.optimizers()
        if batch_idx % 10 != 0:
            loss = compute_basic_cos_loss(head_res, batch['head_dir']) + \
                compute_basic_cos_loss(body_res, batch['body_dir']) + \
                compute_basic_cos_loss(gaze_res, batch['gaze_dir'])
            loss = loss / 3
            opt_direction.zero_grad()
            self.manual_backward(loss)
            opt_direction.step()
            self.log_dict({"direction_loss": loss}, prog_bar=True)
        else:
            loss = compute_kappa_vMF3_loss(head_res, batch['head_dir']) + \
                compute_kappa_vMF3_loss(body_res, batch['body_dir']) + \
                compute_kappa_vMF3_loss(gaze_res, batch['gaze_dir'])
            loss = loss / 3
            opt_kappa.zero_grad()
            self.manual_backward(loss)
            opt_kappa.step()
            self.log_dict({"kappa_loss": loss}, prog_bar=True)

        mae = compute_mae(gaze_res['direction'], batch['gaze_dir'])
        self.log('train_mae', mae)

        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        head_mask = batch['head_mask']
        body_dv = batch['body_dv']
        gaze_label = batch['gaze_dir']

        gaze_res, _, _ = self.forward(image, head_mask, body_dv)

        # loss for gaze
        loss = compute_kappa_vMF3_loss(gaze_res, gaze_label)
        mae = compute_mae(gaze_res['direction'], gaze_label)

        self.log("val_mae", mae)
        self.log("val_loss", loss)

        return mae

    def validation_epoch_end(self, outputs):
        val_mae = torch.stack([x for x in outputs]).flatten()
        val_mae_mean = val_mae[~torch.isnan(val_mae)].mean()
        print('MAE (validation): ', val_mae_mean)
        self.log("val_mae", val_mae_mean)

    def test_step(self, batch, batch_idx):
        image = batch['image']
        head_mask = batch['head_mask']
        body_dv = batch['body_dv']
        gaze_label = batch['gaze_dir']

        gaze_res, _, _ = self.forward(image, head_mask, body_dv)
        prediction = gaze_res['direction']

        if gaze_label.shape[-1] == 3:
            front_index = torch.arange(gaze_label.shape[0])[gaze_label[:, 0, -1] <= 0]
            back_index = torch.arange(gaze_label.shape[0])[gaze_label[:, 0, -1] > 0]

            # 3D MAE
            mae = compute_mae(prediction, gaze_label)

            # 3D MAE for front facing
            front_mae = compute_mae(prediction[front_index], gaze_label[front_index])

            # 3D MAE for back facing
            back_mae = compute_mae(prediction[back_index], gaze_label[back_index])

            gaze_label_2d = gaze_label[..., :2] / torch.norm(gaze_label[..., :2], dim=-1, keepdim=True)
            prediction_2d = prediction[..., :2] / torch.norm(prediction[..., :2], dim=-1, keepdim=True)

            # 2D MAE
            mae_2d = compute_mae(prediction_2d, gaze_label_2d)
            front_mae_2d = compute_mae(prediction_2d[front_index], gaze_label_2d[front_index])
            back_mae_2d = compute_mae(prediction_2d[back_index], gaze_label_2d[back_index])

        elif gaze_label.shape[-1] == 2:
            gaze_label_2d = gaze_label[..., :2] / torch.norm(gaze_label[..., :2], dim=-1, keepdim=True)
            prediction_2d = prediction[..., :2] / torch.norm(prediction[..., :2], dim=-1, keepdim=True)
            mae = front_mae = back_mae = 0
            front_mae_2d = back_mae_2d = 0

            # 2D MAE
            mae_2d = compute_mae(prediction_2d, gaze_label_2d)
            print(mae_2d)

        return mae, mae_2d, front_mae, front_mae_2d, back_mae, back_mae_2d

    def test_epoch_end(self, outputs):
        mae = np.nanmean([x[0] for x in outputs])
        mae_2d = np.nanmean([x[1] for x in outputs])
        front_mae = np.nanmean([x[2] for x in outputs])
        front_mae_2d = np.nanmean([x[3] for x in outputs])
        back_mae = np.nanmean([x[4] for x in outputs])
        back_mae_2d = np.nanmean([x[5] for x in outputs])

        print('MAE (3D front): ', front_mae)
        print('MAE (2D front): ', front_mae_2d)
        print('MAE (3D back): ', back_mae)
        print('MAE (2D back): ', back_mae_2d)
        print('MAE (3D all): ', mae)
        print('MAE (2D all): ', mae_2d)
