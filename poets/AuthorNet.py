import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

PAD = 0

class AuthorNet(nn.Module):
    def __init__(self, cfg):
        super(AuthorNet, self). __init__()
        self.cfg = cfg

        if cfg.USE_1D:
            self.embed = nn.Embedding(len(cfg.char_to_id),cfg.EMBEDDING_LENGTH, padding_idx=0)
            self.conv1 = nn.Conv1d(cfg.EMBEDDING_LENGTH,cfg.CONV1_OUT,cfg.CONV1_KERNEL,
                                   cfg.CONV1_STRIDE)
            if cfg.DO_CONV2:
                self.conv2 = nn.Conv1d(cfg.CONV1_OUT,cfg.CONV2_OUT,cfg.CONV2_KERNEL,
                                   cfg.CONV2_STRIDE)
            else:
                self.conv2 = None
        else:
            self.embed = None
            self.conv1 = None
            self.conv2 = None

        if cfg.USE_2D:
            self.embed_2D = nn.Embedding(len(cfg.char_to_id),cfg.EMBEDDING_LENGTH_2D, padding_idx=0)
            self.conv1_2D = nn.Conv2d(cfg.EMBEDDING_LENGTH_2D,cfg.CONV1_OUT_2D,
                                      cfg.CONV1_KERNEL_2D, 1)
            if cfg.DO_CONV2_2D:
                self.conv2_2D = nn.Conv2d(cfg.CONV1_OUT_2D * (1 + cfg.DO_POOL1_2D),cfg.CONV2_OUT_2D,
                                          cfg.CONV2_KERNEL_2D, 1)
            else:
                self.conv2_2D = None
            if cfg.DO_POOL1_2D:
                self.max_pool1_2D = nn.AdaptiveMaxPool2d(cfg.POOL1_REGIONS_2D)
                self.avg_pool1_2D = nn.AdaptiveAvgPool2d(cfg.POOL1_REGIONS_2D)
            else:
                self.max_pool1_2D = None
                self.avg_pool1_2D = None
            self.max_pool2_2D = nn.AdaptiveMaxPool2d(cfg.POOL2_REGIONS_2D)
            self.avg_pool2_2D = nn.AdaptiveAvgPool2d(cfg.POOL2_REGIONS_2D)
        else:
            self.embed_2D = None
            self.conv1_2D = None
            self.conv2_2D = None
            self.max_pool1_2D = None
            self.avg_pool1_2D = None
            self.max_pool2_2D = None
            self.avg_pool2_2D = None

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=cfg.DROPOUT_RATIO)
        self.num_authors = len(cfg.author_to_id)
        FC_IN_1D = (2 * cfg.POOL_REGIONS *
                        (cfg.CONV2_OUT if cfg.DO_CONV2 else cfg.CONV1_OUT))

        FC_IN_2D = (2 * np.prod(cfg.POOL2_REGIONS_2D) *
                        (cfg.CONV2_OUT_2D if cfg.DO_CONV2_2D else cfg.CONV1_OUT_2D * (1 + cfg.DO_POOL1_2D)))

        self.fc = nn.Linear(int(cfg.USE_1D * FC_IN_1D + cfg.USE_2D * FC_IN_2D), self.num_authors)

    def forward(self, x, x_2d):
        cfg = self.cfg
        features = []
        if cfg.USE_1D:
            lengths = (x != 0).sum(dim=1)  # [batch_size]
            length_offset = cfg.CONV1_KERNEL - 1 + cfg.DO_CONV2 * (cfg.CONV2_KERNEL - 1)
            effective_lengths = (lengths - length_offset).clamp(min=0)
            cut_points = (torch.arange(cfg.POOL_REGIONS + 1, device=x.device) *
                          effective_lengths[:, None] // cfg.POOL_REGIONS)
            positions = torch.arange(cfg.message_len - length_offset, device=x.device)[None, None, :]
            padding_mask = (cut_points[:,:-1,None] <= positions) & (positions < cut_points[:, 1:, None])

            x = self.embed(x)
            x = self.conv1(x.transpose(1, 2))
            x = self.relu(x)
            if cfg.DO_CONV2:
                x = self.conv2(x)
                x = self.relu(x)
            region_sizes = padding_mask.sum(dim=2)
            x1 = x[:,:,None,:].masked_fill(~padding_mask[:, None,:, :], torch.finfo(x.dtype).min)
            x1 = x1.max(dim=3).values
            x1 = x1.masked_fill(region_sizes[:, None, :] == 0, 0)
            x2 = x[:,:,None,:].masked_fill(~padding_mask[:, None, :, :], 0)
            x2 = x2.sum(dim=3) / region_sizes[:, None, :].clamp(min=1)
            x_1d = torch.cat((x1, x2), dim=1).flatten(1)
            features.append(x_1d)

        if cfg.USE_2D:
            valid = (x_2d != PAD).unsqueeze(1).float()
            z = self.embed_2D(x_2d).permute(0, 3, 1, 2)
            z = self.conv1_2D(z)
            z = self.relu(z)
            valid = F.max_pool2d(valid, kernel_size=cfg.CONV1_KERNEL_2D, stride=1)

            if cfg.DO_POOL1_2D:
                mask = valid > 0
                mean_z = (self.avg_pool1_2D(z * mask) / self.avg_pool1_2D(valid).clamp(min=1e-8))
                max_z = self.max_pool1_2D(z.masked_fill(~mask, torch.finfo(z.dtype).min))
                valid = self.max_pool1_2D(valid)
                max_z = max_z.masked_fill(valid == 0, 0)
                z = torch.cat((max_z, mean_z), dim=1)

            if cfg.DO_CONV2_2D:
                z = self.conv2_2D(z)
                z = self.relu(z)
                valid = F.max_pool2d(valid, kernel_size=cfg.CONV2_KERNEL_2D, stride=1)

            mask = valid > 0
            max_z = self.max_pool2_2D(z.masked_fill(~mask, torch.finfo(z.dtype).min))
            max_z = max_z.masked_fill(self.max_pool2_2D(valid) == 0, 0)
            mean_z = (self.avg_pool2_2D(z * mask) / self.avg_pool2_2D(valid).clamp(min=1e-8))
            features.append(torch.cat((max_z, mean_z), dim=1).flatten(1))

        x = torch.cat(features, dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        return x
