import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class AuthorNet(nn.Module):
    def __init__(self, nn_config_obj):
        super(AuthorNet, self). __init__()
        self.nn_config_obj = nn_config_obj

        if nn_config_obj.USE_1D:
            self.embed = nn.Embedding(len(nn_config_obj.char_to_id),nn_config_obj.EMBEDDING_LENGTH, padding_idx=0)
            self.conv1 = nn.Conv1d(nn_config_obj.EMBEDDING_LENGTH,nn_config_obj.CONV1_OUT,nn_config_obj.CONV1_KERNEL,
                                   nn_config_obj.CONV1_STRIDE)
            if nn_config_obj.DO_CONV2:
                self.conv2 = nn.Conv1d(nn_config_obj.CONV1_OUT,nn_config_obj.CONV2_OUT,nn_config_obj.CONV2_KERNEL,
                                   nn_config_obj.CONV2_STRIDE)
            else:
                self.conv2 = None
        else:
            self.embed = None
            self.conv1 = None
            self.conv2 = None

        if nn_config_obj.USE_2D:
            self.embed_2D = nn.Embedding(len(nn_config_obj.char_to_id),nn_config_obj.EMBEDDING_LENGTH_2D, padding_idx=0)
            self.conv1_2D = nn.Conv2d(nn_config_obj.EMBEDDING_LENGTH_2D,nn_config_obj.CONV1_OUT_2D,
                                      nn_config_obj.CONV1_KERNEL_2D, 1)
            if nn_config_obj.DO_CONV2_2D:
                self.conv2_2D = nn.Conv2d(nn_config_obj.CONV1_OUT_2D,nn_config_obj.CONV2_OUT_2D,
                                          nn_config_obj.CONV2_KERNEL_2D, 1)
            else:
                self.conv2_2D = None
        else:
            self.embed_2D = None
            self.conv1_2D = None
            self.conv2_2D = None

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=nn_config_obj.DROPOUT_RATIO)
        self.num_authors = len(nn_config_obj.author_to_id)
        FC_IN_1D = (2 * nn_config_obj.POOL_REGIONS *
                        (nn_config_obj.CONV2_OUT if nn_config_obj.DO_CONV2 else nn_config_obj.CONV1_OUT))

        FC_IN_2D = (2 * np.prod(nn_config_obj.POOL_REGIONS_2D) *
                        (nn_config_obj.CONV2_OUT_2D if nn_config_obj.DO_CONV2_2D else nn_config_obj.CONV1_OUT_2D))

        self.fc = nn.Linear(int(nn_config_obj.USE_1D * FC_IN_1D + nn_config_obj.USE_2D * FC_IN_2D), self.num_authors)

    def forward(self, x, x_2d):
        nnconf = self.nn_config_obj
        features = []
        if nnconf.USE_1D:
            lengths = (x != 0).sum(dim=1)  # [batch_size]
            length_offset = nnconf.CONV1_KERNEL - 1 + nnconf.DO_CONV2 * (nnconf.CONV2_KERNEL - 1)
            effective_lengths = (lengths - length_offset).clamp(min=0)
            cut_points = (torch.arange(nnconf.POOL_REGIONS + 1, device=x.device) *
                          effective_lengths[:, None] // nnconf.POOL_REGIONS)
            positions = torch.arange(nnconf.message_len - length_offset, device=x.device)[None, None, :]
            padding_mask = (cut_points[:,:-1,None] <= positions) & (positions < cut_points[:, 1:, None])

            x = self.embed(x)
            x = self.conv1(x.transpose(1, 2))
            x = self.relu(x)
            if self.nn_config_obj.DO_CONV2:
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

        if nnconf.USE_2D:
            valid = (x_2d != 0).unsqueeze(1).float()
            z = self.embed_2D(x_2d).permute(0, 3, 1, 2)
            z = self.conv1_2D(z)
            z = self.relu(z)
            valid = F.max_pool2d(valid, kernel_size=nnconf.CONV1_KERNEL_2D, stride=1)

            if nnconf.DO_CONV2_2D:
                z = self.conv2_2D(z)
                z = self.relu(z)
                valid = F.max_pool2d(valid, kernel_size=nnconf.CONV2_KERNEL_2D, stride=1)
            mask = valid > 0
            max_z = F.adaptive_max_pool2d(z.masked_fill(~mask, torch.finfo(z.dtype).min),nnconf.POOL_REGIONS_2D)
            max_z = max_z.masked_fill(F.adaptive_max_pool2d(valid, nnconf.POOL_REGIONS_2D) == 0, 0)
            mean_z = (F.adaptive_avg_pool2d(z * mask, nnconf.POOL_REGIONS_2D)
                    / F.adaptive_avg_pool2d(valid, nnconf.POOL_REGIONS_2D).clamp(min=1e-8))
            features.append(torch.cat((max_z, mean_z), dim=1).flatten(1))

        x = torch.cat(features, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
