import torch.nn as nn
import torch


class AuthorNet(nn.Module):
    def __init__(self, nn_config_obj):
        super(AuthorNet, self). __init__()
        self.nn_config_obj = nn_config_obj
        self.embed = nn.Embedding(len(nn_config_obj.char_to_id),nn_config_obj.EMBEDDING_LENGTH, padding_idx=0)
        self.conv1 = nn.Conv1d(nn_config_obj.EMBEDDING_LENGTH,nn_config_obj.CONV1_OUT,nn_config_obj.CONV1_KERNEL,
                               nn_config_obj.CONV1_STRIDE)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(nn_config_obj.CONV1_OUT,nn_config_obj.CONV2_OUT,nn_config_obj.CONV2_KERNEL,
                               nn_config_obj.CONV2_STRIDE)
        self.dropout = nn.Dropout(p=nn_config_obj.DROPOUT_RATIO)
        self.num_authors = len(nn_config_obj.author_to_id)
        self.fc = nn.Linear(2 * nn_config_obj.FC_IN * nn_config_obj.POOL_REGIONS, self.num_authors)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1)  # [batch_size]
        length_offset = self.nn_config_obj.CONV1_KERNEL - 1 + self.nn_config_obj.DO_CONV2 * (self.nn_config_obj.CONV2_KERNEL - 1)
        cut_points = (torch.arange(self.nn_config_obj.POOL_REGIONS + 1, device=x.device) *
                      (lengths - length_offset)[:,None]//self.nn_config_obj.POOL_REGIONS)
        positions = torch.arange(self.nn_config_obj.message_len - length_offset, device=x.device)[None, None, :]
        padding_mask = (cut_points[:,:-1,None] <= positions) & (positions < cut_points[:,1:,None])

        x = self.embed(x)
        x = self.conv1(x.transpose(1, 2))
        x = self.relu(x)
        if self.nn_config_obj.DO_CONV2:
            x = self.conv2(x)
            x = self.relu(x)

        x1 = x[:,:,None,:].masked_fill(~padding_mask[:, None,:, :], float("-inf"))
        x1 = x1.max(dim=3).values
        x2 = x[:,:,None,:].masked_fill(~padding_mask[:, None, :, :], 0)
        x2 = x2.sum(dim=3) / padding_mask.sum(dim=2)[:, None].clamp(min=1)
        x = torch.cat((x1, x2), dim=1).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
