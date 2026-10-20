import data
import nn_config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
PAD = 0

# Read data
filtered_posts_df = data.load_filtered_posts()
# Save to CSV for debugging.
# filtered_posts_df.to_csv("filtered_posts_df.csv", index=False, encoding="utf-8-sig")

# prepare coding dictionaries
train_set_df = filtered_posts_df[filtered_posts_df["split"] == "train"].copy()
char_set = ["PAD", "UNK"] + sorted(set(train_set_df["masked_text"].str.cat(sep=" ")))
id_to_char = {index: value for index, value in enumerate(char_set)}
char_to_id = {value: index for index, value in enumerate(char_set)}
politicians = sorted(train_set_df["PageName"].unique())
author_to_id = {name: i for i, name in enumerate(politicians)}
id_to_author = {i: name for i, name in enumerate(politicians)}


def conv_char_to_code(data_text, conv_dict, msg_len):
    def encode(text):
        return [conv_dict.get(char, 1) for char in text[0:msg_len]]+[PAD]*(msg_len-len(text))

    if isinstance(data_text, str):
        return encode(data_text)
    return data_text.apply(encode)


def conv_code_to_char(coded_data, conv_dict):
    def decode(codes):
        return "".join(conv_dict[code] for code in codes)

    if isinstance(coded_data, list):
        return decode(coded_data)
    return coded_data.apply(decode)


train_set_df["coded_text"] = conv_char_to_code(train_set_df["masked_text"], char_to_id, nn_config.message_len)

# testing
# train_set_df["decoded_text"] = conv_code_to_char(train_set_df["coded_text"], id_to_char)
# print((train_set_df["decoded_text"] == train_set_df["masked_text"]).all())

train_data = torch.tensor(train_set_df["coded_text"].tolist(), dtype=torch.long)
train_labels = torch.tensor(train_set_df["PageName"].map(author_to_id).tolist(), dtype=torch.long)
training_data = TensorDataset(train_data, train_labels)
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)


class AuthorNet(nn.Module):
    def __init__(self):
        super(AuthorNet, self). __init__()
        self.embed = nn.Embedding(len(char_set),32, padding_idx=0)
        self.conv1 = nn.Conv1d(32,64,11,1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128, 7)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1)  # [batch_size]
        padding_mask = (
                torch.arange(502, device=x.device)[None, :]
                < (lengths - 10)[:, None]
        )  # [batch_size, 502]
        x = self.embed(x)
        x = self.conv1(x.transpose(1, 2))
        x = self.relu(x)
        x1 = x.masked_fill(~padding_mask[:, None, :], float("-inf"))
        x1 = x1.max(dim=2).values  # [batch_size, 64]
        x2 = x.masked_fill(~padding_mask[:, None, :], 0)
        x2 = x2.sum(dim=2) / padding_mask.sum(dim=1)[:, None].clamp(min=1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


model = AuthorNet()
model.train()
q, labels = next(iter(train_dataloader))
output = model(q)
print(output.shape)