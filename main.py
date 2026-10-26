import copy
import random
import data
import nn_config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
import time

PAD = 0
random.seed(nn_config.SEED)
np.random.seed(nn_config.SEED)
torch.manual_seed(nn_config.SEED)

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
num_authors = len (politicians)


def conv_char_to_code_and_truncate(data_text, conv_dict, msg_len):
    def encode(text):
        return [conv_dict.get(char, 1) for char in text[0:msg_len]]+[PAD]*(msg_len-len(text))

    if isinstance(data_text, str):
        return encode(data_text)
    return data_text.apply(encode)


def conv_char_to_code(data_text, conv_dict):
    def encode(text):
        return [conv_dict.get(char, 1) for char in text]

    if isinstance(data_text, str):
        return encode(data_text)
    return data_text.apply(encode)


def conv_code_to_char(coded_data, conv_dict):
    def decode(codes):
        return "".join(conv_dict[code] for code in codes)

    if isinstance(coded_data, list):
        return decode(coded_data)
    return coded_data.apply(decode)


train_set_df["coded_text"] = conv_char_to_code(train_set_df["masked_text"], char_to_id)

# testing
# train_set_df["decoded_text"] = conv_code_to_char(train_set_df["coded_text"], id_to_char)
# print((train_set_df["decoded_text"] == train_set_df["masked_text"]).all())


class PostDataset(Dataset):

    def __init__(self, text, authors):
        self.text = text
        self.authors = authors

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        this_text = self.text[idx]
        text_len = len(this_text)

        if text_len <= nn_config.message_len:
            coded = this_text + [PAD] * (nn_config.message_len - text_len)
        else:
            start_pointer = random.randrange(text_len - nn_config.message_len + 1)
            coded = this_text[start_pointer:start_pointer + nn_config.message_len]

        return torch.tensor(coded, dtype=torch.long), self.authors[idx]


train_data = train_set_df["coded_text"].tolist()
train_labels = torch.tensor(train_set_df["PageName"].map(author_to_id).tolist(), dtype=torch.long)
train_set = PostDataset(train_data, train_labels)
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

# Validation setup
val_set_df = filtered_posts_df[filtered_posts_df["split"] == "val"].copy()
val_set_df["coded_text"] = conv_char_to_code_and_truncate(val_set_df["masked_text"], char_to_id, nn_config.message_len)

val_data = torch.tensor(val_set_df["coded_text"].tolist(), dtype=torch.long)
val_labels = torch.tensor(val_set_df["PageName"].map(author_to_id).tolist(), dtype=torch.long)
val_set = TensorDataset(val_data, val_labels)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)


class AuthorNet(nn.Module):
    def __init__(self):
        super(AuthorNet, self). __init__()
        self.embed = nn.Embedding(len(char_set),nn_config.EMBEDDING_LENGTH, padding_idx=0)
        self.conv1 = nn.Conv1d(nn_config.EMBEDDING_LENGTH,nn_config.CONV1_OUT,nn_config.CONV1_KERNEL,
                               nn_config.CONV1_STRIDE)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(nn_config.CONV1_OUT,nn_config.CONV2_OUT,nn_config.CONV2_KERNEL,
                               nn_config.CONV2_STRIDE)
        self.dropout = nn.Dropout(p=nn_config.DROPOUT_RATIO)
        self.fc = nn.Linear(2*nn_config.FC_IN, num_authors)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1)  # [batch_size]
        length_offset = nn_config.CONV1_KERNEL - 1 + nn_config.DO_CONV2 * (nn_config.CONV2_KERNEL - 1)
        padding_mask = (
                torch.arange(nn_config.message_len - length_offset, device=x.device)[None, :]
                < (lengths - length_offset)[:, None]
        )
        x = self.embed(x)
        x = self.conv1(x.transpose(1, 2))
        x = self.relu(x)
        if nn_config.DO_CONV2:
            x = self.conv2(x)
            x = self.relu(x)

        x1 = x.masked_fill(~padding_mask[:, None, :], float("-inf"))
        x1 = x1.max(dim=2).values
        x2 = x.masked_fill(~padding_mask[:, None, :], 0)
        x2 = x2.sum(dim=2) / padding_mask.sum(dim=1)[:, None].clamp(min=1)
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


model = AuthorNet()
weights = torch.tensor(1 / np.bincount(train_labels), dtype=torch.float32)
criterion_train = nn.CrossEntropyLoss(weight=weights)
criterion_val = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = np.inf
best_balanced_accuracy = -np.inf
best_model_state = copy.deepcopy(model.state_dict())
best_conf_mat = []
best_accuracy = 0
start_time = time.perf_counter()
last_improve_epoch = -1
for epoch in range(nn_config.max_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()           # איפוס הגרדיאנטים
        outputs = model(inputs)          # Forward Pass
        loss = criterion_train(outputs, labels) # חישוב המחיר
        loss.backward()                  # Backward Pass (חישוב גרדיאנטים)
        optimizer.step()                 # עדכון משקולות
        train_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {train_loss/len(train_dataloader)}")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        val_loss = 0.0
        conf_mat = np.zeros([num_authors,num_authors],dtype=np.int64)
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            loss = criterion_val(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for p, l in zip(predicted, labels):
                conf_mat[l, p] += 1

    recall_per_author = (np.diag(conf_mat) / conf_mat.sum(axis=1))
    balanced_accuracy = recall_per_author.mean()
    accuracy = 100 * correct / total
    val_loss_avg = val_loss/len(val_dataloader)
    print(f'Validation Accuracy: {accuracy:.2f}%  Loss: {val_loss_avg}')
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
#    if val_loss_avg < best_val_loss - nn_config.min_delta:
    if balanced_accuracy > best_balanced_accuracy + nn_config.min_delta:
        best_val_loss = val_loss_avg
        best_balanced_accuracy = balanced_accuracy
        last_improve_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        best_conf_mat = conf_mat
        best_accuracy = accuracy
    elif epoch - last_improve_epoch >= nn_config.patience:
        break

elapsed_time = time.perf_counter() - start_time
print(f"Training time: {elapsed_time:.2f} seconds")
print(f"Best validation loss: {best_val_loss} Best balanced accuracy: {best_balanced_accuracy} at epoch {last_improve_epoch + 1}\n Confusion Matrix:")
print(best_conf_mat)
print("Training Finished!")
model.load_state_dict(best_model_state)
model.eval()
torch.save(model.state_dict(), "best_model.pth")

# model = AuthorNet()
# model.load_state_dict(torch.load("best_model.pth", weights_only=True))
# model.eval()
