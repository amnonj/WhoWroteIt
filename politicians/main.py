import copy
import random
import data
import nn_config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import time
import os
import json
from common import AuthorNet as an, text_utils
from politicians import datasets
from types import SimpleNamespace

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

train_set_df["coded_text"] = text_utils.conv_char_to_code(train_set_df["masked_text"], char_to_id)

# testing
# train_set_df["decoded_text"] = text_utils.conv_code_to_char(train_set_df["coded_text"], id_to_char)
# print((train_set_df["decoded_text"] == train_set_df["masked_text"]).all())




train_data = train_set_df["coded_text"].tolist()
train_labels = torch.tensor(train_set_df["PageName"].map(author_to_id).tolist(), dtype=torch.long)
train_set = datasets.PostDataset(train_data, train_labels, nn_config.message_len)
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

# Validation setup
val_set_df = filtered_posts_df[filtered_posts_df["split"] == "val"].copy()
val_set_df["coded_text"] = text_utils.conv_char_to_code_and_truncate(val_set_df["masked_text"],
                                                                     char_to_id, nn_config.message_len)

val_data = torch.tensor(val_set_df["coded_text"].tolist(), dtype=torch.long)
val_labels = torch.tensor(val_set_df["PageName"].map(author_to_id).tolist(), dtype=torch.long)
val_set = TensorDataset(val_data, val_labels)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)


def run_validation(model, criterion_val, val_dataloader):
    model.eval()
    m={}
    correct = 0
    total = 0
    logits_list= []
    labels_list = []
    predicted_list = []
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
            logits_list.append(outputs)
            labels_list.append(labels)
            predicted_list.append(predicted)

    m['recall_per_author'] = (np.diag(conf_mat) / conf_mat.sum(axis=1))
    m['balanced_accuracy'] = m['recall_per_author'].mean()
    m['accuracy'] = 100 * correct / total
    m['val_loss_avg'] = val_loss/len(val_dataloader)
    m['conf_mat'] = conf_mat
    m['logits'] = torch.cat(logits_list)
    m['labels'] = torch.cat(labels_list)
    m['predicted'] = torch.cat(predicted_list)
    m['probabilities'] = torch.softmax(m['logits'], dim=1)

    return m


nnconf = {k: v for k, v in vars(nn_config).items() if not k.startswith("_")}
nnconf['char_to_id'] = char_to_id
nnconf['author_to_id'] = author_to_id

model = an.AuthorNet(SimpleNamespace(**nnconf))
weights = torch.tensor(1 / np.bincount(train_labels), dtype=torch.float32)
criterion_train = nn.CrossEntropyLoss(weight=weights)
criterion_val = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = np.inf
best_balanced_accuracy = -np.inf
best_model_state = copy.deepcopy(model.state_dict())
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

    metrics = run_validation(model, criterion_val, val_dataloader)
    print(f"Validation Accuracy: {metrics['accuracy']:.2f}%  Loss: {metrics['val_loss_avg']}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
#    if val_loss_avg < best_val_loss - nn_config.min_delta:
    if metrics['balanced_accuracy'] > best_balanced_accuracy + nn_config.min_delta:
        best_val_loss = metrics['val_loss_avg']
        best_balanced_accuracy = metrics['balanced_accuracy']
        last_improve_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        best_accuracy = metrics['accuracy']
    elif epoch - last_improve_epoch >= nn_config.patience:
        break

train_time = time.perf_counter() - start_time
print(f"Training time: {train_time:.2f} seconds")
print(f"Selected epoch {last_improve_epoch + 1} from {epoch + 1}")
print("Training Finished!")

start_time = time.perf_counter()
model.load_state_dict(best_model_state)
metrics = run_validation(model, criterion_val, val_dataloader)

val_time = time.perf_counter() - start_time
print(f"Validation time: {val_time:.2f} seconds")
print(f"Validation loss: {metrics['val_loss_avg']} Balanced accuracy: {metrics['balanced_accuracy']} \n Confusion Matrix:")
print(metrics['conf_mat'])
print("Validation Finished!")


metrics['post_ids'] = val_set_df["Post_ID"]
directory = "experiments/" + time.strftime("%Y%m%d-%H%M%S")
os.makedirs(directory, exist_ok=False)
torch.save(model.state_dict(), os.path.join(directory,"model.pth"))
with open(os.path.join(directory,'netconfig.json'), 'w') as f:
    json.dump(nnconf, f, indent=4)
np.savez_compressed(os.path.join(directory,'validation.npz'),
                    recall_per_author=metrics['recall_per_author'],
                    balanced_accuracy=metrics['balanced_accuracy'],
                    accuracy=metrics['accuracy'],
                    val_loss_avg = metrics['val_loss_avg'],
                    logits=metrics['logits'].cpu().numpy(),
                    labels=metrics['labels'].cpu().numpy(),
                    predicted = metrics['predicted'].cpu().numpy(),
                    probabilities=metrics['probabilities'].cpu().numpy(),
                    post_ids=metrics['post_ids'],
                    masked_length=val_set_df["masked_length"].to_numpy(),
                    conf_mat=metrics['conf_mat'],
                    epochs=epoch + 1,
                    selected_epoch=last_improve_epoch + 1,
                    train_time=train_time
                    )

model.eval()

# model = AuthorNet()
# model.load_state_dict(torch.load("best_model.pth", weights_only=True))
# model.eval()
