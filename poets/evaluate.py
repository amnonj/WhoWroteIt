import random
import data
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import json
from common import text_utils
from poets import AuthorNet as an
from types import SimpleNamespace
import time

IN_FOLDER = 'E32_C9_256_S111'
FROM_END = True
DATA_SET = 'val'

directory = "experiments/" + IN_FOLDER
model_state = torch.load(os.path.join(directory, "model.pth"))
with open(os.path.join(directory, 'netconfig.json'), 'r') as f:
    nnconf_dic = json.load(f)
    nnconf = SimpleNamespace(**nnconf_dic)

PAD = 0
random.seed(nnconf.SEED)
np.random.seed(nnconf.SEED)
torch.manual_seed(nnconf.SEED)

# Read data
filtered_posts_df = data.load_filtered_posts()

# prepare coding dictionaries
train_set_df = filtered_posts_df[filtered_posts_df["split"] == "train"].copy()
char_to_id = nnconf.char_to_id
author_to_id = nnconf.author_to_id
num_authors = len (author_to_id)

# Validation setup
val_set_df = filtered_posts_df[filtered_posts_df["split"] == DATA_SET].copy()
val_set_df["coded_text"] = text_utils.conv_char_to_code_and_truncate(val_set_df["masked_text"], char_to_id,
                                                                     nnconf.message_len, from_end=FROM_END)

val_data = torch.tensor(val_set_df["coded_text"].tolist(), dtype=torch.long)
val_labels = torch.tensor(val_set_df["PageName"].map(author_to_id).tolist(), dtype=torch.long)
val_set = TensorDataset(val_data, val_labels)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)
criterion_val = nn.CrossEntropyLoss()

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

model = an.AuthorNet(SimpleNamespace(**nnconf_dic))

start_time = time.perf_counter()
model.load_state_dict(model_state)
metrics = run_validation(model, criterion_val, val_dataloader)
elapsed_time = time.perf_counter() - start_time
print(f"Validation time: {elapsed_time:.2f} seconds")
print(
    f"Validation loss: {metrics['val_loss_avg']} Balanced accuracy: {metrics['balanced_accuracy']} \n Confusion Matrix:")
print(metrics['conf_mat'])
print("Validation Finished!")

np.savez_compressed(os.path.join("experiments", IN_FOLDER, DATA_SET + ('_end' if FROM_END else '_start') + '.npz'),
                    recall_per_author = metrics['recall_per_author'],
                    balanced_accuracy = metrics['balanced_accuracy'],
                    accuracy = metrics['accuracy'],
                    val_loss_avg = metrics['val_loss_avg'],
                    logits = metrics['logits'].cpu().numpy(),
                    labels = metrics['labels'].cpu().numpy(),
                    predicted = metrics['predicted'].cpu().numpy(),
                    probabilities=metrics['probabilities'].cpu().numpy(),
                    conf_mat=metrics['conf_mat'],
                    masked_length=val_set_df["masked_length"].to_numpy(),
                    post_ids=val_set_df["Post_ID"].to_numpy()
                    )

model.eval()

