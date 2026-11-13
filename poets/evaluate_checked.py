import random
import data
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
from common import text_utils
from poets import AuthorNet as an
from poets import datasets
from types import SimpleNamespace
import time

IN_FOLDER = '20261108-113317'
DATA_SET = 'val'  # 'val' for verification against saved validation.npz, or 'test'

assert DATA_SET in ('val', 'test')
directory = os.path.join('experiments', IN_FOLDER)

with open(os.path.join(directory, 'netconfig.json'), 'r', encoding='utf-8') as f:
    nnconf = SimpleNamespace(**json.load(f))
model_state = torch.load(os.path.join(directory, 'model.pth'), map_location='cpu', weights_only=True)

random.seed(nnconf.SEED)
np.random.seed(nnconf.SEED)
torch.manual_seed(nnconf.SEED)

# Same dataset loading, encoding, and deterministic validation sampling as main.py.
filtered_posts_df = data.load_filtered_posts()
char_to_id = nnconf.char_to_id
author_to_id = nnconf.author_to_id
num_authors = len(author_to_id)
authors = [name for name, idx in sorted(author_to_id.items(), key=lambda pair: pair[1])]

val_set_df = filtered_posts_df[filtered_posts_df['split'] == DATA_SET].copy()
val_set_df['coded_text'] = text_utils.conv_char_to_code(val_set_df['masked_text'], char_to_id)
val_set_df['coded_lines'] = text_utils.conv_char_to_code(val_set_df['lines'], char_to_id)

val_data_1d = val_set_df['coded_text'].tolist()
val_data_2d = val_set_df['coded_lines'].tolist()
val_labels = torch.tensor(val_set_df['PageName'].map(author_to_id).tolist(), dtype=torch.long)
val_set = datasets.PostDataset(val_data_1d, val_data_2d, val_labels, nnconf, VALIDATION=True)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)
criterion_val = nn.CrossEntropyLoss()


def run_validation(model, criterion_val, val_dataloader):
    # Identical to main.py's run_validation().
    model.eval()
    m = {}
    correct = 0
    total = 0
    logits_list = []
    labels_list = []
    predicted_list = []
    with torch.no_grad():
        val_loss = 0.0
        conf_mat = np.zeros([num_authors, num_authors], dtype=np.int64)
        for inputs_1D, inputs_2D, labels in val_dataloader:
            outputs = model(inputs_1D, inputs_2D)
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

    m['recall_per_author'] = np.diag(conf_mat) / conf_mat.sum(axis=1)
    m['balanced_accuracy'] = m['recall_per_author'].mean()
    m['accuracy'] = 100 * correct / total
    m['val_loss_avg'] = val_loss / len(val_dataloader)
    m['conf_mat'] = conf_mat
    m['logits'] = torch.cat(logits_list)
    m['labels'] = torch.cat(labels_list)
    m['predicted'] = torch.cat(predicted_list)
    m['probabilities'] = torch.softmax(m['logits'], dim=1)
    return m


model = an.AuthorNet(nnconf)
model.load_state_dict(model_state)
start_time = time.perf_counter()
metrics = run_validation(model, criterion_val, val_dataloader)
val_time = time.perf_counter() - start_time

print(f'{DATA_SET} inference time: {val_time:.2f} seconds')
print(f"Loss: {metrics['val_loss_avg']:.6f}, "
      f"Balanced accuracy: {metrics['balanced_accuracy']:.2%}, "
      f"Accuracy: {metrics['accuracy']:.2f}%")
print('     ' + ' '.join(f'{i:3d}' for i in range(num_authors)))
for i, row in enumerate(metrics['conf_mat']):
    numbers = ' '.join(f'{n:3d}' for n in row)
    print(f'{chr(65 + i)}   {numbers}   {authors[i]}')

# Crucial sanity check: VAL should reproduce the stored predictions before evaluating TEST.
if DATA_SET == 'val':
    original_path = os.path.join(directory, 'validation.npz')
    with np.load(original_path) as original:
        assert np.array_equal(val_set_df['Post_ID'].to_numpy(), original['post_ids']), \
            'VAL post IDs or order differ from the original run!'
        assert np.array_equal(metrics['labels'].cpu().numpy(), original['labels']), \
            'VAL labels differ from the original run!'
        max_diff = np.max(np.abs(metrics['logits'].cpu().numpy() - original['logits']))
        print(f'Max absolute difference vs saved validation logits: {max_diff:.8g}')
        assert np.allclose(metrics['logits'].cpu().numpy(), original['logits'],
                           rtol=1e-5, atol=1e-5), \
            'VAL logits differ from original! Check model and preprocessing versions.'
    print('VAL comparison passed: predictions match the original saved validation.npz')

# Do not overwrite the original validation predictions from main.py.
filename = 'validation_recheck.npz' if DATA_SET == 'val' else 'test.npz'
out_path = os.path.join(directory, filename)
np.savez_compressed(
    out_path,
    recall_per_author=metrics['recall_per_author'],
    balanced_accuracy=metrics['balanced_accuracy'],
    accuracy=metrics['accuracy'],
    val_loss_avg=metrics['val_loss_avg'],
    logits=metrics['logits'].cpu().numpy(),
    labels=metrics['labels'].cpu().numpy(),
    predicted=metrics['predicted'].cpu().numpy(),
    probabilities=metrics['probabilities'].cpu().numpy(),
    conf_mat=metrics['conf_mat'],
    masked_length=val_set_df['masked_length'].to_numpy(),
    post_ids=val_set_df['Post_ID'].to_numpy(),
)
print(f'Saved {out_path}')
