import numpy as np
import os
import json
SELECTION = 'MAJORITY'   # SUM/ MUL / MAX / MAJORITY
SEGMENTS = 1
folders = (
           #'E32_C9_256_S111', 'E32_C9_256_S222', 'E32_C9_256_S333', 'E32_C9_256_S444', 'E32_C9_256_S555',
           'E32_C9_256_S111_end', 'E32_C9_256_S222_end', 'E32_C9_256_S333_end',
           'E32_C9_256_S444_end', 'E32_C9_256_S555_end'
           )

num_nets = len(folders) // SEGMENTS
metrics = []
for net_idx in range(num_nets * SEGMENTS):
    directory = "experiments/" + folders[net_idx]
    metrics.append(np.load(os.path.join(directory, 'validation.npz')))
#with open(os.path.join("experiments/" + folders[0], 'netconfig.json'), 'r') as f:
#    nnconf_dic = json.load(f)
#    author_to_id = nnconf_dic['author_to_id']
#id_to_author = {v: k for k, v in author_to_id.items()}
num_authors = np.size(metrics[0]['probabilities'],1)
labels = metrics[0]['labels']

all_comb = ((np.arange(1, 2 ** num_nets)[:, None] >> np.arange(num_nets)) & 1)
all_comb = np.tile(all_comb,[1,SEGMENTS])
if SELECTION == 'SUM':
    probabilities = np.stack([m['probabilities'] for m in metrics], axis=2)
    ensemble_scores = probabilities @ all_comb.T
    ensemble_scores /= all_comb.sum(axis=1)
elif SELECTION == 'MUL':
    probabilities = np.stack([m['logits'] for m in metrics], axis=2)
    ensemble_scores = probabilities @ all_comb.T
    ensemble_scores /= all_comb.sum(axis=1)
elif SELECTION == 'MAX':
    probabilities = np.stack([m['probabilities'] for m in metrics], axis=2)
    ensemble_scores = np.max(probabilities[:, :, :, None] * all_comb.T[None, None, :, :], axis=2)
elif SELECTION == 'MAJORITY':
    probabilities = np.stack([m['probabilities'] for m in metrics], axis=2)
    probabilities[probabilities == probabilities.max(axis=1, keepdims=True)] += 2*num_nets*SEGMENTS
    ensemble_scores = probabilities @ all_comb.T
    ensemble_scores /= all_comb.sum(axis=1)
else:
    raise Exception("SELECTION is not valid")

predicted = np.argmax(ensemble_scores, axis=1)
correct = predicted == labels[:, None]
accuracy = np.sum(correct, axis=0) / len(labels)
recall = np.array([correct[labels == author_id].mean(axis=0)
                   for author_id in range(num_authors)])
balanced_accuracy = recall.mean(axis=0)

print(f"{'Comb':>4}  {'Nets':<18} {'Count':>5} {'Accuracy':>10} {'Balanced':>10}")
print("-" * 55)

for idx in np.argsort(-balanced_accuracy):
    nets = ",".join(str(i + 1) for i in np.flatnonzero(all_comb[idx]))
    print(f"{idx + 1:4d}  {nets:<18} {all_comb[idx].sum():5d} "
          f"{accuracy[idx]:10.2%} {balanced_accuracy[idx]:10.2%}")
