import numpy as np
import os
import pandas as pd
import json
from itertools import combinations
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold

SELECTION = 'MUL'   # SUM/ MUL / MAX / MAJORITY
SEGMENTS = ['']  #['_start','_end']
DATA_SET = 'validation'
#FOLDERS = ('20261108-113317', '20261108-123838', '20261108-124709')

FOLDERS = (
    # 1D only
    '20261108-113317',
    '20261108-123838',
    '20261108-124709',

    # 2D only
    '20261109-224015',
    '20261110-231445',
    '20261109-212906',
    '20261109-213902',

    # Combined: new modular pooling
    '20261110-232829',
    '20261111-141613',
    '20261111-002142',
    '20261111-010609',
    '20261111-005359',
    '20261110-233042',
    '20261111-141949',
    '20261111-024938',

    # Combined: previous architectures
    '20261109-224724',
    '20261109-230443',
    '20261109-211521',
    '20261109-215802',
    '20261109-224347',
)

MAX_NETS = 4
COMB = [387]  #None   # None for all or a list
#BASE_COMB = 1282
#COMB = [BASE_COMB | (1 << i) for i in range(len(FOLDERS)) if not (BASE_COMB & (1 << i))]

num_seg = len(SEGMENTS)
num_nets = len(FOLDERS)
metrics = []

folder_names = FOLDERS * num_seg
file_names = [DATA_SET + suffix + '.npz' for suffix in SEGMENTS for _ in range(num_nets)]
for net_idx in range(num_nets * num_seg):
    metrics.append(np.load(os.path.join("experiments", folder_names[net_idx], file_names[net_idx])))
assert all(
    np.array_equal(metrics[0]['post_ids'], m['post_ids'])
    for m in metrics[1:]
), "Post IDs or order differ between files"

num_authors = np.size(metrics[0]['probabilities'],1)
labels = metrics[0]['labels']

#################################
# Optimize weights of the selected ensemble (1-based net indices)
WEIGHTED_NETS = (1, 2, 8, 9)
N_FOLDS = 5

assert SEGMENTS == ['']
assert all(np.array_equal(m['labels'], labels) for m in metrics)

# [poems, authors, networks]
weighted_logits = np.stack(
    [metrics[i - 1]['logits'] for i in WEIGHTED_NETS], axis=2
).astype(np.float64)

n_weighted = len(WEIGHTED_NETS)
equal_weights = np.full(n_weighted, 1.0 / n_weighted)


def weighted_scores(z, w):
    return np.einsum('nkm,m->nk', z, w)


def balanced_ce_and_gradient(w, z, y):
    scores = weighted_scores(z, w)
    log_probs = scores - logsumexp(scores, axis=1, keepdims=True)
    probs = np.exp(log_probs)

    counts = np.bincount(y, minlength=z.shape[1])
    sample_weights = 1.0 / counts[y]
    sample_weights /= sample_weights.sum()

    rows = np.arange(len(y))
    loss = -np.dot(sample_weights, log_probs[rows, y])

    probs[rows, y] -= 1.0
    gradient = np.einsum(
        'nkm,nk,n->m', z, probs, sample_weights
    )
    return loss, gradient


def fit_weights(z, y):
    n = z.shape[2]
    result = minimize(
        balanced_ce_and_gradient,
        np.full(n, 1.0 / n),
        args=(z, y),
        jac=True,
        method='SLSQP',
        bounds=[(0.0, 1.0)] * n,
        constraints=[{
            'type': 'eq',
            'fun': lambda w: w.sum() - 1.0,
            'jac': lambda w: np.ones_like(w)
        }],
        options={'ftol': 1e-11, 'maxiter': 500}
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def print_metrics(title, scores, y):
    predicted = scores.argmax(axis=1)
    accuracy = (predicted == y).mean()
    balanced_accuracy = np.mean([
        (predicted[y == k] == k).mean()
        for k in np.unique(y)
    ])

    log_probs = scores - logsumexp(
        scores, axis=1, keepdims=True
    )
    balanced_ce = np.mean([
        -log_probs[y == k, k].mean()
        for k in np.unique(y)
    ])

    print(
        f'{title}: Accuracy={accuracy:.2%}, '
        f'Balanced={balanced_accuracy:.2%}, '
        f'Balanced CE={balanced_ce:.4f}'
    )


# Cross-validation: fit on 4 folds, predict the held-out fold
splitter = StratifiedKFold(
    n_splits=N_FOLDS, shuffle=True, random_state=42
)

oof_equal = np.empty(weighted_logits.shape[:2])
oof_optimized = np.empty_like(oof_equal)
cv_weights = []

for fold, (train_idx, val_idx) in enumerate(
    splitter.split(weighted_logits, labels), 1
):
    w = fit_weights(
        weighted_logits[train_idx], labels[train_idx]
    )
    cv_weights.append(w)

    oof_equal[val_idx] = weighted_scores(
        weighted_logits[val_idx], equal_weights
    )
    oof_optimized[val_idx] = weighted_scores(
        weighted_logits[val_idx], w
    )

    print(f'Fold {fold} weights: {np.round(w, 4)}')


print('\n--- Cross-validation (held-out predictions) ---')
print_metrics('Equal weights    ', oof_equal, labels)
print_metrics('Optimized weights', oof_optimized, labels)

print('Mean weights:', np.round(np.mean(cv_weights, axis=0), 4))
print('Std weights: ', np.round(np.std(cv_weights, axis=0), 4))


# Final weights fitted on all 441 validation poems
final_weights = fit_weights(weighted_logits, labels)

print('\n--- Final fit on all validation poems ---')
for i, w in zip(WEIGHTED_NETS, final_weights):
    print(f'{i}: {FOLDERS[i - 1]}  weight={w:.4f}')

print_metrics(
    'Equal weights    ',
    weighted_scores(weighted_logits, equal_weights), labels
)
print_metrics(
    'Optimized weights',
    weighted_scores(weighted_logits, final_weights), labels
)
###########################

comb_ids = np.atleast_1d(COMB if COMB is not None else
                         np.array([sum(1 << i for i in combo) for k in range(1, MAX_NETS + 1)
                                  for combo in combinations(range(num_nets), k)], dtype=np.int64))
all_comb = ((comb_ids[:, None] >> np.arange(num_nets)) & 1)
all_comb = np.tile(all_comb,[1,num_seg])
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
    probabilities[probabilities == probabilities.max(axis=1, keepdims=True)] += 2*num_nets*num_seg
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

post_ids = metrics[0]['post_ids']
post_len = metrics[-1]['masked_length']

raw_probs = np.stack([m['probabilities'] for m in metrics], axis=2)
individual_predicted = np.argmax(raw_probs, axis=1)
selected = all_comb.T.astype(bool)
votes_for_label = np.sum((individual_predicted[:, :, None] == labels[:, None, None]) & selected[None, :, :],axis=1)
votes_for_predicted = np.sum((individual_predicted[:, :, None] == predicted[:, None, :]) & selected[None, :, :],axis=1)
individual_correct = individual_predicted == labels[:, None, None]
individual_chosen = individual_predicted[:,:,None] == predicted[:,None,:]


print(f"Selection: {SELECTION}")
print(f"{'Comb':>4}  {'Nets':<18} {'Count':>5} {'Accuracy':>10} {'Balanced':>10}")
print("-" * 55)

for idx in np.argsort(-balanced_accuracy):
    nets = ",".join(str(i + 1) for i in np.flatnonzero(all_comb[idx]))
    print(f"{comb_ids[idx]:4d}  {nets:<18} {all_comb[idx].sum():5d} "
          f"{accuracy[idx]:10.2%} {balanced_accuracy[idx]:10.2%}")


# Error analysis
num_selected = all_comb.sum(axis=1)
mean_probs = raw_probs @ all_comb.T / num_selected

sorted_scores = np.sort(ensemble_scores, axis=1)
score_margin = sorted_scores[:, -1, :] - sorted_scores[:, -2, :]

sorted_probs = np.sort(mean_probs, axis=1)
prob_margin = sorted_probs[:, -1, :] - sorted_probs[:, -2, :]

true_rank = 1 + np.sum(
    ensemble_scores > np.take_along_axis(
        ensemble_scores, labels[:, None, None], axis=1
    ),
    axis=1
)

for c in np.argsort(-balanced_accuracy)[:10]:
    combination = np.flatnonzero(all_comb[c]) + 1

    df = pd.DataFrame({
        'post_id': post_ids,
        'label': labels,
        'predicted': predicted[:, c],
        'correct': correct[:, c],
        'length': post_len,
        'votes_predicted': votes_for_predicted[:, c],
        'votes_label': votes_for_label[:, c],
        'agreement': votes_for_predicted[:, c] / num_selected[c],
        'score_margin': score_margin[:, c],
        'prob_margin': prob_margin[:, c],
        'true_rank': true_rank[:, c],
        'score_predicted': ensemble_scores[
            np.arange(len(labels)), predicted[:, c], c
        ],
        'score_label': ensemble_scores[
            np.arange(len(labels)), labels, c
        ]
    })

    unanimous_errors = df[
        (df['votes_predicted'] == num_selected[c]) &
        (~df['correct'])
        ]
    print("\n--- Unanimous errors ---")
    print(unanimous_errors[
              ['post_id', 'label', 'predicted', 'length', 'prob_margin']
          ].to_string(index=False))
    # Two guesses for 20% of posts longer than 100 characters
    long_posts = df[df['length'] > 100].copy()

    n_two = int(np.ceil(len(long_posts) * 0.20))
    two_guess_ids = long_posts.nsmallest(n_two, 'prob_margin').index

    # Two leading authors according to ensemble scores
    top2 = np.argsort(ensemble_scores[:, :, c], axis=1)[:, -2:]
    top2_correct = np.any(top2 == labels[:, None], axis=1)

    # One guess by default, two guesses for low-confidence posts
    hybrid_correct = correct[:, c].copy()
    hybrid_correct[two_guess_ids] = top2_correct[two_guess_ids]

    # Evaluate only long posts
    long_correct = hybrid_correct[long_posts.index]

    balanced = pd.DataFrame({
        'label': long_posts['label'].to_numpy(),
        'correct': long_correct
    }).groupby('label')['correct'].mean().mean()

    print("\n--- Hybrid Top-1 / Top-2: posts > 100 ---")
    print(f"Posts: {len(long_posts)}")
    print(f"Two guesses: {n_two} ({n_two / len(long_posts):.1%})")
    print(f"Accuracy: {long_correct.mean():.2%}")
    print(f"Balanced Accuracy: {balanced:.2%}")
    print(f"Errors: {(~long_correct).sum()}")

    long_posts = df[df['length'] > 100].copy()
    ranked = long_posts.sort_values('prob_margin', ascending=False)

    print("\n--- Risk-Coverage: posts longer than 100 characters ---")
    print(f"Total posts: {len(ranked)}")
    print(f"{'Coverage':>10} {'Answered':>10} {'Accuracy':>10} "
          f"{'Balanced':>10} {'Risk':>10}")

    for coverage in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        n = int(np.ceil(len(ranked) * coverage))
        answered = ranked.iloc[:n]

        accuracy_filtered = answered['correct'].mean()
        balanced_filtered = answered.groupby('label')['correct'].mean().mean()
        risk = 1 - accuracy_filtered

        print(f"{coverage:10.0%} {n:10d} {accuracy_filtered:10.2%} "
              f"{balanced_filtered:10.2%} {risk:10.2%}")

    long_posts = df[df['length'] > 100]

    balanced = long_posts.groupby('label')['correct'].mean().mean()

    print(f"\n--- Posts longer than 100 characters ---")
    print(f"Posts: {len(long_posts)}")
    print(f"Accuracy: {long_posts['correct'].mean():.2%}")
    print(f"Balanced Accuracy: {balanced:.2%}")


    print(f"\n=== Combination {combination.tolist()} ===")
    print(f"Posts: {len(df)}")
    print(f"Accuracy: {df['correct'].mean():.2%}")
    print(f"Balanced Accuracy: {df.groupby('label')['correct'].mean().mean():.2%}")

    # Errors by author
    print("\n--- By author ---")
    print(df.groupby('label').agg(
        posts=('correct', 'size'),
        accuracy=('correct', 'mean'),
        avg_length=('length', 'mean')
    ).round(3).to_string())

    # Errors by post length
    df['length_group'] = pd.cut(
        df['length'],
        bins=[0, 100, 200, 384, 768, np.inf],
        labels=['<=100', '101-200', '201-384', '385-768', '>768'],
        include_lowest=True
    )

    print("\n--- By post length ---")
    print(df.groupby('length_group', observed=True).agg(
        posts=('correct', 'size'),
        accuracy=('correct', 'mean')
    ).round(3).to_string())

    # Errors by agreement
    print("\n--- By number of agreeing predictions ---")
    print(df.groupby('votes_predicted').agg(
        posts=('correct', 'size'),
        accuracy=('correct', 'mean'),
        avg_margin=('prob_margin', 'mean')
    ).round(3).to_string())

    # Correct label ranking
    print("\n--- Rank of correct author ---")
    print(df['true_rank'].value_counts().sort_index().to_string())

    # Selective prediction
    print("\n--- Risk-Coverage ---")
    for confidence_name in ['agreement', 'prob_margin', 'score_margin']:
        print(f"\nConfidence metric: {confidence_name}")
        ranked = df.sort_values(confidence_name, ascending=False)

        print(f"{'Coverage':>10} {'Answered':>10} {'Accuracy':>10} {'Risk':>10}")

        for coverage in [0.5, 0.7, 0.8, 0.9, 1.0]:
            n = int(np.ceil(len(ranked) * coverage))
            answered = ranked.iloc[:n]
            acc = answered['correct'].mean()

            print(f"{coverage:10.0%} {n:10d} {acc:10.2%} {1-acc:10.2%}")
