import numpy as np
import os
import pandas as pd
import json
SELECTION = 'MAJORITY'   # SUM/ MUL / MAX / MAJORITY
SEGMENTS = ['_start','_end']
COMB = [29, 8]   # None for all or a list
DATA_SET = 'test'
FOLDERS = ('E32_C9_256_S111', 'E32_C9_256_S222', 'E32_C9_256_S333', 'E32_C9_256_S444', 'E32_C9_256_S555')

num_seg = len(SEGMENTS)
num_nets = len(FOLDERS)
metrics = []

folder_names = FOLDERS * num_seg
file_names = [DATA_SET + suffix + '.npz' for suffix in SEGMENTS for _ in range(num_nets)]
for net_idx in range(num_nets * num_seg):
    metrics.append(np.load(os.path.join("experiments",folder_names[net_idx],file_names[net_idx])))
assert all(
    np.array_equal(metrics[0]['post_ids'], m['post_ids'])
    for m in metrics[1:]
), "Post IDs or order differ between files"

num_authors = np.size(metrics[0]['probabilities'],1)
labels = metrics[0]['labels']

comb_ids = np.atleast_1d(COMB if COMB is not None else np.arange(1, 2 ** num_nets))
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

for c in range(all_comb.shape[0]):
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
