import numpy as np
import pandas as pd
from collections import Counter

from matplotlib import pyplot as plt

from poets import data
from poets import config

filtered_posts_df = data.load_filtered_posts()

print(filtered_posts_df[["PageName", "Title", "text_length", "masked_length"]].head())

print("\nRemaining characters:")
print(filtered_posts_df.groupby("PageName")["masked_length"].agg(
    ["count", "mean", "median", "min", "max"]
))


posts_by_author = filtered_posts_df.groupby('PageName')
print(len(posts_by_author))
print(posts_by_author["Data"].agg(["count", "nunique"]))

posts_stats = posts_by_author.agg(
    works=("Data", "size"),
    valid=("valid", "sum"),
    mean_length=("text_length", "mean"),
    median_length=("text_length", "median"),
    max_length=("text_length", "max")
)
print(posts_stats)


# 20 longest works
print("\n--- 20 longest works ---")

print(
    filtered_posts_df.nlargest(20, "text_length")[
        ["PageName", "Title", "text_length", "Genre"]
    ].to_string(index=False)
)


import re

# Hebrew vowel marks and cantillation marks
NIQQUD_PATTERN = re.compile(
    r"[\u0591-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C5\u05C7]"
)
HEBREW_PATTERN = re.compile(r"[א-ת]")

LOW_NIQQUD_RATIO = 0.10
LONG_LINE_LENGTH = 100


def analyze_poem(row):
    text = row["masked_text"]

    # Non-empty lines only
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Line lengths after removing niqqud
    lengths = [len(NIQQUD_PATTERN.sub("", line)) for line in lines]

    # Niqqud statistics for each line
    niqqud_counts = [
        len(NIQQUD_PATTERN.findall(line)) for line in lines
    ]
    hebrew_counts = [
        len(HEBREW_PATTERN.findall(line)) for line in lines
    ]

    total_niqqud = sum(niqqud_counts)
    total_hebrew = sum(hebrew_counts)

    niqqud_ratio = total_niqqud / max(1, total_hebrew)

    # First and last lines containing niqqud
    niqqud_lines = [
        i for i, count in enumerate(niqqud_counts) if count > 0
    ]

    if niqqud_lines:
        first = niqqud_lines[0]
        last = niqqud_lines[-1]

        before_lines = first
        after_lines = len(lines) - last - 1

        before_chars = sum(lengths[:first])
        after_chars = sum(lengths[last + 1:])

        first_line = first + 1
        last_line = last + 1
    else:
        before_lines = after_lines = np.nan
        before_chars = after_chars = np.nan
        first_line = last_line = np.nan

    return pd.Series({
        "Post_ID": row["Post_ID"],
        "PageName": row["PageName"],
        "Title": row["Title"],
        "text_length": len(text),
        "num_lines": len(lines),

        # Line lengths
        "mean_line_len": np.mean(lengths) if lengths else 0,
        "median_line_len": np.median(lengths) if lengths else 0,
        "p90_line_len": np.percentile(lengths, 90) if lengths else 0,
        "max_line_len": max(lengths, default=0),
        "pct_long_lines": (
            np.mean(np.array(lengths) > LONG_LINE_LENGTH)
            if lengths else 0
        ),

        # Niqqud
        "niqqud_count": total_niqqud,
        "hebrew_letters": total_hebrew,
        "niqqud_ratio": niqqud_ratio,
        "num_niqqud_lines": len(niqqud_lines),
        "pct_niqqud_lines": (
            len(niqqud_lines) / len(lines) if lines else 0
        ),

        # Position of first/last niqqud
        "first_niqqud_line": first_line,
        "last_niqqud_line": last_line,
        "before_niqqud_lines": before_lines,
        "after_niqqud_lines": after_lines,
        "before_niqqud_chars": before_chars,
        "after_niqqud_chars": after_chars
    })


# Analyze all works
filtered_posts_df = data.load_filtered_posts()
stats = filtered_posts_df.apply(analyze_poem, axis=1)

# Save per-work statistics for manual inspection
#stats.to_csv("poetry_analysis.csv", index=False, encoding="utf-8-sig")


# Summary by poet
stats["no_niqqud"] = stats["niqqud_count"] == 0
stats["low_niqqud"] = stats["niqqud_ratio"] < LOW_NIQQUD_RATIO

summary = stats.groupby("PageName").agg(
    works=("Post_ID", "size"),
    median_line=("median_line_len", "median"),
    median_p90=("p90_line_len", "median"),
    max_line=("max_line_len", "max"),
    avg_long_lines=("pct_long_lines", "mean"),
    median_niqqud=("niqqud_ratio", "median"),
    no_niqqud=("no_niqqud", "sum"),
    low_niqqud=("low_niqqud", "sum"),
    median_before=("before_niqqud_lines", "median"),
    median_after=("after_niqqud_lines", "median")
)

summary["avg_long_lines"] *= 100

print("\n--- Statistics by poet ---")
print(summary.round(2).to_string())


# Overall niqqud distribution
print("\n--- Niqqud ratio distribution ---")
print(stats["niqqud_ratio"].describe(
    percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
))

print(f"\nWorks without niqqud: {stats['no_niqqud'].sum()}")
print(f"Works with low niqqud (<{LOW_NIQQUD_RATIO:.0%}): "
      f"{stats['low_niqqud'].sum()}")


# Works with substantial text before/after niqqud
columns = [
    "PageName", "Title", "Post_ID", "niqqud_ratio",
    "before_niqqud_lines", "before_niqqud_chars",
    "after_niqqud_lines", "after_niqqud_chars"
]

print("\n--- 20 largest prefixes before first niqqud ---")
print(stats.nlargest(20, "before_niqqud_chars")[
    columns
].to_string(index=False))

print("\n--- 20 largest suffixes after last niqqud ---")
print(stats.nlargest(20, "after_niqqud_chars")[
    columns
].to_string(index=False))


# Longest-line outliers: possible prose
print("\n--- 20 works with longest lines ---")
print(stats.nlargest(20, "p90_line_len")[
    ["PageName", "Title", "Post_ID", "num_lines",
     "median_line_len", "p90_line_len",
     "max_line_len", "pct_long_lines"]
].to_string(index=False))


# --- Niqqud ratio distribution by line ---

NIQQUD_WORK_THRESHOLD = 0.05
MAX_MEAN_LINE_LENGTH = 200
EDGE_LINES = 10

# Use the previously calculated per-work statistics
valid_ids = stats.loc[
    (stats["niqqud_ratio"] >= NIQQUD_WORK_THRESHOLD) &
    (stats["mean_line_len"] <= MAX_MEAN_LINE_LENGTH),
    "Post_ID"
]

poems = filtered_posts_df[
    filtered_posts_df["Post_ID"].isin(valid_ids)
]

all_ratios = []
edge_ratios = []

for text in poems["masked_text"]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        hebrew_count = len(HEBREW_PATTERN.findall(line))
        if hebrew_count == 0:
            continue

        niqqud_count = len(NIQQUD_PATTERN.findall(line))
        ratio = niqqud_count / hebrew_count

        all_ratios.append(ratio)

        if i < EDGE_LINES or i >= len(lines) - EDGE_LINES:
            edge_ratios.append(ratio)


fig, axes = plt.subplots(1, 2, figsize=(13, 4))

bins = np.arange(0, 1.051, 0.025)

axes[0].hist(all_ratios, bins=bins)
axes[0].set_title(f"All lines (n={len(all_ratios):,})")

axes[1].hist(edge_ratios, bins=bins)
axes[1].set_title(f"First/last {EDGE_LINES} lines (n={len(edge_ratios):,})")

for ax in axes:
    ax.set_xlabel("Niqqud marks / Hebrew letters")
    ax.set_ylabel("Number of lines")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# Detail of the low-niqqud region
print("\n--- Niqqud ratio, low range ---")
hist, edges = np.histogram(
    all_ratios,
    bins=[0, 0.001, 0.05, 0.10, 0.15, 0.20,
          0.25, 0.30, 0.40, 0.50, np.inf]
)

for count, left, right in zip(hist, edges[:-1], edges[1:]):
    print(f"{left:.3f} - {right:.3f}: {count:,}")

print(f"\nWorks included: {len(poems):,}")
print(f"Total lines analyzed: {len(all_ratios):,}")
print(f"Edge lines analyzed: {len(edge_ratios):,}")
