import numpy as np
import pandas as pd
from collections import Counter
import re
from poets import data
from poets import config

filtered_posts_df = data.load_filtered_posts()


print(filtered_posts_df[["PageName", "Title", "text_length", "masked_length"]].head())

print(filtered_posts_df.groupby("PageName")["masked_length"].agg(
    ["count", "mean", "median", "min", "max"]
))


posts_by_author = filtered_posts_df.groupby('PageName')
print(len(posts_by_author))

posts_stats = posts_by_author.agg(
    works=("masked_text", "size"),
    mean_length=("masked_length", "mean"),
    median_length=("masked_length", "median"),
    max_length=("masked_length", "max")
)
print(posts_stats)

# Number of sampling regions per work
INPUT_LENGTH = 768
MAX_REGIONS_PER_WORK = 2

filtered_posts_df["num_regions"] = np.minimum(
    MAX_REGIONS_PER_WORK,
    np.maximum(
        1,
        np.ceil(filtered_posts_df["masked_length"] / INPUT_LENGTH)
    )
).astype(int)

for split in ["train", "val", "test"]:
    split_df = filtered_posts_df[filtered_posts_df["split"] == split]

    region_stats = split_df.groupby("PageName").agg(
        works=("Post_ID", "size"),
        regions=("num_regions", "sum"),
        mean_regions=("num_regions", "mean"),
        max_regions=("num_regions", "max")
    )

    print(f"\n--- Sampling regions: {split.upper()} ---")
    print(region_stats.round(2).to_string())

    print(f"\nTotal regions: {region_stats['regions'].sum():,}")
    print(f"Minimum regions per poet: {region_stats['regions'].min():,}")
    print(f"Maximum regions per poet: {region_stats['regions'].max():,}")

print(pd.crosstab(filtered_posts_df["PageName"], filtered_posts_df["split"]))