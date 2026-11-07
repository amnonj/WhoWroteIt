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


