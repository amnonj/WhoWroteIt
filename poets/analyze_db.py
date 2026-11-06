import numpy as np
import pandas as pd
from collections import Counter
import data
import config
import matplotlib.pyplot as plt

posts_df = data.load_posts()
filtered_posts_df = posts_df[
    posts_df["PageName"].isin(config.AUTHOR_LIST) & posts_df["valid"]
    ].drop_duplicates(subset=["PageName", "Data"]).copy()
filtered_posts_df["datetime"] = pd.to_datetime(filtered_posts_df["Date"], format="%b %d, %Y @ %H:%M:%S.%f")
filtered_posts_df["month"] = filtered_posts_df["datetime"].dt.to_period("M")
filtered_posts_df["split"] = np.select(
    [filtered_posts_df["datetime"] >= config.TEST_START,
     filtered_posts_df["datetime"] >= config.VALIDATION_START],
    ["test", "val"], default="train")
posts_by_author = filtered_posts_df.groupby('PageName')
print(len(posts_by_author))
print(posts_by_author["Data"].agg(["count", "nunique"]))
print(posts_by_author["text_length"].agg(["mean", "median"]))

posts_stats = posts_by_author.agg(
    posts_num=("Data", "count"),
    mean_length=("text_length", "mean"),
    median_length=("text_length", "median"),
    first_post=("datetime", "min"),
    last_post=("datetime", "max")
)
print(posts_stats)

monthly_posts = filtered_posts_df.pivot_table(
    index="month",
    columns="PageName",
    values="Data",
    aggfunc="count",
    fill_value=0
)

print(monthly_posts.to_string())

print(pd.crosstab(filtered_posts_df["PageName"], filtered_posts_df["split"]))

duplicates = filtered_posts_df.loc[filtered_posts_df["Data"].duplicated(keep=False),["PageName", "split", "Data"]]
print(duplicates)
filtered_posts_df = filtered_posts_df[~filtered_posts_df["Data"].duplicated(keep=False)]
print(pd.crosstab(filtered_posts_df["PageName"], filtered_posts_df["split"]))

# Find common words
train_by_author = filtered_posts_df[
    filtered_posts_df["split"] == "train"
].groupby("PageName")
texts_by_author = train_by_author["Data"].agg(lambda x: x.str.cat(sep=" "))
clean_texts = texts_by_author.str.replace(r'[^א-תa-zA-Z]+', ' ', regex=True).str.strip().str.lower()
words_by_author = clean_texts.str.split()
common_words = words_by_author.apply(lambda words: Counter(words).most_common(30))
# Export common words to CSV
rows = [
    (author, word, count)
    for author, word_counts in common_words.items()
    for word, count in word_counts
]

words_df = pd.DataFrame(rows, columns=["Author", "Word", "Count"])
words_df.to_csv("common_words.csv", index=False, encoding="utf-8-sig")

filtered_posts_df = data.load_filtered_posts()
train_set_df = filtered_posts_df[filtered_posts_df["split"] == "train"]
print(train_set_df["masked_length"].quantile([0.5,0.75,0.9,0.95]))
plt.hist(train_set_df["masked_length"], bins=50)
plt.show()
