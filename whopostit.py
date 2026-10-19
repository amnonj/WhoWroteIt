import numpy as np
import pandas as pd
import zipfile as zf
from urllib.request import urlopen
import io
import datetime
from collections import Counter

POSTS_URL = 'https://github.com/jschler/ILElectionData/raw/refs/heads/main/dataset/posts/AllPosts201811_202003.csv.zip'
CSV_FILENAME = 'AllPosts201811_202003.csv'
MIN_POST_LENGTH = 50
AUTHOR_LIST = ["Avigdor Liberman", "Benjamin Netanyahu", "Bezazel Smotrich", "Itamar Ben-Gvir", "Miri Regev",
               "Naftali Bennett", "Yair Lapid"]
VALIDATION_START = datetime.datetime(2019, 12, 1)
TEST_START = datetime.datetime(2020, 2, 1)
pd.set_option("display.max_columns", None)
data_sets_types = pd.CategoricalDtype(categories=["train", "val", "test"], ordered=True)

# Read posts database
response = urlopen(POSTS_URL)
zip_buffer = io.BytesIO(response.read())
with zf.ZipFile(zip_buffer, 'r') as archive:
    # Open the specific file inside the archive
    with archive.open(CSV_FILENAME) as csv_file:
        posts_df = pd.read_csv(csv_file)
print('Columns headers: ', list(posts_df))
print('DB Size: ', posts_df.shape)
print(posts_df.head())
posts_df.info()

# Prepare data
posts_df["text_length"] = posts_df["Data"].str.len()
posts_df["valid"] = posts_df["text_length"] >= MIN_POST_LENGTH
filtered_posts_df = posts_df[
    posts_df["PageName"].isin(AUTHOR_LIST) & posts_df["valid"]
].drop_duplicates(subset=["PageName", "Data"]).copy()
filtered_posts_df["datetime"] = pd.to_datetime(filtered_posts_df["Date"], format="%b %d, %Y @ %H:%M:%S.%f")
filtered_posts_df["month"] = filtered_posts_df["datetime"].dt.to_period("M")
filtered_posts_df["split"] = np.select(
    [filtered_posts_df["datetime"] >= TEST_START,
     filtered_posts_df["datetime"] >= VALIDATION_START],
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
