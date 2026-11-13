import re

import numpy as np
import pandas as pd
from poets import config
from rapidfuzz.fuzz import ratio
from sklearn.model_selection import train_test_split

SPLIT_SEED = 12345
FOOTER_PATTERN = re.compile(
    r"את הטקסט\[ים\] לעיל הפיקו מתנדבי פרויקט בן[־-]יהודה באינטרנט\."
    r"\s*הכל זמין תמיד בכתובת הבאה:\s*"
    r"https?://benyehuda\.org/read/\d+\s*$"
)

NIQQUD_PATTERN = re.compile(r"[\u0591-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C5\u05C7]")
HEBREW_PATTERN = re.compile(r"[א-ת]")


def load_posts():
    posts_df = pd.read_csv(config.CSV_FILENAME)

    posts_df = posts_df[posts_df["Original_Language"] == "he"].copy()
    posts_df["text_length"] = posts_df["Data"].str.len()
    posts_df["valid"] = posts_df["text_length"] >= config.MIN_POST_LENGTH
    return posts_df


def remove_duplicate_works(posts_df):
    """Keep one representative from each group of similar works by the same poet.

    Normalization is only for matching. The saved masked_text is unchanged.
    """
    df = posts_df.reset_index(drop=True).copy()
    df["_norm"] = df["masked_text"].str.replace(
        r"[^בגדזחטכלמנסעפצקרשתךםןףץ]", "", regex=True
    )
    df["_norm_len"] = df["_norm"].str.len()

    # Connected components: A~B and B~C belong to the same duplicate group.
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    for _, group in df.groupby("PageName"):
        group = group.sort_values("_norm_len")
        rows = list(group[["_norm", "_norm_len"]].itertuples(index=True, name=None))

        for i, (idx_a, norm_a, len_a) in enumerate(rows):
            if not norm_a:
                continue
            for idx_b, norm_b, len_b in rows[i + 1:]:
                if len_b > len_a * (1 + config.MAX_LENGTH_DIFFERENCE):
                    break
                if not norm_b:
                    continue
                if ratio(norm_a, norm_b,
                         score_cutoff=config.SIMILARITY_THRESHOLD) >= config.SIMILARITY_THRESHOLD:
                    union(idx_a, idx_b)

    # Keep the longest cleaned version; ties preserve the first original record.
    representatives = {}
    for i in range(len(df)):
        root = find(i)
        current = representatives.get(root)
        if current is None or df.at[i, "masked_length"] > df.at[current, "masked_length"]:
            representatives[root] = i

    return (df.loc[sorted(representatives.values())]
              .drop(columns=["_norm", "_norm_len"])
              .reset_index(drop=True))


def load_filtered_posts():
    posts_df = load_posts()
    filtered_posts_df = posts_df[
        posts_df["PageName"].isin(config.AUTHOR_LIST) & posts_df["valid"]
    ].copy()

    def niqqud_ratio(text):
        hebrew_count = len(HEBREW_PATTERN.findall(text))
        return len(NIQQUD_PATTERN.findall(text)) / hebrew_count if hebrew_count else 0.0

    def remove_fixed_text(row):
        text = row["Data"].replace("\r\n", "\n").replace("\r", "\n").lstrip()
        header, separator, body = text.partition("\n")

        normalized_header = header.replace("“", '"').replace("”", '"')
        normalized_title = row["Title"].replace("“", '"').replace("”", '"')

        if (not separator
                or not normalized_header.startswith(normalized_title)
                or f"מאת {row['PageName']}" not in header):
            raise ValueError(f"Unexpected header: {row['Post_ID']}")

        footer_match = FOOTER_PATTERN.search(body)
        if footer_match is None:
            raise ValueError(f"Unexpected footer: {row['Post_ID']}")

        return body[:footer_match.start()].strip()

    # 1. Remove the fixed header/footer, retaining all original niqqud.
    filtered_posts_df["masked_text"] = filtered_posts_df.apply(
        remove_fixed_text, axis=1
    )

    # 2. Calculate work statistics BEFORE trimming notes at the edges.
    def get_work_stats(text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        lengths = [len(NIQQUD_PATTERN.sub("", line)) for line in lines]
        return pd.Series({
            "work_niqqud_ratio": niqqud_ratio(text),
            "mean_line_len": float(np.mean(lengths)) if lengths else 0.0,
        })

    work_stats = filtered_posts_df["masked_text"].apply(get_work_stats)
    filtered_posts_df = pd.concat([filtered_posts_df, work_stats], axis=1)

    filtered_posts_df = filtered_posts_df[
        (filtered_posts_df["work_niqqud_ratio"] >= config.MIN_NIKUD_RATIO_PER_WORK)
        & (filtered_posts_df["mean_line_len"] <= config.MAX_MEAN_LINE_LENGTH)
    ].copy()

    # 3. Remove only complete low-niqqud lines at the two edges.
    #    Never remove low-niqqud lines from the middle of a poem.
    def trim_edge_notes(row):
        text = row["masked_text"]
        lines = text.split("\n")
        first, last = 0, len(lines)

        while (first < last and
               niqqud_ratio(lines[first]) < config.MIN_NIKUD_RATIO_PER_LINE):
            first += 1

        while (last > first and
               niqqud_ratio(lines[last - 1]) < config.MIN_NIKUD_RATIO_PER_LINE):
            last -= 1

        if first == last:
            raise ValueError(
                f"No sufficiently vocalized line: Post_ID={row['Post_ID']}"
            )

        trimmed_text = "\n".join(lines[first:last]).strip()
        trimmed_text = re.sub(
            rf"\n{{{config.MAX_CONSECUTIVE_LF + 1},}}",
            "\n" * config.MAX_CONSECUTIVE_LF,
            trimmed_text,
        )
        trimmed_text = re.sub(r"\d+", "", trimmed_text)
        trimmed_text = NIQQUD_PATTERN.sub("", trimmed_text)

        lines = [line.strip() for line in trimmed_text.splitlines() if line.strip()]

        return pd.Series({
            "masked_text": trimmed_text,
            "masked_length": len(trimmed_text),
            "lines": lines
        })

    trimmed = filtered_posts_df.apply(trim_edge_notes, axis=1)
    filtered_posts_df = filtered_posts_df.drop(columns="masked_text")
    filtered_posts_df = pd.concat([filtered_posts_df, trimmed], axis=1)

    # 4. Remove exact and near-duplicate works (within each author).
    #    Compare only similar normalized lengths, and keep one work per group.
    filtered_posts_df = remove_duplicate_works(filtered_posts_df)


# Split data sets
    indices = np.arange(len(filtered_posts_df))
    labels = filtered_posts_df["PageName"].to_numpy()

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=SPLIT_SEED
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=labels[temp_idx],
        random_state=SPLIT_SEED
    )

    filtered_posts_df["split"] = ""
    filtered_posts_df.loc[filtered_posts_df.index[train_idx], "split"] = "train"
    filtered_posts_df.loc[filtered_posts_df.index[val_idx], "split"] = "val"
    filtered_posts_df.loc[filtered_posts_df.index[test_idx], "split"] = "test"

    return filtered_posts_df
