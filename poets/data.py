import numpy as np
import pandas as pd
from poets import config
import re

FOOTER_PATTERN = re.compile(
    r"את הטקסט\[ים\] לעיל הפיקו מתנדבי פרויקט בן[־-]יהודה באינטרנט\."
    r"\s*הכל זמין תמיד בכתובת הבאה:\s*"
    r"https?://benyehuda\.org/read/\d+\s*$"
)


# Read posts database
def load_posts():
    posts_df = pd.read_csv(config.CSV_FILENAME)

    # Prepare data
    posts_df = posts_df[posts_df["Original_Language"] == "he"].copy()
    posts_df["text_length"] = posts_df["Data"].str.len()
    posts_df["valid"] = posts_df["text_length"] >= config.MIN_POST_LENGTH
    return posts_df

def load_filtered_posts():
    posts_df = load_posts()

    filtered_posts_df = posts_df[
        posts_df["PageName"].isin(config.AUTHOR_LIST) & posts_df["valid"]
    ].copy()

    def clean_text(row):
        text = row["Data"].replace("\r\n", "\n").replace("\r", "\n").lstrip()

        # Remove the first line: title, author and possibly other credits
        header, separator, body = text.partition("\n")

        # Normalize quotation marks for checking the title
        normalized_header = header.replace("“", '"').replace("”", '"')
        normalized_title = row["Title"].replace("“", '"').replace("”", '"')

        if (not separator
                or not normalized_header.startswith(normalized_title)
                or f"מאת {row['PageName']}" not in header):
            raise ValueError(f"Unexpected header: {row['Post_ID']}")

        # Remove the standard Project Ben-Yehuda footer
        footer_match = FOOTER_PATTERN.search(body)

        if footer_match is None:
            raise ValueError(f"Unexpected footer: {row['Post_ID']}")

        return body[:footer_match.start()].strip()

    filtered_posts_df["masked_text"] = filtered_posts_df.apply(
        clean_text, axis=1
    )
    filtered_posts_df["masked_length"] = filtered_posts_df["masked_text"].str.len()

    filtered_posts_df.reset_index(drop=True, inplace=True)

    return filtered_posts_df
