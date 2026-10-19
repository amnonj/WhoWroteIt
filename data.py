import numpy as np
import pandas as pd
import zipfile as zf
from urllib.request import urlopen
import io
import config
import re


def flexible_quotes(word):
    """Match equivalent Hebrew and English quotation marks."""
    parts = []
    for char in word:
        if char in '"״“”':
            parts.append(r'["״“”]')
        elif char in "'׳‘’":
            parts.append(r"['׳‘’]")
        else:
            parts.append(re.escape(char))
    return "".join(parts)


def build_mask_pattern():
    """Compile a regex for masking names, parties and roles."""

    terms = [
        (term, category)
        for category, names in (
            ("PERSON", config.PERSON_NAMES),
            ("PARTY", config.PARTY_NAMES),
            ("ROLE", config.ROLE_NAMES),
        )
        for term in names
    ]

    # Longer terms first, so "הליכוד" wins over "ליכוד".
    terms.sort(key=lambda item: len(item[0]), reverse=True)

    # Spaces and different kinds of hyphens are interchangeable.
    separator = r"[\s\-־–—]+"

    prefixes = sorted(config.PREFIXES, key=len, reverse=True)
    prefix_pattern = "|".join(re.escape(p) for p in prefixes)

    patterns = []

    for term, category in terms:
        words = re.split(separator, term.strip())
        term_pattern = separator.join(flexible_quotes(word) for word in words)

        patterns.append(
            (re.compile(
                rf"(?<![^\W_])(?P<prefix>{prefix_pattern})?"
                rf"(?P<term>{term_pattern})(?![^\W_])",
                re.IGNORECASE
            ), category)
        )

    return patterns


MASK_PATTERNS = build_mask_pattern()

URL_PATTERN = re.compile(
    r"""(?<![\w@#])(?:
        (?:https?://|www\.)[^\s<>"']+
        |
        (?:[a-z0-9-]+\.)+[a-z]{2,}
        (?::\d+)?
        (?:[/?#][^\s<>"']*)?
    )""",
    re.IGNORECASE | re.VERBOSE
)

MAIL_PATTERN = re.compile(
    r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',
    re.IGNORECASE
)

SOCIAL_TAG_PATTERN = re.compile(r'(?<!\w)[#@][\w\u0590-\u05FF]+')


def replace_url(match):
    url = match.group()
    stripped = url.rstrip(".,;:!?)]}")
    return "URL" + url[len(stripped):]


def mask_text(text):
    """Mask identifying information while preserving other text."""

    if not isinstance(text, str):
        return text

    # Mask email addresses, URLs and social tags first.
    text = MAIL_PATTERN.sub("MAIL", text)
    text = URL_PATTERN.sub(replace_url, text)
    text = SOCIAL_TAG_PATTERN.sub("SOCIAL_TAG", text)

    # Mask names, parties and roles.
    for pattern, category in MASK_PATTERNS:
        text = pattern.sub(
            lambda match: (match.group("prefix") or "") + category,
            text
        )

    return text


# Read posts database
def load_posts():
    response = urlopen(config.POSTS_URL)
    zip_buffer = io.BytesIO(response.read())
    with zf.ZipFile(zip_buffer, 'r') as archive:
        # Open the specific file inside the archive
        with archive.open(config.CSV_FILENAME) as csv_file:
            posts_df = pd.read_csv(csv_file)

    # Prepare data
    posts_df["text_length"] = posts_df["Data"].str.len()
    posts_df["valid"] = posts_df["text_length"] >= config.MIN_POST_LENGTH
    filtered_posts_df = posts_df[
        posts_df["PageName"].isin(config.AUTHOR_LIST) & posts_df["valid"]
        ].drop_duplicates(subset=["PageName", "Data"]).copy()
    filtered_posts_df["datetime"] = pd.to_datetime(filtered_posts_df["Date"], format="%b %d, %Y @ %H:%M:%S.%f")
    filtered_posts_df["month"] = filtered_posts_df["datetime"].dt.to_period("M")
    filtered_posts_df["split"] = np.select(
        [filtered_posts_df["datetime"] >= config.TEST_START,
         filtered_posts_df["datetime"] >= config.VALIDATION_START],
        ["test", "val"], default="train")

    filtered_posts_df = filtered_posts_df[~filtered_posts_df["Data"].duplicated(keep=False)]
    filtered_posts_df["masked_text"] = filtered_posts_df["Data"].apply(mask_text)
    filtered_posts_df.reset_index(drop=True, inplace=True)
    return filtered_posts_df[["Data", "PageName", "Post_ID", "datetime", "split", "text_length", "masked_text"]]
