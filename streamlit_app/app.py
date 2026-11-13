
import sys
import time
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from common import text_utils
from poets import AuthorNet as an


MODEL_FOLDER = BASE_DIR.parent / "poets" / "experiments" / "20261111-141613"

MIN_TEXT_LENGTH = 100
PAD = 0

NIQQUD_PATTERN = re.compile(
    r"[\u0591-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C5\u05C7]"
)


@st.cache_resource
def load_model():
    with open(
        MODEL_FOLDER / "netconfig.json",
        "r",
        encoding="utf-8"
    ) as f:
        nnconf = SimpleNamespace(**json.load(f))

    model = an.AuthorNet(nnconf)

    model_state = torch.load(
        MODEL_FOLDER / "model.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(model_state)
    model.eval()

    return model, nnconf


def encode_line_2d(line, chars_from_start, chars_from_end):
    first = line[:chars_from_start]
    last = line[-chars_from_end:] if chars_from_end else []

    return (
        first
        + [PAD] * (chars_from_start - len(first))
        + [PAD] * (chars_from_end - len(last))
        + last
    )


def predict(text, model, nnconf):
    start_time = time.perf_counter()

    char_to_id = nnconf.char_to_id

    coded_text = text_utils.conv_char_to_code_and_truncate(
        text,
        char_to_id,
        nnconf.message_len
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    lines = lines[:nnconf.NUM_LINES]

    coded_lines = text_utils.conv_char_to_code(
        lines, char_to_id
    )

    lines_segment = (
        coded_lines
        + [[PAD]] * (nnconf.NUM_LINES - len(coded_lines))
    )

    segment_2d = [
        encode_line_2d(
            line,
            nnconf.CHARS_FROM_START,
            nnconf.CHARS_FROM_END
        )
        for line in lines_segment
    ]

    model_input1 = torch.tensor(
        coded_text, dtype=torch.long
    ).unsqueeze(0)

    model_input2 = torch.tensor(
        segment_2d, dtype=torch.long
    ).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(model_input1, model_input2)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    elapsed = time.perf_counter() - start_time

    authors = [
        name for name, idx in sorted(
            nnconf.author_to_id.items(),
            key=lambda pair: pair[1]
        )
    ]

    return authors, probs, elapsed


# -------------------------------
# Streamlit user interface
# -------------------------------

st.set_page_config(
    page_title="מי כתב את השיר?",
    page_icon="📖",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        direction: rtl;
        text-align: right;
    }

    [data-testid="stMainBlockContainer"] {
        direction: rtl;
        text-align: right;
        max-width: 820px;
        padding-top: 2.5rem;
    }

    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stCaptionContainer"] {
        direction: rtl;
        text-align: right;
    }

    textarea {
        direction: rtl !important;
        text-align: right !important;
        font-size: 18px !important;
        line-height: 1.7 !important;
    }

    [data-testid="stButton"] button {
        background-color: #315d85;
        color: white;
        border: none;
        font-size: 18px;
        border-radius: 10px;
        min-height: 48px;
    }

    [data-testid="stButton"] button:hover {
        background-color: #244c70;
        color: white;
    }

    [data-testid="stButton"] button p {
        color: white;
        text-align: center;
    }

    [data-testid="stProgress"] div[role="progressbar"] > div {
        background-color: #517ea5;
    }

    .intro-box {
        background-color: #eff3f8;
        color: #222222;
        padding: 18px 22px;
        border-radius: 12px;
        border-right: 4px solid #517ea5;
        margin-bottom: 10px;
        font-size: 16px;
        line-height: 1.8;
        text-align: right;
        direction: rtl;
    }

    .result-box {
        background-color: #e8f1f7;
        padding: 22px;
        border-radius: 14px;
        margin: 12px 0 20px 0;
        text-align: center;
    }

    .result-label {
        font-size: 19px;
        font-weight: 600;
        color: #222222;
    }

    .result-author {
        font-size: 34px;
        font-weight: 700;
        color: #244c70;
        margin: 10px 0;
    }

    .result-score {
        font-size: 26px;
        font-weight: 600;
        color: #244c70;
    }

    .author-row {
        display: flex;
        direction: rtl;
        justify-content: space-between;
        align-items: center;
        font-size: 16px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


st.title("📖 מי כתב את השיר?")

st.markdown(
    "האם רשת נוירונים יכולה לזהות משורר "
    "לפי סגנון הכתיבה שלו?"
)

st.markdown("""
<div class="intro-box">
    <b>המודל אומן לזהות שירים של שמונה משוררים:</b><br>
    אברהם שלונסקי, דן אלמגור, חיים נחמן ביאליק,
    יהודה הלוי, רחל בלובשטיין, רחל שפירא,
    שאול טשרניחובסקי ושלמה אבן גבירול.
    <br><br>
    אפשר לנסות גם שיר של משורר אחר, או אפילו
    שיר פרי עטכם, ולגלות לאיזה מסגנונות הכתיבה
    הוא דומה, לפי הערכת הרשת.
    <br><br>
    <b>יש להזין לפחות 100 תווים,
    ורצוי לפחות 16 שורות.</b>
</div>
""", unsafe_allow_html=True)


poem = st.text_area(
    "מילות השיר",
    height=320,
    label_visibility="collapsed",
    placeholder=(
        "הדביקו כאן את השיר..."
    )
)

if st.button(
    "זהה את המשורר",
    type="primary",
    use_container_width=True
):

    trimmed_text = re.sub(r"\d+", "", poem)
    trimmed_text = NIQQUD_PATTERN.sub("", trimmed_text)
    trimmed_text = trimmed_text.strip()

    if len(trimmed_text) < MIN_TEXT_LENGTH:
        st.warning(
            f"יש להזין שיר באורך של לפחות "
            f"{MIN_TEXT_LENGTH} תווים."
        )

    else:
        with st.spinner("מנתח את השיר..."):
            model, nnconf = load_model()

            authors, probs, elapsed = predict(
                trimmed_text, model, nnconf
            )

        order = np.argsort(probs)[::-1]
        best = order[0]

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">
                המשורר הסביר ביותר לפי המודל
            </div>
            <div class="result-author">
                {authors[best]}
            </div>
            <div class="result-score">
                {probs[best]:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("התאמה לסגנונות המשוררים")

        for idx in order:
            st.markdown(f"""
            <div class="author-row">
                <span>{authors[idx]}</span>
                <span dir="ltr">{probs[idx]:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(probs[idx]))

        st.caption(f"זמן חיזוי: {elapsed:.3f} שניות")


st.divider()

st.markdown(
    "לקוד המקור ולמידע טכני: "
    "[GitHub](https://github.com/amnonj/WhoWroteIt)"
)

