
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

MIN_TEXT_LENGTH = 50
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
    page_title="Who Wrote It?",
    page_icon="📖",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        direction: rtl;
        text-align: right;
    }

    textarea {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

st.title("📖 מי כתב את השיר?")
st.subheader("Who Wrote It?")

st.write(
    "הדביקו שיר בעברית, "
    "ורשת נוירונים תנסה לזהות מי כתב אותו "
    "מבין שמונה משוררים."
)

poem = st.text_area(
    "מילות השיר",
    height=300,
    placeholder="הדביקו כאן את השיר, כולל מעברי השורות..."
)

if st.button("זהה את המשורר", type="primary"):

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

        st.divider()

        st.caption("המשורר הסביר ביותר לפי המודל")
        st.header(authors[best])
        st.metric(
            "ציון המודל",
            f"{probs[best]:.1%}"
        )

        st.subheader("כל המועמדים")

        for idx in order:
            st.progress(
                float(probs[idx]),
                text=f"{authors[idx]} — {probs[idx]:.1%}"
            )

        st.caption(f"זמן חיזוי: {elapsed:.3f} שניות")

        st.info(
            "המודל בוחר מבין שמונה משוררים בלבד. "
            "ציוני המודל אינם הוכחה לזהות המחבר."
        )

st.divider()

st.caption(
    "Who Wrote It? | Python · PyTorch · Deep Learning"
)
