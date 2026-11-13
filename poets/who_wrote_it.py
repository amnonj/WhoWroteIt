import random
import data
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
from common import text_utils
from poets import AuthorNet as an
from poets import datasets
from types import SimpleNamespace
import time
import tkinter as tk
from tkinter import scrolledtext
import re

IN_FOLDER = '20261111-141613'
MIN_TEXT_LENGTH = 50
PAD = 0
NIQQUD_PATTERN = re.compile(r"[\u0591-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C5\u05C7]")
PAD = 0

# post_text = input('Enter post text: ')
root = tk.Tk()
root.title("הכנס שיר")

text_box = scrolledtext.ScrolledText(root, width=80, height=15, wrap=tk.WORD)


def paste(event=None):
    text_box.insert(tk.INSERT, root.clipboard_get())
    return "break"

text_box.bind("<Control-v>", paste)
text_box.bind("<Control-V>", paste)
text_box.bind("<Shift-Insert>", paste)

text_box.pack(padx=10, pady=10)


def submit():
    global post_text
    post_text = text_box.get("1.0", "end-1c")
    root.destroy()

tk.Button(root, text="אישור", command=submit).pack(pady=10)

post_text = ""
root.mainloop()

trimmed_text = re.sub(r"\d+", "", post_text)
trimmed_text = NIQQUD_PATTERN.sub("", trimmed_text)

assert (len(trimmed_text) >= MIN_TEXT_LENGTH), 'הפוסט קצר מדי'
start_time = time.perf_counter()

directory = "experiments/" + IN_FOLDER
model_state = torch.load(os.path.join(directory, "model.pth"))
with open(os.path.join(directory, 'netconfig.json'), 'r') as f:
    nnconf_dic = json.load(f)
    nnconf = SimpleNamespace(**nnconf_dic)

# prepare coding dictionaries
char_to_id = nnconf.char_to_id
author_to_id = nnconf.author_to_id
num_authors = len(author_to_id)
authors = [name for name, idx in sorted(author_to_id.items(), key=lambda pair: pair[1])]

model = an.AuthorNet(SimpleNamespace(**nnconf_dic))
model.load_state_dict(model_state)

coded_text = text_utils.conv_char_to_code_and_truncate(trimmed_text, char_to_id, nnconf.message_len)
lines = [line.strip() for line in trimmed_text.splitlines() if line.strip()]
lines = lines[:nnconf.NUM_LINES]
coded_lines = text_utils.conv_char_to_code(lines, char_to_id)

def encode_line_2d(line, chars_from_start, chars_from_end):
    first = line[:chars_from_start]
    last = line[-chars_from_end:] if chars_from_end else []
    return first + [PAD] * (chars_from_start - len(first)) + [PAD] * (chars_from_end - len(last)) + last

lines_segment = coded_lines + [[PAD]] * (nnconf.NUM_LINES - len(coded_lines))
segment_2d = []
for l in lines_segment:
    segment_2d.append(encode_line_2d(l, nnconf.CHARS_FROM_START, nnconf.CHARS_FROM_END))

model_input1 = torch.tensor(coded_text, dtype=torch.long).unsqueeze(0)
model_input2 = torch.tensor(segment_2d, dtype=torch.long).unsqueeze(0)
model.eval()
with torch.no_grad():
        outputs = model(model_input1, model_input2)
probs = torch.softmax(outputs, dim=1).numpy()

predicted = np.argmax(probs)
predicted_text = next(k for k, v in author_to_id.items() if v == predicted)
elapsed_time = time.perf_counter() - start_time
print(f'I predict the author is: {predicted_text}')
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(probs)
