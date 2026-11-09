import os
import json
from common import text_utils
from politicians import AuthorNet as an
from types import SimpleNamespace
import torch
import data
import time
import numpy as np
import tkinter as tk
from tkinter import scrolledtext


MIN_TEXT_LENGTH = 100
FOLDERS = ('E32_C9_256_S111', 'E32_C9_256_S333', 'E32_C9_256_S444', 'E32_C9_256_S555')
PAD = 0

# post_text = input('Enter post text: ')
root = tk.Tk()
root.title("WhoPostIt - הכנס פוסט")

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

masked_text = data.mask_text(post_text)
assert (len(masked_text) >= MIN_TEXT_LENGTH), 'הפוסט קצר מדי'
if post_text != masked_text:
    print(f'צינזרתי כמה פרטים שעלולים לחשוף ישירות את הכותב. הפוסט המצונזר:\n{masked_text}')
start_time = time.perf_counter()

probs=[]
for folder in FOLDERS:
    directory = "experiments/" + folder
    model_state = torch.load(os.path.join(directory, "model.pth"))
    with open(os.path.join(directory, 'netconfig.json'), 'r') as f:
        nnconf_dic = json.load(f)
        nnconf = SimpleNamespace(**nnconf_dic)

    # prepare coding dictionaries
    char_to_id = nnconf.char_to_id
    author_to_id = nnconf.author_to_id
    num_authors = len(author_to_id)

    model = an.AuthorNet(SimpleNamespace(**nnconf_dic))
    model.load_state_dict(model_state)

    for seg in (False,True):
        coded_text = text_utils.conv_char_to_code_and_truncate(masked_text, char_to_id,
                                                               nnconf.message_len, from_end=seg)
        model_input = torch.tensor(coded_text, dtype=torch.long).unsqueeze(0)
        model.eval()
        with torch.no_grad():
                outputs = model(model_input)
        probs.append(torch.softmax(outputs, dim=1))

probs = torch.cat(probs, dim=0).numpy()
adj_probs = probs / np.size(probs, axis=0)
adj_probs[adj_probs == adj_probs.max(axis=1, keepdims=True)] += 2
ensemble_scores = adj_probs.sum(axis=0)
predicted = np.argmax(ensemble_scores)
predicted_text = next(k for k, v in author_to_id.items() if v == predicted)
elapsed_time = time.perf_counter() - start_time
print(f'I predict the author is: {predicted_text}')
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(ensemble_scores)
