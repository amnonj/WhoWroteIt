from torch.utils.data import Dataset
import torch
import random

PAD = 0


class PostDataset(Dataset):

    def __init__(self, text, authors, message_len):
        self.text = text
        self.authors = authors
        self.message_len = message_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        this_text = self.text[idx]
        text_len = len(this_text)

        if text_len <= self.message_len:
            coded = this_text + [PAD] * (self.message_len - text_len)
        else:
            start_pointer = random.randrange(text_len - self.message_len + 1)
            coded = this_text[start_pointer:start_pointer + self.message_len]

        return torch.tensor(coded, dtype=torch.long), self.authors[idx]
