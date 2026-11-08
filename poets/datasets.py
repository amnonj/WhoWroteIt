from torch.utils.data import Dataset
import torch
import numpy as np
import random

PAD = 0


class PostDataset(Dataset):

    def __init__(self, text, authors, message_len, max_post_regions, epoch_size):
        self.text = text
        self.authors = authors
        self.message_len = message_len
        self.epoch_size = epoch_size

        self.regions = []

        for i in range(len(text)):
            n_regions = min(max_post_regions,
                            max(1, int(np.ceil(len(text[i]) / message_len))))

            max_start = max(0, len(text[i]) - message_len)

            start_points = np.linspace(
                0, max_start + 1, n_regions + 1, dtype=int
            )

            self.regions.extend(
                (i, int(authors[i]), start, end)
                for start, end in zip(start_points[:-1], start_points[1:])
                if start < end
            )
        self.set_epoch()

    def __len__(self):
        n_authors = len(set(self.authors.tolist()))
        return self.epoch_size // n_authors * n_authors

    def set_epoch(self):
        author_set = set(self.authors.tolist())
        items_per_author = self.epoch_size // len(author_set)
        self.epoch_regions = []
        for author in author_set:
            author_regions = [r for r in self.regions if r[1] == author]
            n_author_regions = len(author_regions)
            random.shuffle(author_regions)
            self.epoch_regions.extend((items_per_author // n_author_regions) * author_regions +
                            author_regions[:(items_per_author % n_author_regions)])
        random.shuffle(self.epoch_regions)


    def __getitem__(self, idx):
        i, label, start_min, start_max = self.epoch_regions[idx]
        start = np.random.randint(start_min, start_max)
        segment = self.text[i][start:start + self.message_len]

        text_len = len(segment)

        if text_len <= self.message_len:
            segment = segment + [PAD] * (self.message_len - text_len)

        return torch.tensor(segment, dtype=torch.long), label
