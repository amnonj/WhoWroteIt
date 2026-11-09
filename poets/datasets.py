from torch.utils.data import Dataset
import torch
import numpy as np
import random

PAD = 0


class PostDataset(Dataset):

    def __init__(self, text, lines, authors, nnconf, VALIDATION = False):
        self.text = text
        self.lines = lines
        self.authors = authors
        self.nnconf = nnconf
        self.VALIDATION = VALIDATION

        self.regions = []

        for i in range(len(text)):
            n_regions = (1 if VALIDATION else min(nnconf.MAX_POST_REGIONS,
                                                  max(1, int(np.ceil(len(text[i]) / nnconf.message_len)))))

            max_start = max(0, len(text[i]) - nnconf.message_len)

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
        if self.VALIDATION:
            return len(self.regions)
        n_authors = len(set(self.authors.tolist()))
        return self.nnconf.EPOCH_SIZE // n_authors * n_authors

    def set_epoch(self):
        if self.VALIDATION:
            self.epoch_regions = self.regions.copy()
            return
        author_set = set(self.authors.tolist())
        items_per_author = self.nnconf.EPOCH_SIZE // len(author_set)
        self.epoch_regions = []
        for author in author_set:
            author_regions = [r for r in self.regions if r[1] == author]
            n_author_regions = len(author_regions)
            random.shuffle(author_regions)
            self.epoch_regions.extend((items_per_author // n_author_regions) * author_regions +
                            author_regions[:(items_per_author % n_author_regions)])
        random.shuffle(self.epoch_regions)

    def __getitem__(self, idx):
        def choose_segment(low, high, text, length, deterministic):
            if high <= low:
                start = low
            elif deterministic:
                h = hash(tuple(x for item in text for x in (item if isinstance(item, list) else [item])))
                start =  low + h % (high - low)
            else:
                start =  np.random.randint(low, high)
            return text[start:start + length]
        # 1D net
        i, label, start_min, start_max = self.epoch_regions[idx]
        segment = choose_segment(start_min, start_max, self.text[i], self.nnconf.message_len, self.VALIDATION)
        text_len = len(segment)
        if text_len <= self.nnconf.message_len:
            segment = segment + [PAD] * (self.nnconf.message_len - text_len)
        # 2D net

        def encode_line_2d(line, chars_from_start, chars_from_end):
            first = line[:chars_from_start]
            last = line[-chars_from_end:] if chars_from_end else []
            return first + [PAD] * (chars_from_start - len(first)) + [PAD] * (chars_from_end - len(last)) + last

        start_max = len(self.lines[i]) - self.nnconf.NUM_LINES
        if start_max>=0:
            lines_segment = choose_segment(0, start_max + 1, self.lines[i], self.nnconf.NUM_LINES, self.VALIDATION)
        else:
            lines_segment = self.lines[i] + [[PAD]] * -start_max
        segment_2d = []
        for l in lines_segment:
            segment_2d.append(encode_line_2d(l, self.nnconf.CHARS_FROM_START, self.nnconf.CHARS_FROM_END))

        return torch.tensor(segment, dtype=torch.long), torch.tensor(segment_2d, dtype=torch.long), label
