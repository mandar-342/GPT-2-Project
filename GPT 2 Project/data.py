import torch
from pathlib import Path


class TextDataset:
    def __init__(self, path, block_size):
        text = Path(path).read_text(encoding="utf-8")

        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        split = int(0.9 * len(data))
        self.train_data = data[:split]
        self.val_data = data[split:]

        self.block_size = block_size

    def get_batch(self, split, batch_size):
        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(0, len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])

        return x, y