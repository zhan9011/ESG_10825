from __future__ import annotations

import numpy as np


def encode(tokenizer, texts: list[str], max_length: int) -> dict[str, np.ndarray]:
    result = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    return {key: value.astype(np.int64) for key, value in result.items()}


def subset(encoded: dict[str, np.ndarray], rows: np.ndarray):
    return {key: value[rows] for key, value in encoded.items()}


def dataset(encoded: dict[str, np.ndarray], labels: np.ndarray | None = None):
    import torch
    from torch.utils.data import Dataset

    class EncodedDataset(Dataset):
        def __len__(self):
            return len(encoded["input_ids"])

        def __getitem__(self, index):
            item = {
                key: torch.tensor(value[index], dtype=torch.long)
                for key, value in encoded.items()
            }
            if labels is not None:
                item["labels"] = torch.tensor(labels[index], dtype=torch.long)
            return item

    return EncodedDataset()


def predict(model, encoded, batch_size: int, device) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset(encoded), batch_size=batch_size, shuffle=False)
    output = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**batch).logits
            output.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(output).numpy()


def expand_binary(probability: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [probability[:, 0], probability[:, 1], np.full(len(probability), 1e-7)]
    )
