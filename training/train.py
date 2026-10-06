"""Train a real conjunctivitis classifier once you have clinically labelled data.

Expected structure:
 data/train/normal/*.jpg
 data/train/conjunctivitis/*.jpg
 data/val/normal/*.jpg
 data/val/conjunctivitis/*.jpg
 data/test/normal/*.jpg
 data/test/conjunctivitis/*.jpg

IMPORTANT: split by patient, not by image, when multiple images belong to one person.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def make_loader(path: Path, image_size: int, batch: int, train: bool) -> DataLoader:
    tfms = [transforms.Resize((image_size, image_size))]
    if train:
        tfms += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(7)]
    tfms += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return DataLoader(datasets.ImageFolder(path, transform=transforms.Compose(tfms)), batch_size=batch, shuffle=train)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="backend/conjunctivitis_model.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    train = make_loader(root / "train", 224, args.batch, True)
    val = make_loader(root / "val", 224, args.batch, False)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()

        model.eval()
        ys, ps, scores = [], [], []
        with torch.no_grad():
            for x, y in val:
                logits = model(x.to(device))
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pred = (prob >= 0.5).astype(int)
                ys.extend(y.numpy().tolist()); ps.extend(pred.tolist()); scores.extend(prob.tolist())
        print({
            "epoch": epoch + 1,
            "accuracy": accuracy_score(ys, ps),
            "f1": f1_score(ys, ps, zero_division=0),
            "auroc": roc_auc_score(ys, scores) if len(set(ys)) > 1 else None,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": train.dataset.classes}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
