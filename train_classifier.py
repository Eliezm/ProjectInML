# # train_classifier.py
#
# import os
# import random
# from pathlib import Path
# from typing import List, Dict
#
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.optim import AdamW
# from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, accuracy_score
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 1) your parser from data_utils.py (inline)
# import re
# from collections import Counter
#
# UTT_RE        = re.compile(r"^\*([A-Z]{3}):\s*(.+)$")
# DEMOG_LINE_RE = re.compile(r"^@ID:.*\|PAR\|")  # not used here
# TIMESTAMP_RE  = re.compile(r"\d+_\d+")
#
# def parse_cha(path: Path) -> str:
#     """Return combined PAR utterances as one string."""
#     utts = []
#     for line in path.read_text(encoding="utf-8").splitlines():
#         if not line.startswith("*PAR:"):
#             continue
#         text = UTT_RE.match(line).group(2)
#         text = TIMESTAMP_RE.sub("", text)
#         utts.append(text)
#     return " ".join(utts)
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 2) Build dataset by walking train/transcription/cc and /cd
# def load_data(root: Path) -> (List[str], List[int]):
#     texts, labels = [], []
#     for label_dir, lbl in [("cc", 0), ("cd", 1)]:
#         folder = root / label_dir
#         for cha in folder.glob("*.cha"):
#             txt = parse_cha(cha)
#             if txt.strip():
#                 texts.append(txt)
#                 labels.append(lbl)
#     return texts, labels
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 3) PyTorch Dataset
# class TranscriptDataset(Dataset):
#     def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length=256):
#         self.texts  = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self): return len(self.texts)
#     def __getitem__(self, i):
#         enc = self.tokenizer(
#             self.texts[i],
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_length,
#             return_tensors="pt",
#         )
#         return {
#             "input_ids":      enc["input_ids"].squeeze(0),
#             "attention_mask": enc["attention_mask"].squeeze(0),
#             "labels":         torch.tensor(self.labels[i], dtype=torch.long)
#         }
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 4) BERT-based classifier
# class ADClassifier(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = BertModel.from_pretrained("bert-base-uncased")
#         self.drop = torch.nn.Dropout(0.3)
#         self.fc   = torch.nn.Linear(self.bert.config.hidden_size, 2)
#
#     def forward(self, input_ids, attention_mask):
#         out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled = out.pooler_output
#         return self.fc(self.drop(pooled))
#
# # ───────────────────────────────────────────────────────────────────────────────
# def train_epoch(model, loader, optim, sched, device):
#     model.train()
#     total_loss = 0
#     for batch in loader:
#         optim.zero_grad()
#         ids  = batch["input_ids"].to(device)
#         mask = batch["attention_mask"].to(device)
#         lbl  = batch["labels"].to(device)
#         logits = model(ids, mask)
#         loss   = torch.nn.functional.cross_entropy(logits, lbl)
#         loss.backward()
#         optim.step(); sched.step()
#         total_loss += loss.item()
#     return total_loss / len(loader)
#
# @torch.no_grad()
# def eval_model(model, loader, device):
#     model.eval()
#     preds, trues = [], []
#     for batch in loader:
#         ids  = batch["input_ids"].to(device)
#         mask = batch["attention_mask"].to(device)
#         lbl  = batch["labels"].to(device)
#         logits = model(ids, mask)
#         preds.extend(logits.argmax(-1).cpu().tolist())
#         trues.extend(lbl.cpu().tolist())
#     return accuracy_score(trues, preds), f1_score(trues, preds)
#
# # ───────────────────────────────────────────────────────────────────────────────
# def main():
#     root = Path("train/transcription")
#     texts, labels = load_data(root)
#     # 80/20 stratified split
#     X_train, X_val, y_train, y_val = train_test_split(
#         texts, labels, test_size=0.2, random_state=42, stratify=labels
#     )
#
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     train_ds = TranscriptDataset(X_train, y_train, tokenizer)
#     val_ds   = TranscriptDataset(X_val,   y_val,   tokenizer)
#
#     train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
#     val_loader   = DataLoader(val_ds,   batch_size=4)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model  = ADClassifier().to(device)
#
#     # optimizer & scheduler
#     epochs = 10
#     total_steps = len(train_loader) * epochs
#     optim = AdamW(model.parameters(), lr=2e-5)
#     sched = get_linear_schedule_with_warmup(
#         optim,
#         num_warmup_steps=int(0.1*total_steps),
#         num_training_steps=total_steps
#     )
#
#     best_f1 = 0
#     for epoch in range(1, epochs+1):
#         loss = train_epoch(model, train_loader, optim, sched, device)
#         acc, f1 = eval_model(model, val_loader, device)
#         print(f"Epoch {epoch}: loss={loss:.4f}  val_acc={acc:.4f}  val_f1={f1:.4f}")
#         if f1 > best_f1:
#             best_f1 = f1
#             torch.save(model.state_dict(), "best_model.pt")
#             print(" → new best_model.pt saved")
#
#     print("Training finished. Best Val F1:", best_f1)
#
# if __name__ == "__main__":
#     main()

# train_classifier.py

import os
import re
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# ───────────────────────────────────────────────────────────────────────────────
# 1) parse .cha into one long patient string
UTT_RE       = re.compile(r"^\*PAR:\s*(.+)$")
TIMESTAMP_RE = re.compile(r"\d+_\d+")

def parse_cha(path: Path) -> str:
    utts = []
    for line in path.read_text(encoding='utf-8').splitlines():
        m = UTT_RE.match(line)
        if not m:
            continue
        text = TIMESTAMP_RE.sub("", m.group(1))
        utts.append(text)
    return " ".join(utts)

def load_data(root: Path) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    for label_dir, lbl in [("cc", 0), ("cd", 1)]:
        for cha in (root/label_dir).glob("*.cha"):
            txt = parse_cha(cha)
            if txt.strip():
                texts.append(txt)
                labels.append(lbl)
    return texts, labels

# ───────────────────────────────────────────────────────────────────────────────
# 2) PyTorch Dataset
class TranscriptDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length=256):
        self.texts, self.labels = texts, labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long)
        }

# ───────────────────────────────────────────────────────────────────────────────
# 3) ClinicalBERT-based classifier + small MLP head
# class ADClassifier(nn.Module):
#     def __init__(self, dropout=0.1, hidden_dim=None):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#         h = self.bert.config.hidden_size
#         h2 = hidden_dim or (h // 2)
#         self.drop = nn.Dropout(dropout)
#         self.fc1  = nn.Linear(h, h2)
#         self.bn   = nn.BatchNorm1d(h2)
#         self.act  = nn.ReLU()
#         self.fc2  = nn.Linear(h2, 2)
#
#     def forward(self, input_ids, attention_mask):
#         out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         x = out.pooler_output           # [batch, hid]
#         x = self.drop(x)
#         x = self.fc1(x)
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.drop(x)
#         return self.fc2(x)

class ADClassifier(nn.Module):
    def __init__(self, dropout=0.1, hidden_dim=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        h = self.bert.config.hidden_size
        h2 = hidden_dim or (h // 2)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(h, h2)
        self.bn   = nn.BatchNorm1d(h2)
        self.act  = nn.ReLU()
        # self.fc2  = nn.Linear(h2, 2)
        self.fc2 = nn.Sequential(
            nn.Linear(h2, h2//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2//2, 2)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.pooler_output         # [batch, hidden]
        x = self.drop(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)

# ───────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optim, sched, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optim.zero_grad()
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["labels"].to(device)
        logits = model(ids, mask)
        loss   = nn.functional.cross_entropy(logits, lbl)
        loss.backward()
        optim.step()
        sched.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["labels"].to(device)
        logits = model(ids, mask)
        preds.extend(logits.argmax(-1).cpu().tolist())
        trues.extend(lbl.cpu().tolist())
    return accuracy_score(trues, preds), f1_score(trues, preds)

# ───────────────────────────────────────────────────────────────────────────────
def main():
    # load
    texts, labels = load_data(Path("train/transcription"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # cross‐validation
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_metrics = []

    for fold,(train_idx, val_idx) in enumerate(kf.split(texts, labels), 1):
        print(f"\n>>> Fold {fold}")

        X_train = [texts[i]  for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_val   = [texts[i]  for i in val_idx]
        y_val   = [labels[i] for i in val_idx]

        train_ds = TranscriptDataset(X_train, y_train, tokenizer, max_length=256)
        val_ds   = TranscriptDataset(X_val,   y_val,   tokenizer, max_length=256)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=4)

        # model, optimizer, scheduler
        model = ADClassifier(dropout=0.1).to(device)
        epochs = 15
        total_steps = len(train_loader) * epochs
        optim = AdamW(model.parameters(), lr=1e-5)
        sched = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=int(0.2 * total_steps),
            num_training_steps=total_steps
        )

        # early stopping
        best_f1, patience = 0.0, 0
        for epoch in range(1, epochs+1):
            loss = train_one_epoch(model, train_loader, optim, sched, device)
            acc, f1 = eval_model(model, val_loader, device)
            print(f" Epoch {epoch:02d} — loss: {loss:.4f}  val_acc: {acc:.4f}  val_f1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f"best_model_fold{fold}.pt")
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    print(" → early stopping")
                    break

        fold_metrics.append(best_f1)
        print(f" → Fold {fold} best F1: {best_f1:.4f}")

    avg_f1 = sum(fold_metrics) / len(fold_metrics)
    print(f"\n=== Cross‐val average F1: {avg_f1:.4f} ===")

if __name__ == "__main__":
    main()

