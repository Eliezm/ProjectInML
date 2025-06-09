# train_classifier.py

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer, BertModel,
    get_linear_schedule_with_warmup
)
from data_utils import load_all_transcripts  # your parser

# ─── 1️⃣ Prep data + labels ───────────────────────────────────────────────
print("► Parsing transcripts…")
root = Path("train/transcription")
records = load_all_transcripts(root)

# build a DataFrame of text + label (0=Control in cc/, 1=AD in cd/)
rows = []
for rec in records:
    path = rec["path"]
    text = rec["combined_text"]
    # determine class by parent folder name
    lbl = 0 if "/cc/" in path.replace("\\","/") else 1
    rows.append({"text": text, "label": lbl})
df = pd.DataFrame(rows)
print(f"  • Found {len(df)} transcripts ({df.label.value_counts().to_dict()})")

# stratified train/val
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df.label, random_state=42
)
print(f"  • Split into {len(train_df)} train / {len(val_df)} val")

# ─── 2️⃣ Dataset & DataLoader ─────────────────────────────────────────────
class TranscriptDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer: BertTokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels":         torch.tensor(self.labels[i], dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_ds = TranscriptDataset(
    train_df.text.tolist(), train_df.label.tolist(),
    tokenizer, max_length=256
)
val_ds = TranscriptDataset(
    val_df.text.tolist(), val_df.label.tolist(),
    tokenizer, max_length=256
)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

# ─── 3️⃣ Model definition ────────────────────────────────────────────────
class ADClassifier(torch.nn.Module):
    def __init__(self, n_classes=2, dropout_p=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = torch.nn.Dropout(dropout_p)
        self.out  = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = out.pooler_output      # [batch, hidden]
        return self.out(self.drop(pooled))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ADClassifier().to(device)
print("► Model & tokenizer ready on", device)

# ─── 4️⃣ Training setup ─────────────────────────────────────────────────
EPOCHS = 10
total_steps = len(train_loader) * EPOCHS
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# metrics
from sklearn.metrics import accuracy_score, f1_score

def train_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        logits = model(ids, mask)
        loss   = torch.nn.functional.cross_entropy(logits, lbls)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls= batch["labels"].to(device)
        logits = model(ids, mask)
        batch_pred = logits.argmax(-1).cpu().tolist()
        preds.extend(batch_pred)
        trues.extend(lbls.cpu().tolist())
    return accuracy_score(trues, preds), f1_score(trues, preds)

# ─── 5️⃣ Run training loop ────────────────────────────────────────────────
best_f1 = 0.0
for epoch in range(1, EPOCHS+1):
    loss = train_epoch()
    train_acc, train_f1 = eval_epoch(train_loader)
    val_acc,   val_f1   = eval_epoch(val_loader)
    print(f"Epoch {epoch:2d}  loss={loss:.4f}  train_f1={train_f1:.4f}  val_f1={val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pt")
        print("  → Saved new best_model.pt")

print("► Training complete. Best val F1 =", best_f1)

# ─── 6️⃣ Sliding‐window inference (optional) ─────────────────────────────
def classify_long_text(text, model, device,
                       win_size=512, stride=256):
    ids = tokenizer.encode(text, add_special_tokens=False)
    core = win_size - 2
    windows = []
    for start in range(0, len(ids), stride):
        chunk = ids[start:start+core]
        chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        pad_len = win_size - len(chunk)
        chunk = chunk + [tokenizer.pad_token_id]*pad_len
        mask  = [1]*len(chunk[:len(chunk)-pad_len]) + [0]*pad_len
        windows.append((chunk, mask))
        if start+core >= len(ids):
            break
    input_ids = torch.tensor([w[0] for w in windows]).to(device)
    attn_mask = torch.tensor([w[1] for w in windows]).to(device)
    with torch.no_grad():
        logits = model(input_ids, attn_mask)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    avgp  = probs.mean(axis=0)
    return int(avgp.argmax()), avgp

print("► Done!  You can now import `classify_long_text` for long inputs.")
