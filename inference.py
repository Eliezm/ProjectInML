# # inference.py
#
# import sys
# from pathlib import Path
# import torch
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from transformers import BertTokenizer
# from data_utils import parse_cha    # your parser
# from train_classifier import ADClassifier  # your model class
#
# def load_cha_texts(base: Path):
#     texts, labels, paths = [], [], []
#     for label_dir, lbl in [("cc", 0), ("cd", 1)]:
#         folder = base / label_dir
#         if not folder.exists():
#             continue
#         for cha in sorted(folder.glob("*.cha")):
#             parsed = parse_cha(cha)
#             txt = parsed.get("combined_text", "").strip()
#             if not txt:
#                 continue
#             texts.append(txt)
#             labels.append(lbl)
#             paths.append(str(cha))
#     return texts, labels, paths
#
# def predict(texts, model, tokenizer, device, max_length=256):
#     model.eval()
#     preds = []
#     with torch.no_grad():
#         for txt in texts:
#             enc = tokenizer(
#                 txt,
#                 truncation=True,
#                 padding="max_length",
#                 max_length=max_length,
#                 return_tensors="pt"
#             ).to(device)
#             logits = model(enc.input_ids, enc.attention_mask)
#             preds.append(int(logits.argmax(-1).cpu()))
#     return preds
#
# def main():
#     base = Path("train/transcription")
#     print(f"Loading .cha from {base}/cc and {base}/cd …")
#     texts, y_true, paths = load_cha_texts(base)
#     print(f" → {len(texts)} samples found\n")
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Loading model to", device)
#     model = ADClassifier().to(device)
#     model.load_state_dict(torch.load("best_model.pt", map_location=device))
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
#     print("\nRunning inference…")
#     y_pred = predict(texts, model, tokenizer, device)
#
#     acc = accuracy_score(y_true, y_pred)
#     f1  = f1_score(y_true, y_pred)
#     print(f"\nOverall Accuracy = {acc:.4f}")
#     print(f"Overall F1       = {f1:.4f}\n")
#     print("Classification report:")
#     print(classification_report(y_true, y_pred, target_names=["Control","AD"]))
#
#     print("\nSome examples:")
#     df = pd.DataFrame({"path": paths, "true": y_true, "pred": y_pred})
#     for _, row in df.sample(5, random_state=0).iterrows():
#         print(f"{Path(row.path).name:8s}  true={row.true}  pred={row.pred}")
#
# if __name__=="__main__":
#     main()

# inference.py

# from pathlib import Path
# import torch
# import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score, classification_report
#
# # 1️⃣ Import your new model class and parser
# from train_classifier import ADClassifier, parse_cha
#
# # 2️⃣ Use AutoTokenizer for the clinical BERT vocab
# from transformers import AutoTokenizer
#
# def load_cha_texts(base: Path):
#     texts, labels, paths = [], [], []
#     for label_dir, lbl in [("cc", 0), ("cd", 1)]:
#         folder = base / label_dir
#         if not folder.exists():
#             continue
#         for cha in sorted(folder.glob("*.cha")):
#             txt = parse_cha(cha).strip()
#             if txt:
#                 texts.append(txt)
#                 labels.append(lbl)
#                 paths.append(str(cha))
#     return texts, labels, paths
#
# def predict(texts, model, tokenizer, device, max_length=256):
#     model.eval()
#     preds = []
#     with torch.no_grad():
#         for txt in texts:
#             enc = tokenizer(
#                 txt,
#                 truncation=True,
#                 padding="max_length",
#                 max_length=max_length,
#                 return_tensors="pt"
#             ).to(device)
#             logits = model(enc.input_ids, enc.attention_mask)
#             preds.append(int(logits.argmax(-1).cpu()))
#     return preds
#
# def main():
#     base = Path("train/transcription")
#     print(f"Loading .cha from {base}/cc and {base}/cd …")
#     texts, y_true, paths = load_cha_texts(base)
#     print(f" → {len(texts)} samples found\n")
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Loading model to", device)
#
#     # 3️⃣ Instantiate your new ClinicalBERT‐based classifier
#     model = ADClassifier(dropout=0.1).to(device)
#
#     # 4️⃣ Load the checkpoint you trained with this exact architecture
#     #    (e.g. best_model_fold1.pt or best_model.pt if you re‐saved)
#     model.load_state_dict(torch.load("best_model_0.84_test.pt", map_location=device))
#
#     # 5️⃣ And use the matching tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#
#     print("\nRunning inference…")
#     y_pred = predict(texts, model, tokenizer, device)
#
#     print(f"\nOverall Accuracy = {accuracy_score(y_true, y_pred):.4f}")
#     print(f"Overall F1       = {f1_score(y_true, y_pred):.4f}\n")
#     print("Classification report:")
#     print(classification_report(y_true, y_pred, target_names=["Control","AD"]))
#
#     print("\nSome examples:")
#     df = pd.DataFrame({"path": paths, "true": y_true, "pred": y_pred})
#     for _, row in df.sample(5, random_state=0).iterrows():
#         print(f"{Path(row.path).name:8s}  true={row.true}  pred={row.pred}")
#
# if __name__=="__main__":
#     main()


# inference.py

import re
from pathlib import Path

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer

from train_classifier import ADClassifier, parse_cha  # your model & parser

# ─── 1️⃣ load CC/CD “train”‐split transcripts ───────────────────────────────
def load_train_split(base: Path):
    texts, labels, paths = [], [], []
    for sub, lbl in [("cc", 0), ("cd", 1)]:
        folder = base / sub
        if not folder.exists():
            continue
        for cha in sorted(folder.glob("*.cha")):
            txt = parse_cha(cha).strip()
            if not txt:
                continue
            texts.append(txt)
            labels.append(lbl)
            paths.append(str(cha))
    return texts, labels, paths

# ─── 2️⃣ load held‐out test set from test/transcription + adress_2020_test_Labels.txt
def load_test_set(test_root: Path, label_file: Path):
    # read the label file; semicolon‐delimited
    df = pd.read_csv(label_file, sep=";", engine="python")
    df = df.rename(columns=lambda c: c.strip())
    df["ID"]    = df["ID"].str.strip()
    df["Label"] = df["Label"].astype(int)
    label_map = dict(zip(df["ID"], df["Label"]))

    texts, labels, paths = [], [], []
    for cha in sorted((test_root / "transcription").glob("*.cha")):
        sid = cha.stem  # e.g. "S160"
        if sid not in label_map:
            continue
        txt = parse_cha(cha).strip()
        if not txt:
            continue
        texts.append(txt)
        labels.append(label_map[sid])
        paths.append(str(cha))
    return texts, labels, paths

# ─── 3️⃣ batch‐predict helper ────────────────────────────────────────────────
@torch.no_grad()
def predict(texts, model, tok, device, max_length=256):
    model.eval()
    preds = []
    for txt in texts:
        enc = tok(
            txt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        logits = model(enc.input_ids, enc.attention_mask)
        preds.append(int(logits.argmax(-1).cpu()))
    return preds

# ─── 4️⃣ main ────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load both sets
    train_texts, train_labels, train_paths = load_train_split(Path("train/transcription"))
    test_texts,  test_labels,  test_paths  = load_test_set(Path("test"), Path("test/adress_2020_test_Labels.txt"))

    print(f"→ {len(train_texts)} cc/cd samples")
    print(f"→ {len(test_texts)} held-out test samples\n")

    # model + tokenizer
    print("Loading ClinicalBERT model…")
    model     = ADClassifier(dropout=0.1).to(device)
    # point this at whichever checkpoint you want:
    model.load_state_dict(torch.load("best_model_0.84_test.pt", map_location=device))
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # inference on cc/cd “train” split (just as sanity check)
    print("Running inference on cc/cd split…")
    y_train_pred = predict(train_texts, model, tokenizer, device)
    acc = accuracy_score(train_labels, y_train_pred)
    f1 = f1_score(train_labels, y_train_pred)

    print(f" cc/cd  Accuracy: {acc:.4f}")
    print(f" cc/cd  F1      : {f1:.4f}")
    print(classification_report(train_labels, y_train_pred, target_names=["Control","AD"]))

    # inference on held-out test set
    print("Running inference on held-out test set…")
    y_test_pred = predict(test_texts, model, tokenizer, device)
    acc_test = accuracy_score(test_labels, y_test_pred)
    f1_test = f1_score(test_labels, y_test_pred)
    print(f" TEST  Accuracy: {acc_test:.4f}")
    print(f" TEST  F1      : {f1_test:.4f}")
    print(classification_report(test_labels, y_test_pred, target_names=["Control","AD"]))

    # show a few random examples
    print("\nSome test examples:")
    df = pd.DataFrame({
        "path": test_paths,
        "true": test_labels,
        "pred": y_test_pred
    })
    for _, row in df.sample(5, random_state=42).iterrows():
        print(f"{Path(row.path).name:8s} true={row.true}  pred={row.pred}")

if __name__ == "__main__":
    main()
