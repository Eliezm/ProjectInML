# explanatory_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 1️⃣ Load features ───────────────────────────────────────────────────
df = pd.read_csv("transcript_features_expanded.csv")

# ─── 2️⃣ Infer binary label from path ────────────────────────────────────
def infer_label(p: str) -> int:
    if "/cc/" in p or "\\cc\\" in p:
        return 0
    if "/cd/" in p or "\\cd\\" in p:
        return 1
    return -1

df["label"]      = df["path"].apply(infer_label)
df = df[df.label >= 0].copy()
df["class"]      = df.label.map({0:"Control",1:"AD"})
print("Samples per class:\n", df["class"].value_counts(), "\n")

# ─── 3️⃣ Quick demographics & MMSE check ─────────────────────────────────
# Age
plt.figure(figsize=(6,4))
sns.histplot(data=df, x="age", hue="class", multiple="stack", bins=10)
plt.title("Age Distribution by Class")
plt.xlabel("Age"); plt.tight_layout(); plt.show()

# Gender
plt.figure(figsize=(5,4))
sns.countplot(data=df, x="class", hue="gender")
plt.title("Gender by Class"); plt.tight_layout(); plt.show()

# MMSE (if numeric)
if df["mmse"].dtype == object:
    mmse = pd.to_numeric(df["mmse"], errors="coerce")
else:
    mmse = df["mmse"]
if mmse.notna().any():
    plt.figure(figsize=(6,4))
    sns.histplot(mmse, bins=10, kde=True)
    plt.title("MMSE Score Distribution"); plt.tight_layout(); plt.show()

# ─── 4️⃣ Core-length distributions ───────────────────────────────────────
# total tokens
plt.figure(figsize=(6,4))
sns.histplot(df["num_tokens"], bins=30, color="skyblue", edgecolor="k")
plt.axvline(256, color="r", linestyle="--", label="256")
plt.axvline(512, color="g", linestyle="--", label="512")
plt.title("Total Token Count per Transcript")
plt.legend(); plt.tight_layout(); plt.show()

# avg tokens per utterance
plt.figure(figsize=(6,4))
sns.histplot(df["avg_tok_per_utt"], bins=30, color="lightgreen", edgecolor="k")
plt.title("Avg Tokens per Utterance"); plt.tight_layout(); plt.show()

# avg tokens per sentence
plt.figure(figsize=(6,4))
sns.histplot(df["avg_tok_per_sent"], bins=30, color="violet", edgecolor="k")
plt.title("Avg Tokens per Sentence"); plt.tight_layout(); plt.show()

# ─── 5️⃣ Fluency & edit rates ────────────────────────────────────────────
# Disfluency rate by class
plt.figure(figsize=(6,4))
sns.kdeplot(data=df, x="disfl_rate", hue="class", fill=True)
plt.title("Disfluency Rate by Class"); plt.tight_layout(); plt.show()

# bracket‐edits
for col, name in [("angle_edits","<> edits"),("square_edits","[] edits"),("paren_edits","() edits")]:
    plt.figure(figsize=(5,3))
    sns.boxplot(data=df, x="class", y=col)
    plt.title(f"{name} by Class"); plt.tight_layout(); plt.show()

# ─── 6️⃣ Lexical richness ────────────────────────────────────────────────
# TTR
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="class", y="ttr")
plt.title("Type–Token Ratio by Class"); plt.tight_layout(); plt.show()

# Hapax ratio
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="class", y="hapax_ratio")
plt.title("Hapax Ratio by Class"); plt.tight_layout(); plt.show()

# ─── 7️⃣ Information‐Unit and content‐word rates ────────────────────────
plt.figure(figsize=(6,4))
sns.kdeplot(df["iu_rate"], label="IU rate", color="navy")
sns.kdeplot(df["content_word_rate"], label="Content‐word rate", color="teal")
plt.title("IU vs Content‐Word Rates (all samples)")
plt.legend(); plt.tight_layout(); plt.show()

# IU‐rate by class
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="class", y="iu_rate")
plt.title("IU Rate by Class"); plt.tight_layout(); plt.show()

# ─── 8️⃣ Punctuation usage ───────────────────────────────────────────────
punct_cols = ["comma_count","period_count","question_count","exclaim_count"]
punct_sum = df[punct_cols].sum()
plt.figure(figsize=(6,4))
punct_sum.plot(kind="bar", color="coral", edgecolor="k")
plt.title("Total Punctuation Counts (all transcripts)")
plt.ylabel("Count"); plt.tight_layout(); plt.show()

# ─── 9️⃣ Age vs Fluency scatter ────────────────────────────────────────
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="age", y="disfl_rate", hue="class")
plt.title("Age vs Disfluency Rate"); plt.tight_layout(); plt.show()

# ── 🔟 Correlation Heatmap (selected features) ──────────────────────────
sel = [
    "num_tokens","avg_tok_per_utt","disfl_rate","ttr",
    "iu_rate","content_word_rate","hapax_ratio"
]
corr = df[sel].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Feature Correlation Matrix"); plt.tight_layout(); plt.show()

# ─── 1️⃣1️⃣ Outlier reporting ───────────────────────────────────────────
print("\n⚠️  Very long transcripts (>2000 tokens):")
print(df[df.num_tokens>2000][["path","num_tokens","class"]], "\n")
print("⚠️  Very short transcripts (<50 tokens):")
print(df[df.num_tokens<50][["path","num_tokens","class"]])
