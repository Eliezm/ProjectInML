import re
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from collections import Counter
import spacy

# ─── 1️⃣ Regex patterns ─────────────────────────────────────────────────────
UTT_RE           = re.compile(r"^\*([A-Z]{3}):\s*(.+)$")
DEMOG_LINE_RE    = re.compile(
    r"^@ID:\s*eng\|Pitt\|(PAR|INV)\|([^|]+)\|([^|]+)\|([^|]+)\|\|"
)
TIMESTAMP_RE     = re.compile(r"\d+_\d+")          # timing markers
DISFLUENCY_RE    = re.compile(r"&\w+")               # &uh &um…
ANGLE_BRACKETS   = re.compile(r"<[^>]+>")            # <…>
SQUARE_BRACKETS  = re.compile(r"\[[^\]]+\]")         # [… ]
PARENTHESIS_RE   = re.compile(r"\([^)]*\)")          # (d), (.), etc.
SENTENCE_SPLIT   = re.compile(r"[\.!?]\s+")          # rough sentence splitter

# ─── 2️⃣ spaCy for POS & morphology ─────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

# ─── 3️⃣ Parse one .cha ─────────────────────────────────────────────────────
def parse_cha(path: Path) -> Dict:
    demographics: Dict[str, Dict] = {}
    utts: List[str] = []
    sign_counts = Counter()

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("@ID:"):
            m = DEMOG_LINE_RE.match(line)
            if m:
                role, age, gender, mmse = m.groups()
                demographics[role] = {
                    "age":    age.strip() or None,
                    "gender": gender.strip() or None,
                    "mmse":   mmse.strip() or None
                }
        elif line.startswith("*"):
            m = UTT_RE.match(line)
            if not m: continue
            spk, text = m.groups()
            if spk != "PAR":   # only patient
                continue
            # strip timing codes
            text = TIMESTAMP_RE.sub("", text)
            # count all special sign occurrences
            sign_counts.update(DISFLUENCY_RE.findall(text))
            sign_counts.update(ANGLE_BRACKETS.findall(text))
            sign_counts.update(SQUARE_BRACKETS.findall(text))
            sign_counts.update(PARENTHESIS_RE.findall(text))
            utts.append(text)

    combined = " ".join(utts)
    return {
        "path":          str(path),
        "demographics":  demographics,
        "utterances":    utts,
        "combined_text": combined,
        "sign_counts":   sign_counts
    }

# ─── 4️⃣ Load all transcripts ────────────────────────────────────────────────
def load_all_transcripts(folder: Path) -> List[Dict]:
    all_recs = []
    for cha in sorted(folder.rglob("*.cha")):
        rec = parse_cha(cha)
        all_recs.append(rec)
    return all_recs

# ─── 5️⃣ Extract 30+ features ───────────────────────────────────────────────
def extract_features(recs: List[Dict]) -> pd.DataFrame:
    rows = []
    for R in recs:
        text = R["combined_text"]
        demo = R["demographics"].get("PAR", {})
        tokens = text.split()
        num_tokens = len(tokens)
        num_utts   = len(R["utterances"])
        num_sents  = max(1, len(SENTENCE_SPLIT.split(text)))
        avg_tok_per_utt   = num_tokens/num_utts if num_utts else 0
        avg_tok_per_sent  = num_tokens/num_sents if num_sents else 0

        # word‐length stats
        word_lens = [len(t) for t in tokens]
        avg_wlen  = np.mean(word_lens) if word_lens else 0
        std_wlen  = np.std(word_lens)  if word_lens else 0

        # disfluencies
        total_disfl  = sum(v for k,v in R["sign_counts"].items() if k.startswith("&"))
        uh_count     = R["sign_counts"].get("&uh",0)
        um_count     = R["sign_counts"].get("&um",0)
        disfl_rate   = total_disfl/num_tokens if num_tokens else 0

        # bracket‐edits
        angle_edits  = sum(R["sign_counts"][k] for k in R["sign_counts"] if k.startswith("<"))
        square_edits = sum(R["sign_counts"][k] for k in R["sign_counts"] if k.startswith("["))
        paren_edits  = sum(R["sign_counts"][k] for k in R["sign_counts"] if PARENTHESIS_RE.fullmatch(k))

        # lexical richness
        uniq_tokens   = len(set(tokens))
        ttr           = uniq_tokens/num_tokens if num_tokens else 0
        hapax         = sum(1 for t,c in Counter(tokens).items() if c==1)
        hapax_ratio   = hapax/num_tokens if num_tokens else 0

        # POS distributions & Information Units
        doc = nlp(text)
        pos_counts = Counter(tok.pos_ for tok in doc)
        noun_cnt    = pos_counts["NOUN"]
        verb_cnt    = pos_counts["VERB"]
        propn_cnt   = pos_counts["PROPN"]
        adj_cnt     = pos_counts["ADJ"]
        adv_cnt     = pos_counts["ADV"]
        pron_cnt    = pos_counts["PRON"]
        cc_cnt      = pos_counts["CCONJ"] + pos_counts["SCONJ"]  # conjunctions
        iu_cnt      = noun_cnt + verb_cnt + propn_cnt + pos_counts["ADP"]  # as in paper
        iu_rate     = iu_cnt/num_tokens if num_tokens else 0
        contw_cnt   = noun_cnt+verb_cnt+adj_cnt+adv_cnt  # content words
        contw_rate  = contw_cnt/num_tokens if num_tokens else 0

        # punctuation stats
        punct_counts = Counter(ch for ch in text if ch in ".,;:!?")
        comma_cnt     = punct_counts[","]
        period_cnt    = punct_counts["."]
        ques_cnt      = punct_counts["?"]
        excl_cnt      = punct_counts["!"]

        rows.append({
            "path":           R["path"],
            "age":            demo.get("age"),
            "gender":         demo.get("gender"),
            "mmse":           demo.get("mmse"),

            "num_tokens":         num_tokens,
            "num_utterances":     num_utts,
            "num_sentences":      num_sents,
            "avg_tok_per_utt":    avg_tok_per_utt,
            "avg_tok_per_sent":   avg_tok_per_sent,

            "avg_word_len":       avg_wlen,
            "std_word_len":       std_wlen,

            "total_disfluencies": total_disfl,
            "uh_count":           uh_count,
            "um_count":           um_count,
            "disfl_rate":         disfl_rate,

            "angle_edits":        angle_edits,
            "square_edits":       square_edits,
            "paren_edits":        paren_edits,

            "uniq_tokens":        uniq_tokens,
            "ttr":                ttr,
            "hapax_count":        hapax,
            "hapax_ratio":        hapax_ratio,

            "noun_count":         noun_cnt,
            "verb_count":         verb_cnt,
            "propn_count":        propn_cnt,
            "adj_count":          adj_cnt,
            "adv_count":          adv_cnt,
            "pron_count":         pron_cnt,
            "conj_count":         cc_cnt,

            "iu_count":           iu_cnt,
            "iu_rate":            iu_rate,
            "content_word_count": contw_cnt,
            "content_word_rate":  contw_rate,

            "comma_count":        comma_cnt,
            "period_count":       period_cnt,
            "question_count":     ques_cnt,
            "exclaim_count":      excl_cnt,
        })

    df = pd.DataFrame(rows).set_index("path")
    return df

# ─── 6️⃣ Quick test & save ────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR = Path("train/transcription")  # adjust as needed
    transcripts = load_all_transcripts(BASE_DIR)
    print(f"Parsed {len(transcripts)} transcripts.")

    # sample one
    sample = transcripts[0]
    print("\n── Sample parse ──")
    print("Path:", sample["path"])
    print("Demographics:", sample["demographics"])
    print("First utt:", sample["utterances"][0][:80], "…")
    print("First 10 sign‐counts:", sample["sign_counts"].most_common(10))
    print("Combined text excerpt:", sample["combined_text"][:200], "…")

    # extract full feature set
    feat_df = extract_features(transcripts)
    print("\n── Feature matrix head ──")
    print(feat_df.head())
    feat_df.to_csv("transcript_features_expanded.csv")
    print("\nSaved => transcript_features_expanded.csv")
