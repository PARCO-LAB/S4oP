"""
Prepara il Genomics Long-Range Benchmark (InstaDeepAI/genomics-long-range-benchmark).

NON usa la rete: i file grezzi vanno scaricati altrove e copiati in data/genomics_lrb/raw/
(vedi l'elenco stampato dall'errore se ne manca qualcuno).

Uso, dalla root del repo:
    python scripts/prepare_genomics.py

Input  data/genomics_lrb/raw/     6 CSV del benchmark + hg38.fa.gz + hg19.fa.gz
Output data/genomics_lrb/genome/{hg38,hg19}/chrN.npy   uint8: 1..4 = A,C,G,T, 5 = N (~3.1 GB l'uno)
       data/genomics_lrb/index/*.npz                   coordinate 0-based + label + split

Idempotente: rilanciandolo salta i passi gia' completati.
Sanity check: chr1 deve risultare 248,956,422 bp in hg38 e 249,250,621 bp in hg19.
"""
import gzip
import json
import sys
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/genomics_lrb")
RAW, GENOME, INDEX = ROOT / "raw", ROOT / "genome", ROOT / "index"
PRIMARY = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

HF = "https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark/resolve/main"
UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath/{g}/bigZips/{g}.fa.gz"
SOURCES = {
    "promoter_dataset.csv": f"{HF}/regulatory_elements/promoter_dataset.csv",
    "promoter_dataset_subset.csv": f"{HF}/regulatory_elements/promoter_dataset_subset.csv",
    "enhancer_dataset.csv": f"{HF}/regulatory_elements/enhancer_dataset.csv",
    "enhancer_dataset_subset.csv": f"{HF}/regulatory_elements/enhancer_dataset_subset.csv",
    "histones_and_dnase.csv": f"{HF}/chromatin_features/histones_and_dnase.csv",
    "histones_and_dnase_subset.csv": f"{HF}/chromatin_features/histones_and_dnase_subset.csv",
    "hg38.fa.gz": UCSC.format(g="hg38"),
    "hg19.fa.gz": UCSC.format(g="hg19"),
}

# Stessa normalizzazione di standardize_sequence(): maiuscolo, tutto cio' che non e' ACGT -> N
LUT = np.full(256, 5, dtype=np.uint8)
for tok, base in enumerate(b"ACGT", start=1):
    LUT[base] = tok
    LUT[base + 32] = tok  # minuscole (regioni soft-masked nei FASTA UCSC)


def check_raw():
    missing = [f for f in SOURCES if not (RAW / f).exists()]
    if missing:
        print(f"File mancanti in {RAW.resolve()}:\n", file=sys.stderr)
        for f in missing:
            print(f"  curl -L -o {f} {SOURCES[f]}", file=sys.stderr)
        print("\nScaricali su una macchina con rete e copiali qui.", file=sys.stderr)
        sys.exit(1)


# ----------------------------------------------------------------------------- genoma

def tokenize_genome(g):
    out = GENOME / g
    if (out / ".done").exists():
        return
    print(f"[{g}] tokenizzazione (solo chr1-22, X, Y) ...")
    out.mkdir(parents=True, exist_ok=True)
    name, chunks = None, []

    def flush():
        if name in PRIMARY:
            arr = LUT[np.frombuffer(b"".join(chunks), dtype=np.uint8)]
            np.save(out / f"{name}.npy", arr)
            print(f"  {name}: {len(arr):,} bp, N={np.mean(arr == 5):.1%}")

    with gzip.open(RAW / f"{g}.fa.gz", "rb") as f:
        for line in f:
            if line.startswith(b">"):
                flush()
                name, chunks = line[1:].split()[0].decode(), []
            elif name in PRIMARY:
                chunks.append(line.rstrip())
    flush()
    (out / ".done").touch()


# ----------------------------------------------------------------------------- indici

def _read(fname, cols):
    df = pd.read_csv(RAW / fname)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{fname}: colonne mancanti {missing}, trovate {list(df.columns)}")
    return df


def _common(df):
    """Cromosomi -> codici interi + nomi; split -> 0 train / 1 test."""
    chrom = df["CHROM"].astype(str).str.strip()
    chrom = chrom.where(chrom.str.startswith("chr"), "chr" + chrom)
    names, code = np.unique(chrom.to_numpy(), return_inverse=True)
    split = df["split"].astype(str).str.strip().str.lower()
    unknown = set(split.unique()) - {"train", "test"}
    if unknown:
        raise ValueError(f"valori di split inattesi: {unknown}")
    return dict(chrom_names=names.astype(str), chrom_code=code.astype(np.int16),
                split=(split == "test").to_numpy().astype(np.int8))


def _parse_lists(col):
    """Le label chromatin sono liste salvate come stringhe, es. '[0, 1, 0, ...]'."""
    try:
        return np.array([json.loads(s) for s in col], dtype=np.uint8)
    except (json.JSONDecodeError, TypeError):
        return np.array([literal_eval(s) for s in col], dtype=np.uint8)


def build_regulatory(task, subset):
    sfx = "_subset" if subset else ""
    out = INDEX / f"{task}{sfx}.npz"
    if out.exists():
        return
    df = _read(f"{task}_dataset{sfx}.csv", ["CHROM", "START", "STOP", "label", "split"])
    d = _common(df)
    d["start"] = df["START"].to_numpy(np.int64) - 1  # -1: coordinate CSV 1-based, come nello script ufficiale
    d["end"] = df["STOP"].to_numpy(np.int64) - 1
    d["label"] = df["label"].to_numpy(np.int64)
    np.savez(out, **d)
    for s, nm in ((0, "train"), (1, "test")):
        m = d["split"] == s
        print(f"[{out.name}] {nm}: {m.sum():,} righe, positivi {d['label'][m].mean():.1%}")
    print(f"[{out.name}] valori di STOP-START: {np.unique(d['end'] - d['start'])[:5].tolist()}")


def build_chromatin(subset):
    sfx = "_subset" if subset else ""
    out = INDEX / f"chromatin{sfx}.npz"
    if out.exists():
        return
    df = _read(f"histones_and_dnase{sfx}.csv", ["CHROM", "POS", "HISTONES", "DNASE", "split"])
    d = _common(df)
    d["pos"] = df["POS"].to_numpy(np.int64) - 1
    d["histones"] = _parse_lists(df["HISTONES"])
    d["dnase"] = _parse_lists(df["DNASE"])
    np.savez(out, **d)
    for s, nm in ((0, "train"), (1, "test")):
        m = d["split"] == s
        print(f"[{out.name}] {nm}: {m.sum():,} righe, positivi medi per traccia: "
              f"histone {d['histones'][m].mean():.1%}, dnase {d['dnase'][m].mean():.1%}")


if __name__ == "__main__":
    check_raw()
    INDEX.mkdir(parents=True, exist_ok=True)
    for subset in (True, False):
        for task in ("promoter", "enhancer"):
            build_regulatory(task, subset)
        build_chromatin(subset)
    for g in ("hg38", "hg19"):
        tokenize_genome(g)
    print("Fatto. Se serve spazio puoi cancellare data/genomics_lrb/raw/*.fa.gz")