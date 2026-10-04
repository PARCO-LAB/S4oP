"""
Genomics Long-Range Benchmark (InstaDeepAI): regulatory elements + chromatin features.
Richiede di aver lanciato una volta scripts/prepare_genomics.py.

Le finestre replicano pad_sequence() dello script ufficiale del benchmark (stesso inizio,
stessi scarti ai bordi del cromosoma) ma sono ritagliate on-the-fly dal genoma tokenizzato
in memmap: nessuna cache per lunghezza, seq_len libero.

    LRBPromoter   binaria sbilanciata (~5% positivi) -> AUPRC     hg38
    LRBEnhancer   binaria ~bilanciata                -> AUROC     hg38
    LRBHistone    multi-label, 20 tracce             -> AUPRC     hg19
    LRBDnase      multi-label, 20 tracce             -> AUPRC     hg19
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .interface import DatasetInterface

__all__ = ["LRBPromoter", "LRBEnhancer", "LRBHistone", "LRBDnase"]

ROOT = Path("data/genomics_lrb")
VOCAB_SIZE = 6   # 0 = pad (mai prodotto: le finestre sono sempre lunghe seq_len), 1..4 = A,C,G,T, 5 = N
SEQ_LEN = 4096   # default, coerente con imdb/ecg
VAL_CHROMS = ("chr7",)  # validazione = un cromosoma intero, vedi _LRBTask._setup


class GenomeWindows(Dataset):
    def __init__(self, genome_dir, chrom_names, chrom_idx, starts, labels, seq_len):
        self.genome_dir = Path(genome_dir)
        self.chrom_names = chrom_names
        self.chrom_idx = chrom_idx  # [N] indice in chrom_names
        self.starts = starts        # [N] inizio finestra, 0-based
        self.labels = labels        # [N] int64 (CrossEntropy) oppure [N, K] float32 (BCE)
        self.seq_len = seq_len
        self._genome = None         # memmap aperti lazy, uno per worker

    def __getstate__(self):         # non serializzare i memmap se i worker non partono con fork
        state = self.__dict__.copy()
        state["_genome"] = None
        return state

    def _get_genome(self):
        if self._genome is None:
            paths = [self.genome_dir / f"{c}.npy" for c in self.chrom_names]
            self._genome = [np.load(p, mmap_mode="r") if p.exists() else None for p in paths]
        return self._genome

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        chrom = self._get_genome()[self.chrom_idx[i]]
        s = self.starts[i]
        x = torch.from_numpy(chrom[s:s + self.seq_len].astype(np.int64))
        return x, torch.tensor(self.labels[i])


class _LRBTask(DatasetInterface):
    """Parte comune: finestre valide + split train/val/test per cromosoma."""

    def _setup(self, genome, d, win_start, labels, seq_len, val_chroms):
        assert seq_len >= 200, "il benchmark richiede seq_len >= 200 (le label coprono un bin di 200 bp)"
        genome_dir = ROOT / "genome" / genome
        names = [str(c) for c in d["chrom_names"]]
        code = d["chrom_code"].astype(np.int64)
        paths = [genome_dir / f"{c}.npy" for c in names]
        if not any(p.exists() for p in paths):
            raise FileNotFoundError(f"{genome_dir} vuota: lancia prima scripts/prepare_genomics.py")
        lengths = np.array([np.load(p, mmap_mode="r").shape[0] if p.exists() else 0 for p in paths])

        # stessa condizione di pad_sequence(): start >= 0 e end < len(chrom).
        # Esclude anche i contig non primari, per cui non esiste il .npy (lengths = 0).
        valid = (win_start >= 0) & (win_start + seq_len < lengths[code])
        is_val = np.isin(code, [k for k, c in enumerate(names) if c in val_chroms])
        is_test = d["split"] == 1

        def make(mask):
            return GenomeWindows(genome_dir, names, code[mask], win_start[mask], labels[mask], seq_len)

        self.trainset = make(valid & ~is_test & ~is_val)
        self.valset = make(valid & ~is_test & is_val)
        self.testset = make(valid & is_test)
        if len(self.valset) == 0:
            raise ValueError(f"validazione vuota: {val_chroms} non e' tra i cromosomi di train")

        self.n_dropped = int((~valid).sum())
        self.seq_len = seq_len
        self.input_size = 1           # un token per timestep
        self.vocab_size = VOCAB_SIZE  # tokenizzato -> nn.Embedding
        self.input_shape = (self.batch_size, seq_len)
        self.metric = "auprc"
        self.pool = "causal_half"


class _LRBRegulatory(_LRBTask):
    """regulatory_element_{promoter,enhancer}: binaria -> CrossEntropy. Genoma hg38."""
    task = None

    def __init__(self, batch_size, valsplit, num_workers,
                 seq_len=SEQ_LEN, subset=False, val_chroms=VAL_CHROMS):
        # valsplit non usato: la validazione e' un cromosoma intero (niente leakage tra finestre vicine)
        super().__init__(f"{self.task}", batch_size, num_workers)

        d = dict(np.load(ROOT / "index" / f"{self.task}{'_subset' if subset else ''}.npz"))
        start, end = d["start"], d["end"]
        win_start = start - (seq_len - (end - start)) // 2  # stesso inizio di pad_sequence(start, end)
        self._setup("hg38", d, win_start, d["label"], seq_len, val_chroms)

        self.labels = [0, 1]
        self.num_classes = 2
        pos = torch.bincount(torch.as_tensor(self.testset.labels), minlength=2).tolist()
        self.summary(extra=f"pos/classe(test)={pos} | val={','.join(val_chroms)} | scartati={self.n_dropped}")


class _LRBChromatin(_LRBTask):
    """chromatin_features_{histone_marks,dna_accessibility}: multi-label (20) -> BCEWithLogits. Genoma hg19."""
    task = None   # nome del dataset
    key = None    # colonna nell'indice: "histones" oppure "dnase"

    def __init__(self, batch_size, valsplit, num_workers,
                 seq_len=SEQ_LEN, subset=False, val_chroms=VAL_CHROMS):
        # valsplit non usato: la validazione e' un cromosoma intero (niente leakage tra finestre vicine)
        super().__init__(f"{self.task}", batch_size, num_workers)
        self.multilabel = True

        d = dict(np.load(ROOT / "index" / f"chromatin{'_subset' if subset else ''}.npz"))
        win_start = d["pos"] - seq_len // 2   # stesso inizio di pad_sequence(pos) senza end
        labels = d[self.key].astype(np.float32)
        self._setup("hg19", d, win_start, labels, seq_len, val_chroms)

        self.num_classes = labels.shape[1]   # 20
        self.labels = list(range(self.num_classes))
        pos = self.testset.labels.sum(axis=0).astype(int).tolist()
        self.summary(extra=f"pos/classe(test)={pos} | val={','.join(val_chroms)} | scartati={self.n_dropped}")


class LRBPromoter(_LRBRegulatory):
    task = "promoter"


class LRBEnhancer(_LRBRegulatory):
    task = "enhancer"


class LRBHistone(_LRBChromatin):
    task = "histone"
    key = "histones"


class LRBDnase(_LRBChromatin):
    task = "dnase"
    key = "dnase"