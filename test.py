# bench_coo.py
import torch, time, argparse, os
import torch.nn as nn

MB = 1024 ** 2

# (vocab_size, input_size, num_classes, seq_len, dual_stream)
DATASET_META = {
    "imdb":       (30522, 1,  2, 4096, False),
    "listops":    (18,    1, 10, 5995, False),
    "retrieval":  (256,   1,  2, 4096, True),
    "pathfinder": (None,  1,  2, 1024, False),
    "ecg":        (None, 12,  6, 4096, False),
    "image":      (None,  1, 10, 1024, False),
}


# ---------- utilities ----------

def _time_op(fn, iters=50, warmup=10):
    for _ in range(warmup): _ = fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

def _mem_coo_bytes(W_coo):
    idx, val = W_coo.indices(), W_coo.values()
    return idx.numel() * idx.element_size() + val.numel() * val.element_size()

def _mem_dense_bytes(W):
    return W.numel() * W.element_size()

def collect_linear_layers(model, skip=("embedding", "fc", "match")):
    """Estrae dal modello nome + shape di ogni nn.Linear prunabile.
    Deduplica le shape ripetute sui layer (i 12 blocchi hanno le stesse)."""
    seen, layers = set(), []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or any(k in name for k in skip):
            continue
        short = name.split(".")[-1]              # 'in_proj', 'out_proj', ...
        shape = tuple(m.weight.shape)            # [out, in]
        if (short, shape) in seen:
            continue
        seen.add((short, shape))
        layers.append((short, shape[0], shape[1]))
    return layers

# ---------- 1. micro-benchmark ----------

@torch.no_grad()
def bench_layer(sparsity, out_f, in_f, n_rows, iters=50):
    dev = "cuda"
    W = torch.randn(out_f, in_f, device=dev)
    W = W * (torch.rand_like(W) > sparsity)
    actual_s = (W == 0).float().mean().item()
    x = torch.randn(n_rows, in_f, device=dev)
    xt = x.T.contiguous()                        # fuori dal timing: generoso verso COO

    t_dense = _time_op(lambda: x @ W.T, iters)
    W_coo = W.to_sparse_coo().coalesce()
    t_coo = _time_op(lambda: torch.sparse.mm(W_coo, xt), iters)

    return {
        "s": actual_s,
        "t_dense_ms": t_dense * 1e3, "t_coo_ms": t_coo * 1e3,
        "speedup": t_dense / t_coo,
        "mem_ratio": _mem_coo_bytes(W_coo) / _mem_dense_bytes(W),
        "idx_dtype": str(W_coo.indices().dtype),
    }

def run_micro(layers, sparsities, batches, seq_len):
    print("=" * 94)
    print("1. MICRO-BENCHMARK — proiezioni lineari (shape estratte dal modello), COO vs denso")
    print("=" * 94)
    print(f"{'layer':>10} {'shape':>12} {'s':>5} {'batch':>6} | {'dense[ms]':>10} "
          f"{'COO[ms]':>10} | {'speedup':>8} | {'mem COO/dense':>14}")
    for name, out_f, in_f in layers:
        for s in sparsities:
            for b in batches:
                r = bench_layer(s, out_f, in_f, b * seq_len)
                print(f"{name:>10} {f'{out_f}x{in_f}':>12} {r['s']:5.2f} {b:6d} | "
                      f"{r['t_dense_ms']:10.3f} {r['t_coo_ms']:10.3f} | "
                      f"{r['speedup']:8.2f} | {r['mem_ratio']:14.2f}")
        print()

# ---------- 2. teoria vs misura ----------

def run_theory_check(out_f, in_f):
    """COO = 4 byte (valore) + 2*8 (coordinate int64) = 20 byte/non-zero.
    Rapporto atteso vs denso (4 byte/elem): 5*(1-s)."""
    print("=" * 94)
    print(f"2. MODELLO TEORICO vs MISURA — memoria COO/denso (su {out_f}x{in_f})")
    print("=" * 94)
    paper = {0.70: 0.45}
    print(f"{'s':>6} | {'atteso 5*(1-s)':>15} | {'misurato':>10} | {'paper':>8}")
    for s in [0.0, 0.5, 0.7, 0.8, 0.9, 0.99]:
        r = bench_layer(s, out_f, in_f, 4096, iters=3)
        p = f"{paper[s]:.2f}" if s in paper else "-"
        print(f"{s:6.2f} | {5.0*(1.0-s):15.2f} | {r['mem_ratio']:10.2f} | {p:>8}")
    print("\nbreak-even COO: 20*(1-s) = 4  ->  s = 0.80")
    print(f"index dtype: {bench_layer(0.7, out_f, in_f, 4096, iters=3)['idx_dtype']}\n")

# ---------- 3. modello reale ----------

@torch.no_grad()
def run_model_memory(model, skip=("embedding",)):
    print("=" * 94)
    print("3. MODELLO COMPLETO — memoria dei pesi lineari reali, denso vs COO")
    print("=" * 94)
    dense_b = coo_b = nnz = numel = 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or any(k in name for k in skip):
            continue
        W = m.weight.data
        coo = W.to_sparse_coo().coalesce()
        dense_b += _mem_dense_bytes(W)
        coo_b   += _mem_coo_bytes(coo)
        nnz     += coo.values().numel()
        numel   += W.numel()
    s = 1.0 - nnz / max(1, numel)
    print(f"sparsità effettiva pesi lineari : {s:.3f}")
    print(f"denso : {dense_b/MB:8.2f} MB")
    print(f"COO   : {coo_b/MB:8.2f} MB   ({coo_b/dense_b:.2f}x del denso)")
    print(f"delta : {(coo_b-dense_b)/MB:+8.2f} MB\n")

# ---------- main ----------

def load_model(dataset, checkpoint):
    from models_config import MODELS_CONFIG
    from models_datasets_and_profiling_implementation.model.net import NetFactory
    from models_datasets_and_profiling_implementation.model.utils import get_device

    cfg = MODELS_CONFIG["mamba"][dataset]
    vocab_size, input_size, num_classes, seq_len, dual_stream = DATASET_META[dataset]

    ckpt = torch.load(checkpoint, map_location=get_device())
    model = NetFactory(
        model_name="mamba", dataset_name=dataset,
        vocab_size=vocab_size, input_size=input_size, num_classes=num_classes,
        d_model=cfg["features"], d_state=cfg["d_state"], depth=cfg["depth"],
        dropout=cfg["dropout"], norm=cfg["norm"], pre_norm=cfg["pre-norm"],
        active_idx_layers=ckpt.get("active_idx_layers"), dual_stream=dataset==dual_stream
    ).get_net().to(get_device())
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, seq_len

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "-d", required=True, choices=list(DATASET_META))
    ap.add_argument("--checkpoint", "-c", required=True)
    args = ap.parse_args()

    model, seq_len = load_model(args.dataset, args.checkpoint)
    layers = collect_linear_layers(model)
    print(f"\nmodello: mamba/{args.dataset} | seq_len={seq_len} | "
          f"checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"layer lineari rilevati: {[(n, f'{o}x{i}') for n, o, i in layers]}\n")

    run_micro(layers, sparsities=[0.5, 0.7, 0.9, 0.99], batches=[1, 32], seq_len=seq_len)
    run_theory_check(*layers[0][1:])
    run_model_memory(model)