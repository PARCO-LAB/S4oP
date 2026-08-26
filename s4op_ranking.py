import argparse
import gc
import os
import time

import torch

from s4op import get_pruning_idx_exponential, convert_layer_mamba, pretty_time
from models_datasets_and_profiling_implementation.model import FineTuning, ModelTest
from models_datasets_and_profiling_implementation.model.net import NetFactory
from models_datasets_and_profiling_implementation.model.utils import set_benchmark, set_seed, get_device
from models_config import *
from pruning_config import *


def _sal(p, alpha):
    """s(w) = |w| * |grad(w)|^alpha"""
    if p.grad is None:
        return torch.zeros_like(p)
    return p.detach().abs() * p.grad.detach().abs().pow(alpha) # detach -> stacca il tensore dal grafo autograd; serve perché qui stai facendo puro calcolo numerico (score di importanza),
                                                               # non vuoi che queste operazioni finiscano nel grafo computazionale e generino ulteriori gradienti. 
                                                               # È un modo per dire "usami solo i valori, non tracciare nulla"


@torch.no_grad()                                               # Essendoci questo il detach sopra non serve in realtà però vabbè
def channel_scores(layer, alpha=1.0):
    d = layer.d_inner
    s = _sal(layer.in_proj.weight, alpha)
    total = s[:d].sum(dim=1) + s[d:].sum(dim=1)
    total = total + _sal(layer.conv1d.weight, alpha).sum(dim=(1, 2))
    total = total + _sal(layer.conv1d.bias, alpha)
    total = total + _sal(layer.x_proj.weight, alpha).sum(dim=0)
    total = total + _sal(layer.dt_proj.weight, alpha).sum(dim=1)
    total = total + _sal(layer.dt_proj.bias, alpha)
    total = total + _sal(layer.A_log, alpha).sum(dim=1)
    total = total + _sal(layer.D, alpha)
    return total


def accumulate_grads(model, loader, criterion, n_batches=5):
    device = next(model.parameters()).device
    model.eval()
    model.zero_grad(set_to_none=True)
    it = iter(loader)
    used = 0
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        criterion(model(x.to(device)), y.to(device)).backward()
        used += 1
    return used


def select_channels(model, prev_active, n_pruned, alpha=1.0):
    active_idx = []
    for i, layer in enumerate(model.mamba_layers):
        available = list(prev_active[i])
        if layer.d_inner != len(available):
            raise RuntimeError(f"layer {i}: d_inner={layer.d_inner} ma prev_active ne ha {len(available)}")

        k = min(max(0, int(n_pruned[i])), len(available) - 1)
        if k == 0:
            active_idx.append(available)
            print(f"[Layer {i}]: canali rimanenti {len(available)}")
            continue

        scores = channel_scores(layer, alpha)
        kill = set(torch.argsort(scores)[:k].tolist())
        keep = [g for j, g in enumerate(available) if j not in kill]
        active_idx.append(keep)
        print(f"[Layer {i}]: canali rimanenti {len(keep)} (tolti {k})")
    return active_idx


def prune_and_finetune(args):
    config = MODELS_CONFIG
    pc = PRUNING_CONFIG[args.model_name][args.dataset_name]
    pc2 = PRUNING_CONFIG
    cfg = config[args.model_name][args.dataset_name]

    os.makedirs(args.checkpoint_folder, exist_ok=True)
    path = os.path.join(f"./{args.checkpoint_folder}", f"{args.model_name}_{args.dataset_name}_pruned_{int(pc2['perc'][0]*100)}%.pth")
    if os.path.exists(path):
        raise ValueError(f"Il file {path} esiste già. Scegliere un'altra cartella o un altro nome per il file.")
    set_seed(42)
    set_benchmark(False)

    H = cfg["features"] * 2           
    n_layers = cfg["depth"]
    model_checkpoint = (f"./{args.base_model_folder}/{args.model_name}_{args.dataset_name}_best.pth")

    accuracies, timers = {}, {}

    for perc in pc2["perc"]:
        print(f"\n=== PRUNING {perc*100:.0f}% ===")
        t1 = time.time()

        prev_model = ModelTest.from_pth(
            model_path=model_checkpoint,
            batch_size=cfg["batch_size"],
            valsplit=config["val_split"],
            num_workers=config["num_workers"],
            d_model=cfg["features"], 
            d_state=cfg["d_state"],
            depth=n_layers, 
            dropout=cfg["dropout"],
            norm=cfg["norm"], 
            pre_norm=cfg["pre-norm"],
        )
        ckpt = torch.load(model_checkpoint, map_location=get_device())
        prev_active = [list(idx) for idx in ckpt["active_idx_layers"]]

        # budget per layer
        n_pruned_perc = get_pruning_idx_exponential(perc, n_layers, H)
        n_pruned = [max(0, n_pruned_perc[i] - (H - len(prev_active[i]))) for i in range(n_layers)]

        # ranking 
        criterion = torch.nn.BCEWithLogitsLoss() if prev_model.dataset.multilabel else torch.nn.CrossEntropyLoss()
        used = accumulate_grads(prev_model.model, prev_model.model_train.trainloader, criterion, args.importance_batches)
        print(f"gradienti accumulati su {used} batch")
        active_idx = select_channels(prev_model.model, prev_active, n_pruned, alpha=args.alpha)
        prev_model.model.zero_grad(set_to_none=True)

        # costruzione del modello prunato 
        model = NetFactory(
            model_name=args.model_name, 
            dataset_name=args.dataset_name,
            vocab_size=prev_model.dataset.vocab_size,
            input_size=prev_model.dataset.input_size,
            d_model=cfg["features"], 
            d_state=cfg["d_state"],
            depth=n_layers, 
            dropout=cfg["dropout"],
            num_classes=prev_model.dataset.num_classes,
            norm=cfg["norm"], 
            pre_norm=cfg["pre-norm"],
            active_idx_layers=active_idx,
            dual_stream=prev_model.dataset.dual_stream,
        ).get_net()

        model.embedding.load_state_dict(prev_model.model.embedding.state_dict())
        for i, (lm, ls) in enumerate(zip(prev_model.model.mamba_layers, model.mamba_layers)):
            mapping = {int(g): j for j, g in enumerate(prev_active[i])}
            local_idx = [mapping[int(g)] for g in active_idx[i]]
            convert_layer_mamba(lm, ls, local_idx)
        for nm, ns in zip(prev_model.model.norms, model.norms):
            ns.load_state_dict(nm.state_dict())
        if hasattr(model, "norm_f"):
            model.norm_f.load_state_dict(prev_model.model.norm_f.state_dict())
        if hasattr(model, "fc") and hasattr(prev_model.model, "fc"):
            model.fc.load_state_dict(prev_model.model.fc.state_dict())
        if hasattr(model, "match") and hasattr(prev_model.model, "match"):
            model.match.load_state_dict(prev_model.model.match.state_dict())

        model_test = ModelTest(model=model, dataset=prev_model.dataset)
        model, dataset = model_test.model, model_test.dataset

        # fine-tuning 
        path = os.path.join(f"./{args.checkpoint_folder}", f"{args.model_name}_{args.dataset_name}_pruned_{int(perc*100)}%.pth")

        trainer = FineTuning(
            model=model, 
            dataset=dataset,
            epochs=pc["finetune_epochs"], 
            lr=pc["lr"],
            weight_decay=pc["weight_decay"],
            checkpoint_folder=path, 
            patience=pc["early_stopping"])
        trainer.run()
        acc = model_test.run()

        accuracies[perc] = acc
        model_checkpoint = path
        timers[perc] = time.time() - t1

        del trainer, model, model_test, dataset, prev_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== ACCURACY PER OGNI STEP ===")
    for p, a in accuracies.items():
        print(f"    - Pruning {int(p*100)}%: {a:.2f}%")
    print("\n=== TEMPI DI ESECUZIONE ===")
    for p, t in timers.items():
        print(f"    - Pruning {int(p*100)}%: {pretty_time(t)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run")
    ap.add_argument("--model", "-m", dest="model_name", required=True)
    ap.add_argument("--dataset", "-d", dest="dataset_name", required=True)
    ap.add_argument("--base_model_folder", "-b", dest="base_model_folder", required=True)
    ap.add_argument("--checkpoint_folder", "-c", dest="checkpoint_folder", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--importance_batches", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    prune_and_finetune(args)
    print(f"\nTempo totale di esecuzione: {pretty_time(time.time() - t0)}")