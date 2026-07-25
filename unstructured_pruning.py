import os
import argparse
import torch

from models_datasets_and_profiling_implementation.model import ModelTrain, ModelTest
from models_config import *
from pruning_config import * 

# Schedule cubico della sparsità
def cubic_sparsity(epoch, ramp_epochs, s_final, start_frac=0.25):
    """Sparsita' target all'epoca `epoch`: sale in modo cubico da 0 a s_final.
      - prima di start_frac*ramp_epochs -> 0  (warm-up, niente pruning)
      - a fine rampa (ramp_epochs)      -> s_final
      - oltre                           -> resta s_final (solo fine-tuning)
    """
    t0 = start_frac * ramp_epochs
    if epoch <= t0:
        return 0.0
    if epoch >= ramp_epochs:
        return s_final
    r = 1.0 - (epoch - t0) / (ramp_epochs - t0)
    return s_final * (1.0 - r ** 3)

# Pruner
class MambaPruner:
    """Maschere binarie (1 = tengo il peso, 0 = lo pruno)."""

    def __init__(self, model, alpha=1.0, include_embeddings=False, prune_A_log=False):
        self.model = model
        self.alpha = alpha
        self.weights = {} # prunabili
        self.masks = {} # nome -> maschera
        self.protected_numel = 0 # contati eventualmente nel denominatore, mai prunati (A_log)

        for name, p in model.named_parameters():
            if not p.requires_grad:  continue
            if p.dim() < 2: continue
            is_embed = "embed" in name.lower()
            is_A_log = "A_log" in name

            if (not include_embeddings) and is_embed:
                continue # embedding: fuori dal tutto
            if is_A_log and (not prune_A_log):
                self.protected_numel += p.numel() # protetto ma contato
                continue

            self.weights[name] = p
            self.masks[name] = torch.ones_like(p)

        # Hook sui gradienti: azzera il gradiente dei pesi verrann potati, cosi' optimizer/weight-decay non li resuscitano
        for name, p in self.weights.items():
            p.register_hook(lambda g, n=name: g * self.masks[n])

    @torch.no_grad()
    def enforce(self):
        """Forza a zero i pesi potati. Da chiamare dopo OGNI optimizer.step()."""
        for name, p in self.weights.items():
            p.mul_(self.masks[name])

    @torch.no_grad()
    def current_sparsity(self):
        zeros = sum((m == 0).sum().item() for m in self.masks.values())
        total = sum(m.numel() for m in self.masks.values()) + self.protected_numel
        return zeros / max(1, total)

    def update_mask(self, target_sparsity, trainloader, criterion, n_batches=5):
        """Ricalcola l'importanza S(w)=|w|*|grad|^alpha e aggiorna la maschera
        globale per arrivare a `target_sparsity`.
        """
        if target_sparsity <= 0:
            return self.current_sparsity()

        device = next(self.model.parameters()).device
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        it = iter(trainloader)
        for _ in range(n_batches):
            try:
                inputs, labels = next(it)
            except StopIteration:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            criterion(self.model(inputs), labels).backward()

        # Non metto optimizer.sptep() perchè non sto allenando, sto solo calcolando i gradienti

        scores = {}
        for name, p in self.weights.items():
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            scores[name] = p.detach().abs() * g.detach().abs().pow(self.alpha)
        self.model.zero_grad(set_to_none=True) # Lo metto così non rimane "sporcizia" per dopo quando verra fatto il train step

        # Classifica globale -> soglia di taglio
        flat = torch.cat([s.flatten().cpu() for s in scores.values()])

        # Calcola la % esatta di parametri da prunare (se prune_A_log=False)
        prunable_total = flat.numel()
        pool_total = prunable_total + self.protected_numel
        k_global = int(round(target_sparsity * pool_total))
        k = max(1, min(k_global, prunable_total - 1))
        if k_global > prunable_total - 1:
            print(f"[WARN] {target_sparsity:.0%} irraggiungibile: A_log protetto è "
                f"{self.protected_numel/pool_total:.1%} del pool")
        threshold = torch.kthvalue(flat, k).values.to(device) # Prende il k-esimo valore più piccolo

        with torch.no_grad():
            for name in self.weights:
                self.masks[name] = (scores[name] > threshold).to(self.masks[name].dtype)
            self.enforce()
        return self.current_sparsity()

    @torch.no_grad()
    def sparsity_per_group(self):
        groups = {"ssm": [0, 0], "linear": [0, 0]}
        ssm_tags = ("x_proj", "dt_proj", "conv1d", "A_log")
        for name, m in self.masks.items():
            key = "ssm" if any(t in name for t in ssm_tags) else "linear"
            groups[key][0] += (m == 0).sum().item()
            groups[key][1] += m.numel()
        groups["ssm"][1] += self.protected_numel   # A_log conta nel denom. ssm
        return {k: v[0] / max(1, v[1]) for k, v in groups.items()}

# FINE-TUNER
class PruningFineTuning:
    def __init__(self, model, dataset, epochs, ramp_epochs, s_final,
                 lr, lr_min, weight_decay, checkpoint_path,
                 alpha=1.0, start_frac=0.25, importance_batches=5,
                 prune_A_log=True):
        self.mt = ModelTrain(model, dataset)
        self.pruner = MambaPruner(model, alpha=alpha, prune_A_log=prune_A_log)
        self.epochs = epochs
        self.ramp_epochs = ramp_epochs
        self.s_final = s_final
        self.lr = lr
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.checkpoint_path = checkpoint_path
        self.start_frac = start_frac
        self.importance_batches = importance_batches

    def run(self):
        is_ecg = self.mt.dataset.multilabel
        base_criterion = torch.nn.BCEWithLogitsLoss() if is_ecg else torch.nn.CrossEntropyLoss()

        # Optimizer/scheduler interni, come Table 8: AdamW + decay lineare lr->lr_min
        optimizer = torch.optim.AdamW(
            self.mt.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0,
            end_factor=max(self.lr_min / self.lr, 1e-6), total_iters=self.epochs)

        # Hook: riazzera i pesi potati dopo ogni optimizer.step()
        optimizer.register_step_post_hook(lambda *a, **k: self.pruner.enforce())

        best = 0.0
        for epoch in range(self.epochs):
            target = cubic_sparsity(epoch, self.ramp_epochs, self.s_final, self.start_frac)
            sp = self.pruner.update_mask(target, self.mt.trainloader,
                                         base_criterion, self.importance_batches)

            epoch_loss = self.mt.train_step(epoch, optimizer, base_criterion)

            if is_ecg:
                val_acc, val_loss, val_f1 = self.mt.val_step(base_criterion)
                metric = val_f1
            else:
                val_acc, val_loss = self.mt.val_step(base_criterion)
                metric = val_acc

            scheduler.step()

            grp = self.pruner.sparsity_per_group()
            print(f"[Epoch {epoch+1}] val_loss: {val_loss:.3f} | metric: {metric:.3f} | sparsity: {sp:.3f} (ssm {grp['ssm']:.2f} / lin {grp['linear']:.2f})")

            # considera il salvataggio solo quando sei alla sparsità target
            at_target = sp >= self.s_final - 1e-6
            if at_target and metric > best:
                best = metric
                torch.save({
                    "model_state_dict": self.mt.model.state_dict(),
                    "active_idx_layers": self.mt.model.active_idx_layers,
                    "pruning_masks": self.pruner.masks,
                    "sparsity": sp,
                }, self.checkpoint_path)
        print(f"\nMigliore metrica: {best:.3f} | salvato in {self.checkpoint_path}")

# MAIN
def build_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument(
        "--dataset", "-d", 
        dest="dataset_name", 
        required=True,
        help="Dataset name")
    parser.add_argument(
        "--checkpoint", "-c",
        dest="checkpoint",
        required=True,
        help="Nome cartella da dove caricare i modelli base")
    parser.add_argument(
        "--output", "-o",
        dest="output",
        required=True,
        help="Dove salvare il modello prunato (default: stessa cartella)")
    parser.add_argument("--start_frac", type=float, default=0.0, help="Quando inizia il pruning (frazione della rampa)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--importance_batches", type=int, default=5)
    parser.add_argument("--prune_A_log", action="store_true", help="Includi A_log nel pruning ")
    parser.add_argument("--test", action="store_true", default=True, help="Testa il modello potato a fine pruning")
    return parser.parse_args()

def main():

    config = MODELS_CONFIG
    pruning_config = PRUNING_CONFIG
    perc = pruning_config["perc"]

    accuracies = {}

    args = build_args()

    os.makedirs(args.output, exist_ok=True)

    for i, p in enumerate(perc):

        print(f"===== PRUNING {100*p}% =====")

        path = os.path.join(f"./{args.output}", f"mamba_{args.dataset_name}_pruned_{int(p*100)}%.pth")
        if os.path.exists(path):
            print(f"[SKIP] esiste già: {path}")
            continue

        model_checkpoint = f"./{args.checkpoint}/mamba_{args.dataset_name}_best.pth"

        # Carico baseline (modello + dataset)
        model = ModelTest.from_pth(
                model_path=model_checkpoint,
                batch_size=config["mamba"][args.dataset_name]["batch_size"],
                valsplit=config["val_split"],
                num_workers=config["num_workers"],
                d_model=config["mamba"][args.dataset_name]["features"],
                d_state=config["mamba"][args.dataset_name]["d_state"],
                depth=config["mamba"][args.dataset_name]["depth"],
                dropout=config["mamba"][args.dataset_name]["dropout"],
                norm=config["mamba"][args.dataset_name]["norm"],
                pre_norm=config["mamba"][args.dataset_name]["pre-norm"]
            )
        model, dataset = model.model, model.dataset

        # Setup iperparametri di pruning
        lr = pruning_config["mamba"][dataset.name]["lr"]
        lr_min = lr * 0.01
        weight_decay = pruning_config["mamba"][dataset.name]["weight_decay"]
        epochs = pruning_config["mamba"][dataset.name]["finetune_epochs"] * (i+1)
        ramp_epochs = max(1, int(epochs * 0.4))
        print(f"Pruning parameters: lr={lr}, lr_min={lr_min}, weight_decay={weight_decay}, epochs={epochs}, ramp_epochs={ramp_epochs}, alpha={args.alpha}, start_frac={args.start_frac}, importance_batches={args.importance_batches}")
        
        job = PruningFineTuning(
            model=model, 
            dataset=dataset,
            epochs=epochs, 
            ramp_epochs=ramp_epochs, 
            s_final=p,
            lr=lr, 
            lr_min=lr_min, 
            weight_decay=weight_decay,
            checkpoint_path=path,
            alpha=args.alpha, 
            start_frac=args.start_frac,
            importance_batches=args.importance_batches,
            prune_A_log=args.prune_A_log,
        )
        job.run()

        if args.test:
            print("\n--- TEST modello potato ---")
            acc = ModelTest.from_pth(
                model_path=path,
                batch_size=config["mamba"][args.dataset_name]["batch_size"],
                valsplit=config["val_split"],
                num_workers=config["num_workers"],
                d_model=config["mamba"][args.dataset_name]["features"],
                d_state=config["mamba"][args.dataset_name]["d_state"],
                depth=config["mamba"][args.dataset_name]["depth"],
                dropout=config["mamba"][args.dataset_name]["dropout"],
                norm=config["mamba"][args.dataset_name]["norm"],
                pre_norm=config["mamba"][args.dataset_name]["pre-norm"]
            ).run()
            accuracies[p] = acc

    print(f"\n=== ACCURACY MIGLIORI PER OGNI STEP ===")
    for per, acc in accuracies.items():
        print(f"    - Pruning {int(per*100)}%: {acc:.2f}%.")

if __name__ == "__main__":
    main()