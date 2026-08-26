import torch
import pandas as pd

class MemoryProfile:
    def __init__(self):
        self.data = {}

    def add(self, name):
        self.data[name] = {}

    @torch.no_grad()
    def run(self, name, model, get_example_input):
        MB = 1024 ** 2
        model.eval()
        example_input = get_example_input()

        # warm-up
        for _ in range(5):
            _ = model(example_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            _ = model(example_input)
            torch.cuda.synchronize()
            peak = {
                "peak_allocated_MB": torch.cuda.max_memory_allocated() / MB,
                "peak_reserved_MB": torch.cuda.max_memory_reserved() / MB,
            }
        else:
            _ = model(example_input)
            peak = {"peak_allocated_MB": float("nan"), "peak_reserved_MB": float("nan")}

        self.data[name] = {**peak, **self.model_size_MB(model)}

    def model_size_MB(self, model):
        MB = 1024 ** 2

        p_bytes_all = p_bytes_noembed = 0
        n_params_all = n_params_noembed = 0
        for name, p in model.named_parameters():
            b = p.numel() * p.element_size()
            p_bytes_all += b
            n_params_all += p.numel()
            if "embedding" not in name:
                p_bytes_noembed += b
                n_params_noembed += p.numel()

        buf_bytes = n_buf = 0
        for buf in model.buffers():
            buf_bytes += buf.numel() * buf.element_size()
            n_buf += buf.numel()

        return {
            # --- SOLO PARAMETRI (no buffer) ---
            "num_params_with_embedding": n_params_all,
            "num_params_no_embedding": n_params_noembed,

            # --- PARAMETRI + BUFFER (per la dimensione reale in memoria) ---
            "num_total_with_embedding": n_params_all + n_buf,
            "num_total_no_embedding": n_params_noembed + n_buf,
            "size_with_embedding_MB": (p_bytes_all + buf_bytes) / MB,
            "size_no_embedding_MB": (p_bytes_noembed + buf_bytes) / MB,

            # --- DETTAGLIO BUFFER ---
            "num_buffers": n_buf,
            "buffers_MB": buf_bytes / MB,
        }


    def info(self, name=None):
        print("=========== MemoryProfile ===========")
        data_names = self.data if name is None else [name]
        for n in data_names:
            print(f"[[ {n} ]]")
            for k, v in self.data[n].items():
                if k in ["num_params_with_embedding", "num_params_no_embedding", "num_total_with_embedding", "num_total_no_embedding", "num_buffers"]:
                    print(f"{k}: {v:,} params")
                else:
                    print(f"{k}: {v:.2f} MB")
        print("====================================")

    def dataframe(self):
        rows = []
        for name in self.data:
            for k, v in self.data[name].items():
                rows.append({
                    "Test": name,
                    "Section": "Memory",
                    "Metric": k,
                    "Unit": "params" if k == "model_params" or k == "model_total_params" else "MB",
                    "Value": v
                })
        return pd.DataFrame(rows)
