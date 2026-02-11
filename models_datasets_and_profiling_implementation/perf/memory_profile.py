import torch
import pandas as pd

class MemoryProfile:
    def __init__(self):
        self.data = {}

    def add(self, name):
        self.data[name] = {}

    @torch.no_grad()
    def run(self, name, model, get_example_input):
        model.eval()
        example_input = get_example_input()

        # warm-up
        for _ in range(5):
            _ = model(example_input)
            
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        _ = model(example_input)
        torch.cuda.synchronize()

        params_size, model_params, model_size, model_total_params = self.model_size_MB(model)

        self.data[name] = {
            "peak_allocated_MB": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_MB": torch.cuda.max_memory_reserved() / 1024**2,
            "model_size_params_MB": params_size,
            "model_params": model_params,
            "model_size_total_MB": model_size,
            "model_total_params": model_total_params
        }

    def model_size_MB(self, model):
        total_bytes_params = 0
        total_bytes_buffers = 0
        total_params = 0
        total_buffers = 0
        for name, p in model.named_parameters():
            if name != "embedding.weight" or model.dataset_name != "imdb":
                total_bytes_params += p.numel() * p.element_size()
                total_params += p.numel()
        for b in model.buffers():
            total_bytes_buffers += b.numel() * b.element_size()
            total_buffers += b.numel()
        return total_bytes_params / 1024**2, total_params, (total_bytes_params + total_bytes_buffers) / 1024**2, total_params + total_buffers


    def info(self, name=None):
        print("=========== MemoryProfile ===========")
        data_names = self.data if name is None else [name]
        for n in data_names:
            print(f"[[ {n} ]]")
            for k, v in self.data[n].items():
                if k == "model_params" or k == "model_total_params":
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
