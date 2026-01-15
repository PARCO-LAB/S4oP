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

        model_size = self.model_size_MB(model)

        self.data[name] = {
            "peak_allocated_MB": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_MB": torch.cuda.max_memory_reserved() / 1024**2,
            "model_size_MB": model_size,
        }

    def model_size_MB(self, model):
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
        for b in model.buffers():
            total_bytes += b.numel() * b.element_size()
        return total_bytes / 1024**2


    def info(self, name=None):
        print("=========== MemoryProfile ===========")
        data_names = self.data if name is None else [name]
        for n in data_names:
            print(f"[[ {n} ]]")
            for k, v in self.data[n].items():
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
                    "Unit": "MB",
                    "Value": v
                })
        return pd.DataFrame(rows)
