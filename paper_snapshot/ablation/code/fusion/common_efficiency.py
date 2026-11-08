"""Small resource/timing helpers for later GPU benchmark runners."""
from __future__ import annotations

import time
import torch


def synchronized_timer(device: torch.device):
    class Timer:
        def __enter__(self):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            self.started = time.perf_counter()
            return self

        def __exit__(self, *_):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.seconds = time.perf_counter() - self.started
    return Timer()


def environment_record(device: torch.device, precision: str = "fp32") -> dict:
    record = {"device": str(device), "pytorch_version": torch.__version__,
              "cuda_version": torch.version.cuda, "precision_mode": precision}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        record.update(gpu_model=props.name, gpu_vram_bytes=props.total_memory,
                      peak_gpu_memory_mb=torch.cuda.max_memory_allocated(device) / 2**20)
    return record


def parameter_count(model: torch.nn.Module) -> dict:
    return {"TOTAL_PARAMS": int(sum(p.numel() for p in model.parameters()))}
