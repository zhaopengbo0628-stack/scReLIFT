"""Stage accounting for the PBMC S-mode efficiency protocol V2."""
import hashlib
import json
import time
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[1]/'docs/PBMC_SMODE_SCORING_TIMING_V2.json'


def begin(device, reset=True):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)
        if reset:
            torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def end(started, device):
    gpu = str(device).startswith('cuda')
    if gpu:
        import torch
        torch.cuda.synchronize(device)
    return dict(seconds=time.perf_counter()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if gpu else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if gpu else None)


def record(training, fitting, inference, trainable, resident, batch, device):
    return dict(protocol_id=json.loads(PROTOCOL.read_text())['protocol_id'],
                protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                training=training, source_fitting=fitting, inference=inference,
                trainable_parameters=trainable, inference_resident_parameters=resident,
                batch_size=batch, device=str(device), end_to_end_seconds=None,
                end_to_end_status='Requires an audited fresh Source workflow and all requested targets')
