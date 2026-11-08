from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_dir: Path


def load_specs(path: Path) -> list[DatasetSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = [DatasetSpec(str(row["name"]), Path(row["data_dir"])) for row in payload["datasets"]]
    if len(specs) < 2:
        raise ValueError("V5-D requires at least two paired source datasets")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("Dataset names must be unique")
    return specs
