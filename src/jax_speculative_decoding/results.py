from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    name: str
    model: str
    output_tokens: int
    elapsed_s: float
    tokens_per_second: float
    batch_size: int = 1
    prompt_tokens: int = 0
    draft_model: str | None = None
    k: int | None = None
    compile_s: float | None = None
    acceptance_rate: float | None = None
    proposed_tokens: int | None = None
    accepted_draft_tokens: int | None = None
    pcie_transfer_s: float | None = None
    draft_compute_s: float | None = None
    target_verify_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def print_result(result: BenchmarkResult) -> None:
    payload = result.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_result(path: str | Path | None, result: BenchmarkResult | list[BenchmarkResult]) -> None:
    if path is None:
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = result if isinstance(result, list) else [result]
    payload = [row.to_dict() for row in rows]

    if out.suffix.lower() == ".csv":
        fieldnames = sorted({key for row in payload for key in row})
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(payload)
        return

    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload[0] if len(payload) == 1 else payload, handle, indent=2, sort_keys=True)
