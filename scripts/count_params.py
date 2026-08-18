#!/usr/bin/env python3
"""Print closed-form Qwen3.8-35B-A3B parameter counts (no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen3_8_moe.params import parameter_report


def main() -> None:
    report = parameter_report()
    billions = {key: (value / 1e9 if isinstance(value, int) and value > 1000 else value) for key, value in report.items()}
    print("Qwen3.8-35B-A3B parameter report")
    print(json.dumps({key: report[key] for key in report}, indent=2))
    print()
    print(f"Total:  {report['total'] / 1e9:.3f}B")
    print(f"Active: {report['active'] / 1e9:.3f}B  (8 routed + 1 shared experts / token)")
    print(f"Text:   {report['text_total'] / 1e9:.3f}B")
    print(f"Vision: {report['vision_total'] / 1e9:.3f}B")
    _ = billions


if __name__ == "__main__":
    main()
