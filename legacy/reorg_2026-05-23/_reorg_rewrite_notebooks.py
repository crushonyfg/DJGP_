"""Rewrite legacy imports inside .ipynb code cells using nbformat-safe JSON edits."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RENAMES = [
    ("new_highdata_gen_utils", "data_gen.highdata_utils"),
    ("new_highdata_gen", "data_gen.highdata"),
    ("check_VI_utils_gpu_acc_UZ_qumodified_cor", "djgp.variational"),
    ("DJGP_test_validation", "shared.djgp_validation"),
    ("UCIdataset_autoencoder", "data_gen.uci_autoencoder"),
    ("calculate_bias_and_variance", "djgp.acquisition_metrics"),
    ("active_djgp_acquisition", "djgp.active_learning"),
    ("LHDataset_AE", "data_gen.lh_autoencoder"),
    ("new_minibatch_try", "djgp.minibatch"),
    ("dataset_analysis", "data_gen.analysis"),
    ("maxmin_design", "shared.maxmin_design"),
    ("data_generate", "data_gen.synthetic"),
    ("DeepGP_test", "shared.deepgp"),
    ("JumpGP_test", "shared.jumpgp_runner"),
    ("DJGP_test", "shared.djgp_runner"),
    ("utils1", "shared.utils1"),
    ("utils", "shared.utils"),
    ("AE", "data_gen.autoencoder"),
]

PATTERNS = []
for old, new in RENAMES:
    p_from = re.compile(rf"(?m)^(?P<lead>\s*from\s+){re.escape(old)}(?P<tail>(\s|\.))")
    p_import = re.compile(rf"(?m)^(?P<lead>\s*import\s+){re.escape(old)}(?P<tail>(\s|,|$))")
    PATTERNS.append((old, new, p_from, p_import))


def rewrite_source(text: str) -> tuple[str, int]:
    n = 0
    for old, new, p_from, p_import in PATTERNS:
        text, c1 = p_from.subn(lambda m, n=new: f"{m.group('lead')}{n}{m.group('tail')}", text)
        text, c2 = p_import.subn(lambda m, o=old, n=new: f"{m.group('lead')}{n} as {o}{m.group('tail')}", text)
        n += c1 + c2
    return text, n


def process_ipynb(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    total = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            joined = "".join(src)
            new_joined, n = rewrite_source(joined)
            if n > 0:
                # Re-split into list, preserving newlines as in nbformat convention
                lines = new_joined.splitlines(keepends=True)
                cell["source"] = lines
                total += n
        elif isinstance(src, str):
            new_src, n = rewrite_source(src)
            if n > 0:
                cell["source"] = new_src
                total += n
    if total > 0:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return total


def main() -> None:
    total_changes = 0
    changed = []
    for nb in ROOT.rglob("*.ipynb"):
        rel = nb.relative_to(ROOT)
        if any(p in {".ipynb_checkpoints"} for p in rel.parts):
            continue
        n = process_ipynb(nb)
        if n > 0:
            changed.append((rel.as_posix(), n))
            total_changes += n
    for f, n in sorted(changed):
        print(f"{n:3d}  {f}")
    print(f"\n{len(changed)} notebooks changed, {total_changes} replacements")


if __name__ == "__main__":
    main()
