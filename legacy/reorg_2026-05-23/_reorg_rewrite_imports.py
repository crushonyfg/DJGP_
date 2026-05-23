"""One-shot import rewriter for the 2026-05-23 repo reorganization.

Walks all .py files outside excluded dirs and rewrites legacy top-level
module imports to the new src/ layout. Order matters: longer module names
first so substring matches don't shadow longer ones.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (old_module, new_module) — order: longer/more-specific first
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

EXCLUDE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "build", "dist",
                "djgp_research.egg-info", "node_modules", ".venv", "venv"}

# Build one big set of compiled patterns. Each entry rewrites
# `from OLD ` / `from OLD\n` / `from OLD\t` and `import OLD` (with word boundary
# but also not followed by `.something` because that means a submodule).
PATTERNS = []
for old, new in RENAMES:
    # `from OLD<word_boundary, NOT a dot>`
    p_from = re.compile(rf"(?m)^(?P<lead>\s*from\s+){re.escape(old)}(?P<tail>(\s|\.))")
    # `import OLD<word_boundary, NOT a dot>`  e.g. `import utils1` or `import utils1, foo`
    p_import = re.compile(rf"(?m)^(?P<lead>\s*import\s+){re.escape(old)}(?P<tail>(\s|,|$))")
    PATTERNS.append((old, new, p_from, p_import))


def rewrite(text: str) -> tuple[str, int]:
    n_changes = 0
    for old, new, p_from, p_import in PATTERNS:
        def sub_from(m: re.Match) -> str:
            return f"{m.group('lead')}{new}{m.group('tail')}"

        def sub_import(m: re.Match) -> str:
            # `import shared.utils1 as utils1` so downstream references stay valid
            tail = m.group("tail")
            return f"{m.group('lead')}{new} as {old}{tail}"

        new_text, c1 = p_from.subn(sub_from, text)
        new_text, c2 = p_import.subn(sub_import, new_text)
        text = new_text
        n_changes += c1 + c2
    return text, n_changes


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.name == Path(__file__).name:
        return True
    return False


def main() -> None:
    total_files = 0
    total_changes = 0
    changed_files = []
    for py in ROOT.rglob("*.py"):
        if should_skip(py.relative_to(ROOT)):
            continue
        total_files += 1
        original = py.read_text(encoding="utf-8")
        new, n = rewrite(original)
        if n > 0:
            py.write_text(new, encoding="utf-8")
            total_changes += n
            changed_files.append((py.relative_to(ROOT).as_posix(), n))
    changed_files.sort()
    for f, n in changed_files:
        print(f"{n:3d}  {f}")
    print(f"\n{len(changed_files)} files changed / {total_files} scanned, {total_changes} replacements")


if __name__ == "__main__":
    main()
