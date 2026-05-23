"""Compatibility entrypoint for the UCI benchmark experiment.

The implementation lives in `experiments/uci/new_dataset.py`. Importing this
shim is intentionally cheap and does not start the full UCI experiment.
"""

from __future__ import annotations

from experiments.uci.new_dataset import main as _main


def main():
    return _main()


if __name__ == "__main__":
    main()
