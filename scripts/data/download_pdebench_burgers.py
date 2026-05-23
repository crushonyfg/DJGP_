from __future__ import annotations

import argparse
import csv
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


DEFAULT_METADATA = Path("..") / "external" / "PDEBench" / "pdebench" / "data_download" / "pdebench_data_urls.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one PDEBench Burgers HDF5 shard. Full Burgers is about 93GB; one shard is about 8GB."
    )
    parser.add_argument("--nu", default="0.01", help="Burgers viscosity shard, e.g. 0.01, 0.1, 1.0.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="PDEBench pdebench_data_urls.csv path.")
    parser.add_argument("--output-root", default="data/raw/pdebench", help="Root folder for downloaded raw PDEBench data.")
    parser.add_argument("--yes-large-download", action="store_true", help="Required to actually start the large download.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume an existing .part download.")
    parser.add_argument(
        "--insecure-skip-tls-verify",
        action="store_true",
        help="Last-resort workaround for local CA problems. Prefer fixing certifi/conda certificates.",
    )
    return parser.parse_args()


def find_burgers_row(metadata: Path, nu: str) -> dict[str, str]:
    expected = f"1D_Burgers_Sols_Nu{nu}.hdf5"
    with metadata.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["PDE"].lower() == "burgers" and row["Filename"] == expected:
                return row
    raise ValueError(f"Could not find Burgers shard {expected} in {metadata}")


def ssl_context(*, insecure_skip_tls_verify: bool = False) -> ssl.SSLContext:
    if insecure_skip_tls_verify:
        return ssl._create_unverified_context()
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _parse_total_bytes(response, existing_bytes: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        try:
            return int(content_range.rsplit("/", 1)[1])
        except ValueError:
            pass
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    try:
        return existing_bytes + int(content_length)
    except ValueError:
        return None


def _format_bytes(n: int | float) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def download_file(url: str, output_path: Path, *, context: ssl.SSLContext, resume: bool = True) -> None:
    part_path = output_path.with_name(output_path.name + ".part")
    existing = part_path.stat().st_size if resume and part_path.exists() else 0
    headers = {}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"Resuming from {_format_bytes(existing)}: {part_path}")

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, context=context, timeout=60)
    except urllib.error.URLError as exc:
        message = str(exc.reason)
        if "CERTIFICATE_VERIFY_FAILED" in message:
            raise SystemExit(
                "TLS certificate verification failed. Run this through the jumpGP env, e.g.\n"
                "  conda run -n jumpGP python scripts\\data\\download_pdebench_burgers.py --nu 0.01 --yes-large-download\n"
                "If your local Conda CA bundle is still broken, update certificates with:\n"
                "  conda update -n jumpGP certifi ca-certificates openssl\n"
                "Last resort on a trusted network only: add --insecure-skip-tls-verify"
            ) from exc
        raise

    with response:
        status = getattr(response, "status", None)
        if existing and status != 206:
            print("Server did not honor Range resume; restarting partial download.")
            existing = 0
            mode = "wb"
        total = _parse_total_bytes(response, existing)
        downloaded = existing
        last_report = time.perf_counter()
        with part_path.open(mode) as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.perf_counter()
                if now - last_report >= 2:
                    if total:
                        pct = 100.0 * downloaded / total
                        print(f"\rDownloaded {_format_bytes(downloaded)} / {_format_bytes(total)} ({pct:.1f}%)", end="")
                    else:
                        print(f"\rDownloaded {_format_bytes(downloaded)}", end="")
                    last_report = now
    print()
    part_path.replace(output_path)


def main() -> None:
    args = parse_args()
    if not args.yes_large_download:
        raise SystemExit(
            "Refusing to start a multi-GB download without --yes-large-download. "
            "Use --nu to choose a single shard, not the full 93GB Burgers set."
        )

    metadata = Path(args.metadata)
    row = find_burgers_row(metadata, args.nu)
    output_dir = Path(args.output_root) / row["Path"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / row["Filename"]
    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    if "jumpGP" not in sys.prefix:
        print(
            "Warning: current Python does not look like the jumpGP conda env. "
            "Recommended command: conda run -n jumpGP python scripts\\data\\download_pdebench_burgers.py ..."
        )

    print(f"Downloading {row['Filename']} to {output_path}")
    download_file(
        row["URL"],
        output_path,
        context=ssl_context(insecure_skip_tls_verify=args.insecure_skip_tls_verify),
        resume=not args.no_resume,
    )
    print(f"Downloaded: {output_path}")


if __name__ == "__main__":
    main()
