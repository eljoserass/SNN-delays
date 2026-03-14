#!/usr/bin/env python3
import argparse
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config
from datasets import SSC_dataloaders

try:
    import spikingjelly.datasets.shd as shd_module
except Exception as exc:
    print(f"ERROR: failed to import spikingjelly.datasets.shd: {exc}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight SSC diagnostics without launching full training."
    )
    parser.add_argument(
        "--probe-loader",
        action="store_true",
        help="Try constructing SSC dataloaders (may trigger download/preprocess).",
    )
    parser.add_argument(
        "--datasets-path",
        default="Datasets",
        help="Datasets root path (default: Datasets).",
    )
    return parser.parse_args()


def print_metadata():
    print(f"python={sys.version.split()[0]}")
    print(f"spikingjelly.datasets.shd={getattr(shd_module, '__file__', 'unknown')}")

    has_ssc = hasattr(shd_module, "SpikingSpeechCommands")
    print(f"has_spiking_speech_commands={has_ssc}")
    if not has_ssc:
        return

    cls = shd_module.SpikingSpeechCommands
    print(f"SpikingSpeechCommands.__init__={inspect.signature(cls.__init__)}")

    if hasattr(cls, "resource_url_md5"):
        try:
            resources = cls.resource_url_md5()
            print(f"resource_count={len(resources)}")
            for item in resources:
                print(f"resource={item[0]}|{item[1]}")
            shd_like = [r[0] for r in resources if str(r[0]).startswith("shd_")]
            if shd_like:
                print(f"WARNING: SSC resources contain SHD filenames: {shd_like}")
        except Exception as exc:
            print(f"resource_url_md5_error={exc}")


def probe_loader(datasets_path):
    cfg = Config()
    cfg.dataset = "ssc"
    cfg.datasets_path = datasets_path
    print(f"probe_loader_datasets_path={cfg.datasets_path}")
    try:
        SSC_dataloaders(cfg)
        print("probe_loader=OK")
    except Exception as exc:
        print(f"probe_loader_error={type(exc).__name__}: {exc}")


def main():
    args = parse_args()
    print_metadata()
    if args.probe_loader:
        probe_loader(args.datasets_path)


if __name__ == "__main__":
    main()
