"""Smoke test for the torch-env conda env.

Auto-detects the platform and verifies:
  - torch, numpy, scipy import cleanly
  - the expected accelerator is available (CUDA on linux, MPS on mac)

With --ci, the device-available checks are relaxed to build-time checks
(torch.version.cuda set on linux, torch.backends.mps.is_built() on mac), so
the script can run on GPU-less CI runners.

Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import platform
import sys


def main(ci: bool) -> int:
    failures: list[str] = []

    try:
        import numpy as np
        print(f"numpy   {np.__version__}")
    except Exception as e:
        failures.append(f"numpy import failed: {e}")

    try:
        import scipy
        print(f"scipy   {scipy.__version__}")
    except Exception as e:
        failures.append(f"scipy import failed: {e}")

    try:
        import torch
        print(f"torch   {torch.__version__}")
    except Exception as e:
        failures.append(f"torch import failed: {e}")
        return _report(failures)

    system = platform.system()
    machine = platform.machine()
    print(f"system  {system} ({machine})  ci={ci}")

    if system == "Linux":
        if torch.version.cuda is None:
            failures.append("torch was not built with CUDA (torch.version.cuda is None)")
        else:
            print(f"cuda    torch built with CUDA {torch.version.cuda}")
            if ci:
                pass  # no GPU on CI; build-check is enough
            elif not torch.cuda.is_available():
                failures.append("CUDA not available at runtime")
            else:
                print(f"cuda    devices={torch.cuda.device_count()}, "
                      f"name={torch.cuda.get_device_name(0)}")
                x = torch.randn(8, 8, device="cuda")
                (x @ x).sum().item()
                print("cuda    matmul ok")

    elif system == "Darwin":
        if not torch.backends.mps.is_built():
            failures.append("torch was not built with MPS")
        else:
            print("mps     torch built with MPS")
            if ci:
                pass
            elif not torch.backends.mps.is_available():
                failures.append("MPS not available at runtime")
            else:
                x = torch.randn(8, 8, device="mps")
                (x @ x).sum().item()
                print("mps     matmul ok")

    else:
        failures.append(f"unsupported platform: {system}")

    return _report(failures)


def _report(failures: list[str]) -> int:
    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true",
                        help="Relax device-availability checks to build-only checks.")
    args = parser.parse_args()
    sys.exit(main(ci=args.ci))
