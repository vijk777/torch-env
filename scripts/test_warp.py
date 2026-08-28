"""Smoke test for NVIDIA Warp in the torch-env conda env.

Verifies:
  - warp imports and initializes
  - a compiled kernel runs on CPU and matches numpy
  - the same kernel runs on CUDA, when a CUDA device is present
  - zero-copy torch interop (wp.from_torch / wp.to_torch) round-trips

Warp's CUDA backend is linux-only; on macOS it initializes CPU-only and the
CUDA checks are skipped rather than failed.

Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp


@wp.kernel
def saxpy(a: float, x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = a * x[i] + y[i]


def run_on(device: str, n: int = 1024) -> None:
    """Run saxpy on `device` and check it against the numpy result."""
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(n, dtype=np.float32)
    y_np = rng.standard_normal(n, dtype=np.float32)

    x = wp.array(x_np, dtype=float, device=device)
    y = wp.array(y_np, dtype=float, device=device)

    wp.launch(saxpy, dim=n, inputs=[2.0, x, y], device=device)
    wp.synchronize_device(device)

    np.testing.assert_allclose(y.numpy(), 2.0 * x_np + y_np, rtol=1e-5)
    print(f"kernel  saxpy ok on {device}")


def main() -> int:
    failures: list[str] = []

    wp.init()
    print(f"warp    {wp.config.version}")
    print(f"devices {[str(d) for d in wp.get_devices()]}")

    try:
        run_on("cpu")
    except Exception as e:
        failures.append(f"cpu kernel failed: {e}")

    if wp.is_cuda_available():
        try:
            run_on("cuda:0")
        except Exception as e:
            failures.append(f"cuda kernel failed: {e}")
    else:
        print("cuda    no CUDA device (expected on macOS); skipping")

    # Zero-copy interop: a warp kernel writing through a view of a torch
    # tensor must be visible in the torch tensor itself.
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        t = torch.ones(256, dtype=torch.float32, device=device)
        wp.launch(saxpy, dim=t.numel(),
                  inputs=[3.0, wp.from_torch(t), wp.from_torch(t)],
                  device=wp.device_from_torch(t.device))
        wp.synchronize()

        expected = 4.0  # 3*1 + 1, written in place
        if not torch.allclose(t, torch.full_like(t, expected)):
            failures.append(f"torch interop: expected all {expected}, got {t[:4].tolist()}")
        else:
            print(f"interop warp<->torch zero-copy ok on {device}")
    except Exception as e:
        failures.append(f"torch interop failed: {e}")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
