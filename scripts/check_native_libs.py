"""Audit the native libraries backing numpy / torch / warp, on linux and mac.

Two jobs:

1. Exercise a small CPU-only workload through all three stacks, so the check
   runs anywhere -- including GPU-less CI runners.
2. Report which shared libraries actually got loaded, classified by origin:

     env       shipped by conda, from this prefix's lib/  (e.g. libopenblas)
     vendored  bundled inside a pip wheel, under site-packages/
     system    provided by the OS (Accelerate on macOS, system libs on linux)

That classification answers a question the lockfiles can't: whether a conda
BLAS/OpenMP package is doing any work, or is just dead weight because the pip
wheels carry (or bypass) their own.

Duplicate OpenMP runtimes are reported as warnings, since a mixed sklearn +
torch process routinely loads more than one and works fine; pass --strict to
turn them into failures. Compute errors always fail.

Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys

FAILURES: list[str] = []
WARNINGS: list[str] = []


def loaded_libraries() -> list[str]:
    """Absolute paths of every shared library mapped into this process."""
    if platform.system() == "Darwin":
        import ctypes

        libc = ctypes.CDLL(None)
        libc._dyld_image_count.restype = ctypes.c_uint32
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        return [
            libc._dyld_get_image_name(i).decode()
            for i in range(libc._dyld_image_count())
        ]

    paths: set[str] = set()
    with open("/proc/self/maps") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 6 and parts[-1].startswith("/"):
                paths.add(parts[-1])
    return sorted(paths)


def origin(path: str) -> str:
    """Classify a library by who supplied it."""
    if "/site-packages/" in path:
        return "vendored"
    if path.startswith(sys.prefix + os.sep):
        return "env"
    return "system"


def run_cpu_workload() -> None:
    """Small CPU-only computation through numpy, torch and warp."""
    import numpy as np

    print(f"numpy   {np.__version__}")

    rng = np.random.default_rng(0)
    a = rng.standard_normal((256, 256), dtype=np.float32)
    prod = a @ a
    if not np.all(np.isfinite(prod)):
        FAILURES.append("numpy matmul produced non-finite values")
    else:
        print(f"numpy   256x256 matmul ok (trace={float(prod.trace()):.4f})")

    import torch

    print(f"torch   {torch.__version__}")

    t = torch.from_numpy(a)
    got = (t @ t).numpy()
    if not np.allclose(got, prod, rtol=1e-4, atol=1e-4):
        FAILURES.append("torch cpu matmul disagrees with numpy")
    else:
        print("torch   cpu matmul agrees with numpy")

    import warp as wp

    wp.init()
    print(f"warp    {wp.config.version}")

    @wp.kernel
    def scale(x: wp.array(dtype=float), k: float):
        i = wp.tid()
        x[i] = x[i] * k

    n = 4096
    arr = wp.array(np.ones(n, dtype=np.float32), dtype=float, device="cpu")
    wp.launch(scale, dim=n, inputs=[arr, 3.0], device="cpu")
    wp.synchronize_device("cpu")
    if not np.allclose(arr.numpy(), 3.0):
        FAILURES.append("warp cpu kernel produced wrong values")
    else:
        print("warp    cpu kernel ok")

    # Zero-copy interop: warp writing through a view of a torch tensor must be
    # visible in the tensor. Exercises the warp/torch ABI boundary.
    tt = torch.full((1024,), 2.0)
    wp.launch(scale, dim=tt.numel(),
              inputs=[wp.from_torch(tt), 5.0],
              device=wp.device_from_torch(tt.device))
    wp.synchronize()
    if not torch.allclose(tt, torch.full_like(tt, 10.0)):
        FAILURES.append(f"warp/torch zero-copy interop wrong: {tt[:4].tolist()}")
    else:
        print("interop warp<->torch zero-copy ok")


def report_backends() -> None:
    """What BLAS numpy and scipy were actually built against."""
    import numpy as np

    try:
        cfg = np.show_config("dicts") or {}
        name = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "?")
        print(f"numpy   blas backend: {name}")
    except Exception as e:  # show_config's shape is not API-stable
        WARNINGS.append(f"could not read numpy blas backend: {e}")

    try:
        import scipy

        cfg = scipy.__config__.show("dicts") or {}
        name = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "?")
        print(f"scipy   blas backend: {name}")
    except Exception as e:
        WARNINGS.append(f"could not read scipy blas backend: {e}")


def audit_libraries(strict: bool) -> None:
    libs = loaded_libraries()
    print(f"\nshared libraries loaded: {len(libs)}")

    def matching(*keys: str) -> list[str]:
        return sorted(
            p for p in libs
            if any(k in os.path.basename(p).lower() for k in keys)
        )

    # Dedupe by real path, not basename -- several distinct copies of
    # libomp.dylib can be loaded at once and they are NOT interchangeable.
    omp = {os.path.realpath(p) for p in matching("libomp", "libiomp", "libgomp")}
    blas = matching("openblas", "libblas", "liblapack", "mkl",
                    "accelerate", "veclib", "flexiblas")

    print("\nBLAS / LAPACK:")
    for p in blas or ["  (none loaded)"]:
        print(f"  [{origin(p):8}] {p}" if p in blas else p)

    print("\nOpenMP runtimes:")
    for p in sorted(omp) or ["  (none loaded)"]:
        print(f"  [{origin(p):8}] {p}" if p in omp else p)

    from_env = [p for p in blas if origin(p) == "env"]
    if not from_env:
        print("\nnote: no conda-supplied BLAS was loaded; the pip wheels use "
              "their own (vendored or system) BLAS.")
    else:
        print(f"\nnote: {len(from_env)} conda-supplied BLAS library(ies) in use.")

    if len(omp) > 1:
        msg = (f"{len(omp)} distinct OpenMP runtimes loaded: "
               f"{sorted(os.path.basename(p) for p in omp)}")
        (FAILURES if strict else WARNINGS).append(msg)

    if os.environ.get("KMP_DUPLICATE_LIB_OK"):
        WARNINGS.append("KMP_DUPLICATE_LIB_OK is set; a duplicate-OpenMP "
                        "crash could be masked")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Treat duplicate OpenMP runtimes as a failure.")
    args = parser.parse_args()

    print(f"system  {platform.system()} ({platform.machine()})")
    print(f"prefix  {sys.prefix}\n")

    try:
        run_cpu_workload()
    except Exception as e:
        FAILURES.append(f"cpu workload raised {type(e).__name__}: {e}")

    print()
    report_backends()
    audit_libraries(args.strict)

    print()
    for w in WARNINGS:
        print(f"warn: {w}")

    if FAILURES:
        print("\nFAIL")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
