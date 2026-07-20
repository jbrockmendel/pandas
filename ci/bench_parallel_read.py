"""Benchmark parallel read_csv across thread counts; write results as JSON.

Run once per (platform, pandas build).  Fixtures are generated from a fixed
seed without using pandas, so every job reads byte-identical files (the md5 of
each fixture goes into the output to prove it).

Usage: python bench_parallel_read.py <data_dir> <out.json> [label]
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

SEED = 20260720
REPEATS = 7
THREAD_SETTINGS: list = [1, 2, 4, "default"]


# ---------------------------------------------------------------------------
# Fixture generation (no pandas: both refs must read identical bytes)
# ---------------------------------------------------------------------------


def _savetxt(path, arr, fmt, header):
    with open(path, "wb") as fh:
        np.savetxt(
            fh, arr, fmt=fmt, delimiter=",", header=header, comments="", newline="\n"
        )


def gen_int(path, n_rows=900_000, n_cols=10):
    rng = np.random.default_rng(SEED)
    arr = rng.integers(0, 1_000_000, size=(n_rows, n_cols))
    _savetxt(path, arr, "%d", ",".join(f"c{i}" for i in range(n_cols)))


def gen_float(path, n_rows=450_000, n_cols=10):
    rng = np.random.default_rng(SEED + 1)
    arr = rng.random((n_rows, n_cols)) * 1000
    _savetxt(path, arr, "%.12g", ",".join(f"c{i}" for i in range(n_cols)))


def gen_str(path, n_rows=400_000, n_cols=10):
    rng = np.random.default_rng(SEED + 2)
    pool = np.array([f"str_value_{i:06d}" for i in range(1000)])
    arr = pool[rng.integers(0, 1000, size=(n_rows, n_cols))]
    _savetxt(path, arr, "%s", ",".join(f"c{i}" for i in range(n_cols)))


def gen_wide(path, n_rows=11_000, n_cols=800):
    rng = np.random.default_rng(SEED + 3)
    arr = rng.integers(0, 1_000_000, size=(n_rows, n_cols))
    _savetxt(path, arr, "%d", ",".join(f"c{i}" for i in range(n_cols)))


def gen_mixed(path, n_rows=750_000):
    rng = np.random.default_rng(SEED + 4)
    ids = np.arange(n_rows)
    flt1 = rng.random(n_rows) * 1e6
    flt2 = rng.random(n_rows)
    flt3 = rng.random(n_rows) * 1e-3
    int1 = rng.integers(0, 10**9, n_rows)
    int2 = rng.integers(0, 100, n_rows)
    int3 = rng.integers(-1000, 1000, n_rows)
    cats = np.array(["alpha", "beta", "gamma", "delta"])[rng.integers(0, 4, n_rows)]
    notes = np.array([f"note_{i:05d}" for i in range(500)])[
        rng.integers(0, 500, n_rows)
    ]
    flags = np.where(rng.integers(0, 2, n_rows).astype(bool), "True", "False")

    with open(path, "wb") as fh:
        fh.write(b"id,f1,f2,f3,i1,i2,cat,note,flag,i3\n")
        buf: list[str] = []
        for idx in range(n_rows):
            buf.append(
                f"{ids[idx]},{flt1[idx]:.10g},{flt2[idx]:.10g},{flt3[idx]:.10g},"
                f"{int1[idx]},{int2[idx]},{cats[idx]},{notes[idx]},"
                f"{flags[idx]},{int3[idx]}"
            )
            if len(buf) >= 50_000:
                fh.write(("\n".join(buf) + "\n").encode())
                buf = []
        if buf:
            fh.write(("\n".join(buf) + "\n").encode())


FIXTURES = {
    "int10": gen_int,
    "float10": gen_float,
    "str10": gen_str,
    "mixed10": gen_mixed,
    "wide800": gen_wide,
    # ~12 MB: below the baseline's 50 MB parallel floor, above the branch's
    # 5 MB one, so only the branch parallelises it.  Every other fixture is
    # sized clear of 50 MB so both refs go parallel and compare like for like.
    "mixed_small": lambda path: gen_mixed(path, n_rows=150_000),
}


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def read_once(pd, path, threads):
    gc.collect()
    if threads == "default":
        start = time.perf_counter()
        frame = pd.read_csv(path)
        elapsed = time.perf_counter() - start
    else:
        with pd.option_context("mode.max_threads", threads):
            start = time.perf_counter()
            frame = pd.read_csv(path)
            elapsed = time.perf_counter() - start
    del frame
    return elapsed * 1000


def cpu_model():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        return platform.processor()
    except OSError:
        return "unknown"


def main():
    data_dir, out_path = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    os.makedirs(data_dir, exist_ok=True)

    import pandas as pd
    import pandas._testing as tm

    report: dict = {
        "label": label,
        "pandas_version": pd.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "repeats": REPEATS,
        "fixtures": {},
    }

    for name, generator in FIXTURES.items():
        path = os.path.join(data_dir, name + ".csv")
        if not os.path.exists(path):
            generator(path)
        entry: dict = {
            "size_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
            "md5": md5(path),
        }
        print(f"[{name}] {entry['size_mb']}MB md5={entry['md5']}", flush=True)

        # Correctness: the parallel result must equal the serial one.
        try:
            with pd.option_context("mode.max_threads", 1):
                serial_frame = pd.read_csv(path)
            with pd.option_context("mode.max_threads", 4):
                parallel_frame = pd.read_csv(path)
            tm.assert_frame_equal(serial_frame, parallel_frame)
            entry["parallel_equals_serial"] = True
            del serial_frame, parallel_frame
        except AssertionError as err:
            entry["parallel_equals_serial"] = False
            entry["mismatch"] = str(err)[:2000]
        gc.collect()

        # Interleave thread settings across rounds so drift hits them equally.
        samples: dict = {str(setting): [] for setting in THREAD_SETTINGS}
        for _ in range(REPEATS):
            for setting in THREAD_SETTINGS:
                samples[str(setting)].append(read_once(pd, path, setting))
        entry["min_ms"] = {key: round(min(vals), 1) for key, vals in samples.items()}
        entry["median_ms"] = {
            key: round(sorted(vals)[len(vals) // 2], 1) for key, vals in samples.items()
        }
        serial_min = entry["min_ms"]["1"]
        entry["speedup_vs_1t"] = {
            key: round(serial_min / val, 2) for key, val in entry["min_ms"].items()
        }
        report["fixtures"][name] = entry
        print(f"    min_ms={entry['min_ms']}", flush=True)
        print(f"    speedup={entry['speedup_vs_1t']}", flush=True)
        pathlib.Path(path).unlink()  # keep runner disk usage bounded

    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
