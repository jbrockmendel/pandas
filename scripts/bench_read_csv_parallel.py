"""Benchmark parallel ``read_csv`` thread-scaling, aimed at the Windows story.

Parallel CSV reading (GH#64347) is disabled by default on Windows because the
worker threads did not speed up there (and regressed at two threads).  This
script re-measures the thread-scaling curve so we can (a) confirm the inversion
reproduces on a given machine and (b) tell whether a code change removes it.

Usage
-----
Orchestrator (default) -- generate fixtures and sweep threads, one fresh
subprocess per (fixture, thread-count) config::

    python scripts/bench_read_csv_parallel.py --out results.json

Single config (invoked by the orchestrator; also usable by hand)::

    python scripts/bench_read_csv_parallel.py --single <csv> <n_threads> <reps>

Thread count is forced via the ``mode.max_threads`` option, which overrides the
per-platform default (including the Windows serial gate), so ``n_threads=1``
gives the serial baseline and ``n_threads>=2`` forces the parallel path (the
fixtures are sized above ``_PARALLEL_READ_MIN_BYTES`` so the size floor is a
no-op).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# Fixture generation
# --------------------------------------------------------------------------
# (name, n_rows, n_cols) -- row counts chosen to land each fixture comfortably
# above the 50 MB parallel-read floor (actual sizes are reported at run time).
FIXTURE_SPECS = {
    "ints": (2_000_000, 10),
    "floats": (2_000_000, 10),
    "strings": (1_500_000, 10),
    "mixed": (2_000_000, 10),
}


def _build_fixture(name: str, path: str, rows_override: int | None = None) -> None:
    import numpy as np

    import pandas as pd

    n_rows, n_cols = FIXTURE_SPECS[name]
    if rows_override:
        n_rows = rows_override
    rng = np.random.default_rng(0)

    if name == "ints":
        data = {
            f"c{i}": rng.integers(-(10**8), 10**8, size=n_rows, dtype="int64")
            for i in range(n_cols)
        }
    elif name == "floats":
        data = {f"c{i}": rng.standard_normal(n_rows) * 1e6 for i in range(n_cols)}
    elif name == "strings":
        # 8-char ASCII words; a small pool keeps generation cheap and mimics
        # low-cardinality categorical-ish text.
        pool = np.array(
            [
                "".join(chr(97 + (k >> (3 * j)) % 26) for j in range(8))
                for k in range(4096)
            ]
        )
        data = {
            f"c{i}": pool[rng.integers(0, len(pool), size=n_rows)]
            for i in range(n_cols)
        }
    elif name == "mixed":
        pool = np.array(
            [
                "".join(chr(97 + (k >> (3 * j)) % 26) for j in range(6))
                for k in range(2048)
            ]
        )
        data = {}
        for i in range(n_cols):
            kind = i % 3
            if kind == 0:
                data[f"c{i}"] = rng.integers(
                    -(10**8), 10**8, size=n_rows, dtype="int64"
                )
            elif kind == 1:
                data[f"c{i}"] = rng.standard_normal(n_rows) * 1e6
            else:
                data[f"c{i}"] = pool[rng.integers(0, len(pool), size=n_rows)]
    else:
        raise ValueError(name)

    pd.DataFrame(data).to_csv(path, index=False)


def _ensure_fixtures(
    dirpath: str, names: list[str], rows_override: int | None = None
) -> dict[str, str]:
    os.makedirs(dirpath, exist_ok=True)
    suffix = f"_{rows_override}" if rows_override else ""
    paths = {}
    for name in names:
        path = os.path.join(dirpath, f"{name}{suffix}.csv")
        if not os.path.exists(path):
            print(f"[gen] building {name} ...", flush=True)
            _build_fixture(name, path, rows_override)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"[gen] {name}: {size_mb:.1f} MB", flush=True)
        paths[name] = path
    return paths


# --------------------------------------------------------------------------
# Single-config timing (runs in its own process)
# --------------------------------------------------------------------------
def _peak_rss_mb() -> float | None:
    # Windows: query PeakWorkingSetSize directly (no third-party dependency).
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        get_proc = ctypes.windll.kernel32.GetCurrentProcess
        get_proc.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PMC),
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL

        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        if get_info(get_proc(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / 1024 / 1024
        return None
    try:
        import resource

        # ru_maxrss is KB on Linux, bytes on macOS.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024 if sys.platform.startswith("linux") else peak / 1024 / 1024
    except Exception:
        return None


def _run_single(path: str, n_threads: int, reps: int) -> dict:
    import pandas as pd

    import pandas.io.parsers.readers as _readers

    # Optionally drop the size floor so sub-50MB fixtures still take the
    # parallel path (used for size sweeps and cheap smoke tests).
    if os.environ.get("PANDAS_BENCH_FORCE_PARALLEL"):
        _readers._PARALLEL_READ_MIN_BYTES = 1

    # Instrument the parallel entry point so we can confirm the path we think
    # we are timing is actually the one that ran.
    parallel_calls = {"n": 0}
    _orig = _readers._read_csv_parallel

    def _wrapped(*args, **kwargs):
        parallel_calls["n"] += 1
        return _orig(*args, **kwargs)

    _readers._read_csv_parallel = _wrapped

    pd.set_option("mode.max_threads", n_threads)

    # Warm the OS file cache (and the process) before timing.
    pd.read_csv(path)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        pd.read_csv(path)
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    return {
        "path": os.path.basename(path),
        "n_threads": n_threads,
        "reps": reps,
        "median_ms": round(median * 1000, 2),
        "min_ms": round(times[0] * 1000, 2),
        "parallel_engaged": parallel_calls["n"] > 0,
        "peak_rss_mb": _peak_rss_mb(),
    }


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def _orchestrate(args: argparse.Namespace) -> None:
    names = args.fixtures.split(",")
    paths = _ensure_fixtures(args.datadir, names, args.rows)
    threads = [int(t) for t in args.threads.split(",")]

    results = []
    for name in names:
        for nthr in threads:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--single",
                    paths[name],
                    str(nthr),
                    str(args.reps),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                print(f"[err] {name} T={nthr} failed:\n{proc.stderr}", flush=True)
                results.append(
                    {
                        "path": f"{name}.csv",
                        "n_threads": nthr,
                        "error": proc.stderr[-2000:],
                    }
                )
                continue
            # The child prints the JSON result on its last non-empty line.
            line = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
            res = json.loads(line)
            res["fixture"] = name
            results.append(res)
            print(
                f"[run] {name:8s} T={nthr}  median={res['median_ms']:8.2f}ms  "
                f"min={res['min_ms']:8.2f}ms  parallel={res['parallel_engaged']}  "
                f"peakRSS={res.get('peak_rss_mb')}",
                flush=True,
            )

    _print_summary(results, names, threads)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {"results": results, "python": sys.version, "platform": sys.platform},
                fh,
                indent=2,
            )
        print(f"[out] wrote {args.out}", flush=True)


def _print_summary(results: list[dict], names: list[str], threads: list[int]) -> None:
    by_key = {
        (r.get("fixture"), r.get("n_threads")): r for r in results if "median_ms" in r
    }
    print("\n===== SPEEDUP vs T=1 (median) =====", flush=True)
    header = "fixture   " + "".join(f"  T={t:<8d}" for t in threads)
    print(header, flush=True)
    for name in names:
        base = by_key.get((name, 1))
        cells = []
        for t in threads:
            cur = by_key.get((name, t))
            if cur is None or base is None:
                cells.append("     n/a ")
            elif t == 1:
                cells.append(f"{base['median_ms']:7.0f}ms")
            else:
                spd = base["median_ms"] / cur["median_ms"]
                cells.append(f"  {spd:5.2f}x ")
        print(f"{name:8s}  " + "".join(f"  {c}" for c in cells), flush=True)
    print("(T=1 column shows absolute median ms; others show speedup)\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", nargs=3, metavar=("CSV", "THREADS", "REPS"))
    parser.add_argument("--fixtures", default="ints,floats,strings,mixed")
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument(
        "--rows", type=int, default=None, help="override row count for all fixtures"
    )
    parser.add_argument("--datadir", default="csv_bench_data")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.single:
        csv, nthr, reps = args.single
        print(json.dumps(_run_single(csv, int(nthr), int(reps))), flush=True)
    else:
        _orchestrate(args)


if __name__ == "__main__":
    main()
