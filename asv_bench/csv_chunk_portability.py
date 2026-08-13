"""
Does the parallel-``read_csv`` chunk-count calibration hold off macOS?

The two constants on ``perf-read_csv-chunk-count`` -- ``_PARALLEL_MIN_CHUNK_BYTES``
(1 MB) and ``_PARALLEL_MAX_COLUMN_PIECES`` (1800) -- were fitted on one M3 Pro.
Both are expressed in bytes and in (column x chunk) pieces, not in workers, so
their premises are testable on any machine including a 4-vCPU CI runner:

* per-chunk cost rises with chunk count (what the byte floor exists to bound), and
* the rise scales with column count (what the piece budget exists to bound).

What a small runner canNOT answer is where the knee sits at 8-16 workers, or the
4-vs-12-worker ratios.  Those need real cores; nothing here pretends otherwise.

Everything below is measured as a *ratio between configurations interleaved
inside one process*, which is what survives a noisy shared runner.  Absolute
milliseconds from CI are not comparable to anything and are printed only as
context.

Run it with no arguments::

    python asv_bench/csv_chunk_portability.py

Useful flags: ``--fixture-mb`` (default 48) trades runtime for signal,
``--rounds``/``--reps`` control the best-of, ``--quick`` halves everything for a
smoke test.  The script exits non-zero if a configuration it meant to measure in
parallel silently fell back to a serial read, since that failure looks like a
flat, well-behaved result rather than an error.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time

try:
    import psutil
except ImportError:  # optional; see current_rss_bytes
    psutil = None

import numpy as np

import pandas as pd

from pandas.io.parsers import readers

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

_ORIG_OFFSETS = readers._find_chunk_byte_offsets
# Present only on the branch under test; absent on main.  Their absence is not
# an error -- the forced-chunk-count experiments characterise the machine and
# run identically either way.
HAS_CAPS = hasattr(readers, "_PARALLEL_MIN_CHUNK_BYTES") and hasattr(
    readers, "_PARALLEL_MAX_COLUMN_PIECES"
)


def peak_rss_bytes() -> int:
    """Process peak resident set, or 0 where it cannot be read."""
    if sys.platform == "win32":

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return 0
        return int(counters.PeakWorkingSetSize)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def current_rss_bytes() -> int:
    """Resident set *right now*, or 0 where it cannot be read.

    Deliberately not the high-water mark.  ``ru_maxrss`` is a peak that cannot
    be reset, and importing pandas + pyarrow can peak higher than the read
    being measured ever does -- on a CI runner the import mark was 505 MB and
    a 48 MB read never moved it, reporting a flat zero for every chunk count.
    """
    if psutil is not None:
        try:
            return int(psutil.Process().memory_info().rss)
        except Exception:
            return 0
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", encoding="ascii") as handle:
                resident_pages = int(handle.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            return 0
    return 0


class RssSampler:
    """Poll resident set on a thread and keep the maximum seen."""

    def __init__(self, interval: float = 0.001) -> None:
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.peak = current_rss_bytes()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, current_rss_bytes())
            time.sleep(self.interval)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.peak = max(self.peak, current_rss_bytes())


def describe_environment() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pandas_path": os.path.dirname(pd.__file__),
        "os_cpu_count": os.cpu_count(),
        "caps_present": HAS_CAPS,
        "min_chunk_bytes": getattr(readers, "_PARALLEL_MIN_CHUNK_BYTES", None),
        "max_column_pieces": getattr(readers, "_PARALLEL_MAX_COLUMN_PIECES", None),
        "read_min_bytes": readers._PARALLEL_READ_MIN_BYTES,
    }


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def build_fixture(
    path: str, n_cols: int, target_bytes: int, mixed: bool = False
) -> None:
    """Write a deterministic CSV of roughly *target_bytes*.

    *mixed* cycles int/float/str/bool instead of int/float.  Per-chunk cost is
    largely per-*column* work, and a string or bool column costs far more of it
    than an integer one, so an all-numeric fixture understates the effect the
    byte floor exists to bound.
    """
    rng = np.random.default_rng(0)
    # Six characters per cell plus a separator keeps rows a predictable width,
    # so the byte target lands without a second pass.
    approx_row_bytes = n_cols * 7
    n_rows = max(target_bytes // approx_row_bytes, 1000)
    words = np.array([f"w{i:04d}" for i in range(500)])
    block = {}
    for col in range(n_cols):
        kind = col % 4 if mixed else col % 2
        if kind == 0:
            block[f"c{col}"] = np.round(rng.random(n_rows) * 1000, 3)
        elif kind == 1:
            block[f"c{col}"] = rng.integers(0, 999_999, size=n_rows, dtype=np.int64)
        elif kind == 2:
            block[f"c{col}"] = words[rng.integers(0, len(words), size=n_rows)]
        else:
            block[f"c{col}"] = rng.integers(0, 2, size=n_rows).astype(bool)
    pd.DataFrame(block).to_csv(path, index=False)


def ensure_fixtures(directory: str, fixture_mb: int) -> dict[str, str]:
    """Byte-matched wide/narrow pair plus one file just over the size gate.

    wide and narrow carry the same cell count and nearly the same byte count and
    differ only in column count, so any divergence between them is attributable
    to columns rather than to size.
    """
    target = fixture_mb * 1024 * 1024
    specs = {
        # wide and narrow are byte- and cell-matched; only column count differs.
        "wide": (100, target, False),
        "narrow": (10, target, False),
        # small is mixed-dtype on purpose: it stands in for a real small file
        # sitting just over the size gate, which is the case the byte floor is for.
        "small": (10, 6 * 1024 * 1024, True),
    }
    paths = {}
    for name, (n_cols, size, mixed) in specs.items():
        path = os.path.join(directory, f"{name}.csv")
        if not os.path.exists(path):
            build_fixture(path, n_cols, size, mixed)
        paths[name] = path
        actual = os.path.getsize(path)
        if actual < readers._PARALLEL_READ_MIN_BYTES:
            raise SystemExit(
                f"fixture {name} is {actual:,} bytes, under the "
                f"{readers._PARALLEL_READ_MIN_BYTES:,}-byte parallel-read gate; "
                "it would be read serially at every setting"
            )
    return paths


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


class Planner:
    """Records the chunk count the code asks for, and optionally overrides it."""

    def __init__(self) -> None:
        self.requested: list[int] = []
        self.forced: int | None = None
        readers._find_chunk_byte_offsets = self._spy

    def _spy(self, filepath, n_chunks, data_start):
        self.requested.append(n_chunks)
        count = self.forced if self.forced is not None else n_chunks
        return _ORIG_OFFSETS(filepath, count, data_start)

    def restore(self) -> None:
        readers._find_chunk_byte_offsets = _ORIG_OFFSETS


def set_caps(planner_on: bool) -> None:
    """Enable or neutralise the two chunk-count caps (branch builds only)."""
    if not HAS_CAPS:
        return
    if planner_on:
        readers._PARALLEL_MIN_CHUNK_BYTES = _CAPS_ON["bytes"]
        readers._PARALLEL_MAX_COLUMN_PIECES = _CAPS_ON["pieces"]
    else:
        readers._PARALLEL_MIN_CHUNK_BYTES = 1
        readers._PARALLEL_MAX_COLUMN_PIECES = 10**9


_CAPS_ON = {
    "bytes": getattr(readers, "_PARALLEL_MIN_CHUNK_BYTES", None),
    "pieces": getattr(readers, "_PARALLEL_MAX_COLUMN_PIECES", None),
}


def timed_read(
    path: str, planner: Planner, require_parallel: bool = True
) -> tuple[float, float, int]:
    """One read: wall ms, CPU ms, chunks planned (0 when read serially).

    *require_parallel* guards the configurations that are supposed to split: a
    silent fallback to a serial read looks like a flat, well-behaved result
    rather than an error.  One-worker configurations are serial by definition
    and pass ``False``.
    """
    planner.requested.clear()
    cpu_before = time.process_time()
    start = time.perf_counter()
    pd.read_csv(path)
    wall = (time.perf_counter() - start) * 1000
    cpu = (time.process_time() - cpu_before) * 1000
    if not planner.requested and not require_parallel:
        return wall, cpu, 0
    if not planner.requested:
        raise SystemExit(
            f"read of {os.path.basename(path)} never reached the parallel "
            "planner -- it fell back to a serial read, which would be reported "
            "as a flat result rather than a failure.\n"
            f"  size            {os.path.getsize(path):,} bytes\n"
            f"  size gate       {readers._PARALLEL_READ_MIN_BYTES:,}\n"
            f"  max_threads     {pd.get_option('mode.max_threads')}\n"
            f"  default workers {readers._default_n_workers()}\n"
            f"  eligible        {readers._can_parallelize_csv(path, {})}\n"
            f"  spy installed   "
            f"{readers._find_chunk_byte_offsets is not _ORIG_OFFSETS}\n"
            f"  min chunk bytes {getattr(readers, '_PARALLEL_MIN_CHUNK_BYTES', None)}\n"
            f"  max col pieces  {getattr(readers, '_PARALLEL_MAX_COLUMN_PIECES', None)}"
        )
    return wall, cpu, planner.requested[0]


def interleaved(
    configs: list[tuple],
    apply_config,
    path_of,
    planner: Planner,
    rounds: int,
    reps: int,
    requires_parallel=lambda config: True,
) -> dict:
    """Round-robin every config within each round so drift hits them equally.

    Best-of for wall (noise only ever adds time); median for CPU, which is far
    less sensitive to a busy neighbour and is the number to trust on CI.
    """
    walls: dict = {config: [] for config in configs}
    cpus: dict = {config: [] for config in configs}
    chunks: dict = {}

    for config in configs:  # warm the page cache once per fixture/config
        apply_config(config)
        timed_read(path_of(config), planner, requires_parallel(config))

    for _ in range(rounds):
        for config in configs:
            apply_config(config)
            best = None
            for _ in range(reps):
                wall, cpu, planned = timed_read(
                    path_of(config), planner, requires_parallel(config)
                )
                if best is None or wall < best[0]:
                    best = (wall, cpu)
                chunks[config] = planned
            walls[config].append(best[0])
            cpus[config].append(best[1])

    return {
        config: {
            "wall": min(walls[config]),
            "wall_spread": (
                (max(walls[config]) - min(walls[config])) / min(walls[config]) * 100
            ),
            "cpu": statistics.median(cpus[config]),
            "chunks": chunks[config],
        }
        for config in configs
    }


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def experiment_chunk_scaling(paths, planner, args) -> dict:
    """E1: cost vs chunk count at fixed workers, wide vs byte-matched narrow.

    The piece budget's premise is that this curve steepens with column count.
    Worker count is pinned low deliberately: the question is per-chunk cost, and
    that does not need cores to show up.
    """
    counts = [4, 8, 12, 18, 24, 36, 48]
    if args.quick:
        counts = [4, 12, 24, 48]
    configs = [(name, count) for name in ("wide", "narrow") for count in counts]

    def apply_config(config):
        planner.forced = config[1]
        pd.set_option("mode.max_threads", args.workers)

    set_caps(False)  # forced counts bypass the caps; keep them out of the way
    results = interleaved(
        configs, apply_config, lambda c: paths[c[0]], planner, args.rounds, args.reps
    )
    planner.forced = None
    return {"counts": counts, "results": results}


def experiment_small_file(paths, planner, args) -> dict:
    """E2: the byte floor's premise, on a file just over the size gate.

    Counts start at the worker count: below it the split, not the per-chunk
    cost, is what limits the read (n_workers is clamped to n_chunks), so a
    smaller baseline would be measuring under-parallelisation instead.
    """
    counts = [count for count in (4, 6, 8, 12, 24, 48) if count >= args.workers]
    if args.quick:
        counts = [count for count in (4, 12, 48) if count >= args.workers]
    configs = [("small", count) for count in counts]

    def apply_config(config):
        planner.forced = config[1]
        pd.set_option("mode.max_threads", args.workers)

    set_caps(False)
    results = interleaved(
        configs, apply_config, lambda c: paths[c[0]], planner, args.rounds, args.reps
    )
    planner.forced = None
    return {"counts": counts, "results": results}


def experiment_worker_sweep(paths, planner, args) -> dict:
    """E3: natural planning across low worker counts, caps on vs off.

    On Windows this is the GH#64347 shape (T=1/2/4) that the current
    default-off decision rests on -- notably its unexplained T=2 regression,
    which a 4-vCPU runner reproduces natively.
    """
    workers = [1, 2, 4]
    states = [True, False] if HAS_CAPS else [True]
    configs = [
        (name, worker, caps)
        for name in ("wide", "narrow", "small")
        for worker in workers
        for caps in states
    ]

    def apply_config(config):
        planner.forced = None
        set_caps(config[2])
        pd.set_option("mode.max_threads", config[1])

    results = interleaved(
        configs,
        apply_config,
        lambda c: paths[c[0]],
        planner,
        args.rounds,
        args.reps,
        # One worker is the serial reference -- it is *supposed* not to split.
        requires_parallel=lambda config: config[1] > 1,
    )
    set_caps(True)
    return {"workers": workers, "states": states, "results": results}


def experiment_memory(paths, args) -> dict:
    """E4: peak RSS vs chunk count, one subprocess per point.

    Peak RSS is a process high-water mark and cannot be reset, so each point
    needs its own process.  This is here because the caps *reduce* chunk count
    and therefore *increase* per-chunk size: the GH#64347 reporter hit an OOM at
    12 threads, so the direction matters.
    """
    # Wide contrast: the caps' whole effect is to make chunks bigger, and the
    # per-worker parser buffer is what grows with them.
    counts = [4, 8, 18, 36] if not args.quick else [4, 36]
    out = {}
    for name in ("wide", "narrow"):
        for count in counts:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--probe-memory",
                    paths[name],
                    str(count),
                    str(args.workers),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                out[name, count] = None
                continue
            out[name, count] = json.loads(proc.stdout.strip().splitlines()[-1])
    return {"counts": counts, "results": out}


def probe_memory(path: str, count: int, workers: int) -> None:
    """Child-process entry point for E4."""
    planner = Planner()
    planner.forced = count
    pd.set_option("mode.max_threads", workers)
    sampled = current_rss_bytes() > 0
    if sampled:
        baseline = current_rss_bytes()
        with RssSampler() as sampler:
            pd.read_csv(path)
        peak = sampler.peak
    else:
        # No current-RSS source: fall back to the process high-water mark, which
        # is only meaningful when the read peaks above the import peak.
        baseline = peak_rss_bytes()
        pd.read_csv(path)
        peak = peak_rss_bytes()
    print(
        json.dumps(
            {
                "peak_mb": peak / 1024 / 1024,
                "baseline_mb": baseline / 1024 / 1024,
                "sampled": sampled,
                "chunks": planner.requested[0] if planner.requested else None,
            }
        )
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


_TRANSCRIPT: list[str] = []


def emit(line: str = "") -> None:
    """Print, and keep a copy for the CI job summary."""
    print(line)
    _TRANSCRIPT.append(line)


def write_job_summary() -> None:
    """Mirror the transcript into the GitHub Actions run summary, if we are in one."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    heading = (
        f"{platform.system()} {platform.machine()} / Python {platform.python_version()}"
    )
    with open(summary, "a", encoding="utf-8") as handle:
        handle.write(f"## {heading}\n\n```\n")
        handle.write("\n".join(_TRANSCRIPT))
        handle.write("\n```\n")


def print_table(
    title: str, note: str, header: list[str], rows: list[list[str]]
) -> None:
    emit()
    emit(title)
    emit("-" * len(title))
    if note:
        emit(note)
    emit()
    widths = [
        max(len(str(header[i])), max((len(str(r[i])) for r in rows), default=0))
        for i in range(len(header))
    ]
    emit("  ".join(str(h).rjust(widths[i]) for i, h in enumerate(header)))
    for row in rows:
        emit("  ".join(str(cell).rjust(widths[i]) for i, cell in enumerate(row)))


def report_chunk_scaling(data, workers) -> None:
    counts = data["counts"]
    results = data["results"]
    rows = []
    for name in ("wide", "narrow"):
        base = results[name, counts[0]]
        for count in counts:
            entry = results[name, count]
            rows.append(
                [
                    name,
                    count,
                    f"{entry['wall']:.1f}",
                    f"±{entry['wall_spread']:.0f}%",
                    f"{entry['cpu']:.1f}",
                    f"{entry['wall'] / base['wall']:.2f}x",
                    f"{entry['cpu'] / base['cpu']:.2f}x",
                ]
            )
    print_table(
        f"E1  cost vs chunk count at {workers} workers (byte-matched pair)",
        "Ratios are against each fixture's own lowest chunk count.  If the piece\n"
        "budget's premise holds, wide's ratios climb faster than narrow's.",
        ["fixture", "chunks", "wall ms", "spread", "CPU ms", "wall vs", "CPU vs"],
        rows,
    )


def report_small_file(data, workers) -> None:
    counts = data["counts"]
    results = data["results"]
    base = results["small", counts[0]]
    rows = [
        [
            count,
            f"{results['small', count]['wall']:.2f}",
            f"±{results['small', count]['wall_spread']:.0f}%",
            f"{results['small', count]['cpu']:.2f}",
            f"{results['small', count]['wall'] / base['wall']:.2f}x",
        ]
        for count in counts
    ]
    print_table(
        f"E2  small file (~6 MB) vs chunk count at {workers} workers",
        "The byte floor exists because splitting a small file finely costs more\n"
        "than it buys.  Rising wall/CPU with chunk count is that premise.",
        ["chunks", "wall ms", "spread", "CPU ms", "vs fewest"],
        rows,
    )


def report_worker_sweep(data) -> None:
    results = data["results"]
    rows = []
    for name in ("wide", "narrow", "small"):
        serial = results[name, 1, data["states"][0]]["wall"]
        for worker in data["workers"]:
            row = [name, worker]
            for caps in data["states"]:
                entry = results[name, worker, caps]
                chunks = entry["chunks"] or "serial"
                row += [
                    f"{entry['wall']:.1f}",
                    chunks,
                    f"{serial / entry['wall']:.2f}x",
                ]
            if len(data["states"]) == 2:
                on = results[name, worker, True]["wall"]
                off = results[name, worker, False]["wall"]
                row.append(f"{off / on:.2f}x")
            rows.append(row)
    header = ["fixture", "workers"]
    for caps in data["states"]:
        label = "caps-on" if caps else "caps-off"
        header += [f"{label} ms", "chunks", "vs T=1"]
    if len(data["states"]) == 2:
        header.append("caps gain")
    print_table(
        "E3  natural planning across low worker counts",
        "On Windows this is the GH#64347 shape, whose T=2 point was 45% SLOWER\n"
        "than serial -- the datapoint the Windows default-off rests on.  T=1 is\n"
        "the serial reference, so 'vs T=1' below is directly comparable to it.\n"
        "A T=2 regression here is a separate defect from the chunk-count caps\n"
        "and would block a Windows default flip on its own.",
        header,
        rows,
    )


def report_memory(data) -> None:
    counts = data["counts"]
    results = data["results"]
    measured = [entry for entry in results.values() if entry]
    if measured and not any(
        entry["peak_mb"] - entry["baseline_mb"] > 0 for entry in measured
    ):
        # A table of zeros reads as "no effect" when it means "not measured".
        method = "sampled" if measured[0].get("sampled") else "peak high-water mark"
        print_table(
            "E4  memory growth during the read vs chunk count",
            f"NOT MEASURED on this platform (method: {method}).  Every point\n"
            "reported zero growth, which means the resident set never moved --\n"
            "either the RSS source is unavailable, or the interpreter's import\n"
            "peak already exceeds the read's.  Do not read this as 'the caps\n"
            "cost no memory'; it is a failed measurement, not a null result.",
            ["fixture", "chunks", "peak MB", "at start"],
            [
                [
                    name,
                    count,
                    f"{results[name, count]['peak_mb']:.0f}",
                    f"{results[name, count]['baseline_mb']:.0f}",
                ]
                for name in ("wide", "narrow")
                for count in counts
                if results.get((name, count))
            ],
        )
        return

    rows = []
    for name in ("wide", "narrow"):
        base = results.get((name, counts[-1]))
        for count in counts:
            entry = results.get((name, count))
            if entry is None:
                rows.append([name, count, "failed", "", "", ""])
                continue
            # Peak RSS includes ~120 MB of interpreter and imports, which would
            # swamp the difference; the read's own growth is what matters.
            growth = entry["peak_mb"] - entry["baseline_mb"]
            base_growth = base["peak_mb"] - base["baseline_mb"] if base else 0
            ratio = f"{growth / base_growth:.2f}x" if base_growth > 0 else ""
            rows.append(
                [
                    name,
                    count,
                    f"{entry['peak_mb']:.0f}",
                    f"{entry['baseline_mb']:.0f}",
                    f"{growth:.0f}",
                    ratio,
                ]
            )
    print_table(
        "E4  memory growth during the read vs chunk count (one process per point)",
        "The caps REDUCE chunk count, which INCREASES per-chunk size and so the\n"
        "per-worker parser buffer.  'growth' is peak minus the pre-read baseline;\n"
        "ratios are against the finest split, so >1.00x means the caps cost memory.\n"
        "This is here because GH#64347 reported an OOM at 12 threads.",
        ["fixture", "chunks", "peak MB", "at start", "growth MB", "vs finest"],
        rows,
    )


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--fixture-mb", type=int, default=48)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="worker count for the fixed-worker experiments (default 4)",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dir", default=None, help="fixture directory")
    parser.add_argument("--json", default=None, help="also write raw results here")
    parser.add_argument(
        "--probe-memory", nargs=3, metavar=("PATH", "CHUNKS", "WORKERS")
    )
    args = parser.parse_args(argv)

    if args.probe_memory:
        path, count, workers = args.probe_memory
        probe_memory(path, int(count), int(workers))
        return 0

    if args.quick:
        args.fixture_mb = max(args.fixture_mb // 2, 8)
        args.rounds = max(args.rounds // 2, 2)

    env = describe_environment()
    emit("environment")
    emit("-----------")
    for key, value in env.items():
        emit(f"  {key:<20} {value}")
    if not HAS_CAPS:
        emit(
            "\n  NOTE: this build has no chunk-count caps (main, or a stale\n"
            "  branch).  E1/E2/E4 force chunk counts directly and are unaffected;\n"
            "  E3's caps-on/off comparison is skipped."
        )

    directory = args.dir or os.path.join(tempfile.gettempdir(), "pandas-chunk-port")
    os.makedirs(directory, exist_ok=True)
    print(f"\nbuilding fixtures in {directory} ...", flush=True)
    paths = ensure_fixtures(directory, args.fixture_mb)
    for name, path in paths.items():
        print(f"  {name:<8} {os.path.getsize(path) / 1024 / 1024:8.1f} MB  {path}")

    # Each table prints as soon as its experiment finishes: a failure in a later
    # experiment must not discard the results already paid for.
    planner = Planner()
    try:
        chunk_scaling = experiment_chunk_scaling(paths, planner, args)
        report_chunk_scaling(chunk_scaling, args.workers)
        small_file = experiment_small_file(paths, planner, args)
        report_small_file(small_file, args.workers)
        worker_sweep = experiment_worker_sweep(paths, planner, args)
        report_worker_sweep(worker_sweep)
    finally:
        planner.restore()
        set_caps(True)
    memory = experiment_memory(paths, args)
    report_memory(memory)

    emit(
        "\nReminder: absolute milliseconds from a shared CI runner are not\n"
        "comparable across runs or machines.  The ratios within each table are\n"
        "the result; the knee above 4 workers is not measurable here."
    )
    write_job_summary()

    if args.json:
        payload = {
            "environment": env,
            "chunk_scaling": {
                f"{k[0]}:{k[1]}": v for k, v in chunk_scaling["results"].items()
            },
            "small_file": {
                f"{k[0]}:{k[1]}": v for k, v in small_file["results"].items()
            },
            "worker_sweep": {
                f"{k[0]}:{k[1]}:{k[2]}": v for k, v in worker_sweep["results"].items()
            },
            "memory": {f"{k[0]}:{k[1]}": v for k, v in memory["results"].items()},
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nraw results written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
