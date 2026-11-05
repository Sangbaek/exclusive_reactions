"""Merge the per-run parquet files into one file per final state, and keep a list of the merged runs.

Usage
  python slurm/merge_output.py [--scripts eta lambda] [--outdir DIR]

Every run whose files are all there for every selected script (5 for eta and 6 for lambda) is added to
  <outdir>/merged/<script>_<period>_<channel>.parquet   e.g. merged/lambda_sp18_inb_ekppim.parquet
one file per period and final state. It has two extra columns, run (integer) and period (name from periods.py). A run that is already in the
merged file is replaced, other rows stay, so the command can be repeated after every batch of jobs.
A run that has only eta or only lambda output waits until the other one is there. Runs with an empty table add no rows.

It leaves three things in <outdir>/merged
  <script>_<period>_<channel>.parquet   the merged output
  done_runs.txt        the runs merged so far, one 'script run' line each, added to on every call.
                       submit.sh and check_output.py read it, so a merged run counts as done even when its
                       per-run files are gone.
  to_delete.txt        the per-run files that are merged and checked, one path per line, rewritten on every call
                       with the files still on disk. Nothing is deleted. Look at the list, then remove them
                         xargs -a <outdir>/merged/to_delete.txt rm
Run it while no job is writing, and where /cache is mounted (it gives the period of each run).
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from periods import periods, list_files, run_number
from check_output import EXPECTED, OUTDIR, manifest_path, read_done

NAME = re.compile(r"^(eta|lambda)_(\d+)_(.+)\.parquet$")


def run_periods():
    """{run number: period}, from the HIPO file lists."""
    return {int(run_number(f)): p for p in periods for f in list_files(p)}


def merge_channel(path, files, period_of, label):
    """files is {run string: path} of the runs to add or replace in the parquet file at path."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    runs = {int(r) for r in files}
    writer, schema, kept, added = None, None, 0, 0
    if os.path.exists(path):                       # keep the rows of the runs that are not redone
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        writer = pq.ParquetWriter(tmp, schema)
        for batch in pf.iter_batches():
            t = pa.Table.from_batches([batch])
            t = t.filter(pc.invert(pc.is_in(t["run"], value_set=pa.array(sorted(runs), type=t["run"].type))))
            if t.num_rows:
                writer.write_table(t)
                kept += t.num_rows
    expect = {}
    for run, f in sorted(files.items()):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        df["run"] = int(run)
        df["period"] = period_of.get(int(run), "unknown")
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(tmp, schema)
        writer.write_table(table.cast(schema))
        expect[int(run)] = len(df)
        added += len(df)
    if writer is None:
        print(f"{label:34s} {len(files)} runs, no rows, nothing written")
        return None
    writer.close()
    got = pq.read_table(tmp, columns=["run"]).column("run").to_pandas().value_counts()
    if len(got) and got.sum() != kept + added or any(got.get(r, 0) != n for r, n in expect.items()):
        os.remove(tmp)
        sys.exit(f"{script} {channel}: row counts after merging do not match, nothing replaced")
    os.replace(tmp, path)
    print(f"{label:34s} {kept + added:11,} rows in total, {added:,} new from {len(expect)} of {len(files)} runs")
    return path


def write_manifest(outdir, done):
    path = manifest_path(outdir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.writelines(line + "\n" for line in sorted(done))
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts", nargs="+", default=["eta", "lambda"], choices=["eta", "lambda"])
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    period_of = run_periods()
    if not period_of:
        print("warning: no HIPO files found, is /cache mounted here? periods will be 'unknown'", file=sys.stderr)

    found = defaultdict(dict)                        # (script, channel) -> {run: path}
    for f in glob.glob(os.path.join(args.outdir, "*.parquet")):
        m = NAME.match(os.path.basename(f))
        if m and m.group(1) in args.scripts and m.group(3) in EXPECTED[m.group(1)][1]:
            found[(m.group(1), m.group(3))][m.group(2)] = f
    if not found:
        print(f"no per-run files found in {args.outdir}, nothing to merge")
        return

    # A run is merged only when every selected script has all its files for it, since the jobs run both.
    complete, anyfiles = {}, set()
    for script in args.scripts:
        per = [set(found.get((script, ch), {})) for ch in EXPECTED[script][1]]
        complete[script] = set.intersection(*per)
        anyfiles |= set.union(*per)
    ready = set.intersection(*complete.values())
    for script in args.scripts:
        waiting = sorted(complete[script] - ready)
        print(f"{script}: {len(complete[script])} complete runs"
              + (f", {len(waiting)} wait for the other script ({' '.join(waiting[:5])}{' ...' if len(waiting) > 5 else ''})" if waiting else ""))
    print(f"{len(ready)} runs complete for {' and '.join(args.scripts)}, {len(anyfiles - ready)} with incomplete output left alone")
    if not ready:
        return

    for script in args.scripts:
        for ch in EXPECTED[script][1]:
            files = {r: found[(script, ch)][r] for r in ready}
            by_period = defaultdict(dict)
            for r, f in files.items():
                by_period[period_of.get(int(r), "unknown")][r] = f
            for p, fs in sorted(by_period.items()):
                merge_channel(os.path.join(args.outdir, "merged", f"{script}_{p}_{ch}.parquet"), fs, period_of,
                              f"{script}_{p}_{ch}")
    done = read_done(args.outdir) | {f"{script} {r}" for script in args.scripts for r in ready}
    write_manifest(args.outdir, done)
    files = [found[(script, ch)][r] for script in args.scripts for ch in EXPECTED[script][1] for r in sorted(ready)]
    listing = os.path.join(args.outdir, "merged", "to_delete.txt")
    with open(listing, "w") as f:
        f.writelines(p + "\n" for p in files)
    print()
    print(f"merged output   {os.path.join(args.outdir, 'merged')}/<script>_<period>_<channel>.parquet")
    print(f"merged runs     {len(done)} script/run pairs so far, listed in {manifest_path(args.outdir)}")
    print(f"safe to remove  {len(files)} per-run files, listed in {listing} (nothing was removed)")
    print(f"delete with     xargs -a {listing} rm")


if __name__ == "__main__":
    main()
