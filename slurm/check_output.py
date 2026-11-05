"""Scan the output directory and report which runs are missing, per script and period.

Usage
  python slurm/check_output.py [periods...] [--scripts eta lambda] [--outdir DIR] [--list]

The run list comes from periods.py (needs /cache), the expected files per run are the ones written by
find_eta.py and find_lambda.py. A run is
  ok       it is in <outdir>/merged/done_runs.txt (kept by merge_output.py), or all files exist and are not empty
  partial  some files exist, the job stopped part way
  missing  no file at all, the job failed or never started
--list prints only the missing and partial runs, one per line as 'script period run' (summary goes to stderr).
Resubmitting with slurm/submit.sh only redoes runs whose last file is absent, so it fills these gaps.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from periods import periods, list_files, run_number

OUTDIR = "/volatile/clas12/sangbaek/exclusive_reactions"
EXPECTED = {
    "eta":    ("eta_{run}_{ch}.parquet",    ("pi0", "eta_gg", "eta_3pi0", "eta_pippimpi0", "eta_pippimg")),
    "lambda": ("lambda_{run}_{ch}.parquet", ("ekppim", "ekp", "ek", "ekppimg", "ekpg", "ekg")),
}


def manifest_path(outdir):
    return os.path.join(outdir, "merged", "done_runs.txt")


def read_done(outdir):
    """Set of 'script run' lines of the list that merge_output.py keeps of the runs it has merged."""
    try:
        with open(manifest_path(outdir)) as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def status(outdir, script, run, done=frozenset()):
    if f"{script} {run}" in done:      # merged earlier, the per-run files may be gone
        return "ok"
    pattern, channels = EXPECTED[script]
    have = [os.path.isfile(p) and os.path.getsize(p) > 0
            for p in (os.path.join(outdir, pattern.format(run=run, ch=c)) for c in channels)]
    return "ok" if all(have) else "missing" if not any(have) else "partial"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("periods", nargs="*", help="default is every period in periods.py")
    ap.add_argument("--scripts", nargs="+", default=list(EXPECTED), choices=list(EXPECTED))
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--list", action="store_true", help="print the runs to redo as 'script period run'")
    args = ap.parse_args()

    names = args.periods or list(periods)
    unknown = [n for n in names if n not in periods]
    if unknown:
        sys.exit(f"unknown period(s) {unknown}, choose from {list(periods)}")

    done = read_done(args.outdir)
    redo, anything = [], False
    for period in names:
        runs = [run_number(f) for f in list_files(period)]
        if not runs:
            print(f"[{period}] no HIPO files found, is /cache mounted here?", file=sys.stderr)
            continue
        anything = True
        for script in args.scripts:
            st = {run: status(args.outdir, script, run, done) for run in runs}
            bad = [(r, s) for r, s in st.items() if s != "ok"]
            n = {k: sum(v == k for v in st.values()) for k in ("ok", "partial", "missing")}
            print(f"[{period}] {script:6s} {len(runs)} runs  ok {n['ok']}  partial {n['partial']}  missing {n['missing']}",
                  file=sys.stderr if args.list else sys.stdout)
            for r, s in bad:
                redo.append((script, period, r))
                if not args.list:
                    print(f"    {s:8s} {r}")
    if args.list:
        for script, period, run in redo:
            print(script, period, run)
    if not anything:
        sys.exit(1)
    sys.exit(1 if redo else 0)


if __name__ == "__main__":
    main()
