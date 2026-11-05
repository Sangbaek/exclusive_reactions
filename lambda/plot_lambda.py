"""Exclusivity plots for the collected lambda tables of one period.

Usage
  python plot_lambda.py <period|all> [--channel ekppim|ekp] [--indir DIR] [--figdir DIR] [--log] [--tag TEXT]
                        [--recollect]

Step 1, collect: merges <indir>/lambda_<run>_<channel>.parquet over the runs of the period (run numbers from
periods.py) into <indir>/collected/lambda_<period>_<channel>.parquet, with extra columns period and run.
It is redone only if that file is missing, older than a per-run file, or --recollect is given.
Missing runs are reported. Collecting needs /cache mounted, re-plotting an up-to-date file does not.
Step 2, plot: saves <figdir>/lambda_<period>_<channel>_exclusivity.png, default figdir <indir>/collected/figures.

The missing masses are computed here from the saved momenta, with the beam energy from periods.py.
Dashed lines mark the expected values. Every panel is drawn before and after the cut on MM^2(e'K+p)
(default -0.1 to 0.1 GeV^2, --cut LO HI to change).
  ekppim  M2(p pi-) from 1 to 2, MM2 of e'K+ (Lambda^2), MM2 of e'K+p pi- (full final state, 0), MM2 of e'K+p (pi- missing)
  ekp     MM2 of e'K+ (Lambda^2), MM2 of e'K+p (pi- missing, m_pi^2)
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # periods.py
sys.path.insert(0, HERE)                        # find_lambda.py
from periods import periods, beam_energy, list_files, run_number
from find_lambda import M_E, M_P, M_K, M_PI, _p4, _add, _sub, _m2

DEFAULT_INDIR = "/volatile/clas12/sangbaek/exclusive_reactions"

M_LAMBDA, M_SIGMA0 = 1.115683, 1.192642

# light surface, one series colour, recessive grid and reference lines
SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"

ZERO = [(0.0, "0")]
PI2 = [(M_PI**2, r"$m_\pi^2$")]
LAM2 = [(M_LAMBDA**2, r"$\Lambda$"), (M_SIGMA0**2, r"$\Sigma^0$")]
BEFORE = "#9b9a95"      # neutral for the uncut sample
CUT = (-0.1, 0.1)       # window on MM^2(e'K+p) in GeV^2, default for --cut


def read_table(path, columns=None):
    """Parquet file as a DataFrame, read through pyarrow (faster than pd.read_parquet for big or wide files).
    columns limits what is read, names that are not in the file are ignored."""
    if columns is not None:
        have = set(pq.read_schema(path).names)
        columns = [c for c in columns if c in have]
    return pq.read_table(path, columns=columns).to_pandas()


def collected_path(indir, period, channel):
    return os.path.join(indir, "collected", f"lambda_{period}_{channel}.parquet")


def collect(period, channel, indir, force=False):
    """Merge the tables of one period. Returns the output path, or None if there is nothing.
    Rows come from merged/lambda_<period>_<channel>.parquet (slurm/merge_output.py) where it exists,
    and from the per-run files for the runs of this period that are not in it."""
    path = collected_path(indir, period, channel)
    merged = os.path.join(indir, "merged", f"lambda_{period}_{channel}.parquet")
    sources = glob.glob(os.path.join(indir, f"lambda_*_{channel}.parquet")) + ([merged] if os.path.exists(merged) else [])
    if os.path.exists(path) and not force and sources and os.path.getmtime(path) >= max(map(os.path.getmtime, sources)):
        print(f"[{period}] {path} is up to date")
        return path

    runs = [run_number(f) for f in list_files(period)]
    if not runs:
        print(f"[{period}] no HIPO files found, is /cache mounted here?", file=sys.stderr)
        return None
    frames, missing, empty = [], [], 0
    have, n_perrun = set(), 0
    try:
        with open(os.path.join(indir, "merged", "done_runs.txt")) as fh:
            merged_runs = {line.strip() for line in fh}
    except FileNotFoundError:
        merged_runs = set()
    if os.path.exists(merged):
        m = read_table(merged)
        have = set(m["run"].unique())
        if len(m):
            frames.append(m)
    for run in runs:
        if int(run) in have:
            continue
        f = os.path.join(indir, f"lambda_{run}_{channel}.parquet")
        if not os.path.exists(f):
            if f"lambda {run}" in merged_runs:     # merged earlier, it had no rows in this table
                empty += 1
            else:
                missing.append(run)
            continue
        df = read_table(f)
        if df.empty:
            empty += 1
            continue
        df["run"] = int(run)
        frames.append(df)
        n_perrun += 1

    print(f"[{period}] {len(runs)} runs, {len(have)} in the merged file, {n_perrun} from per-run files, "
          f"{empty} empty, {len(missing)} missing")
    if missing:
        shown = " ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        print(f"[{period}] missing runs (job failed or not run yet) {shown}", file=sys.stderr)
    if not frames:
        print(f"[{period}] nothing to write", file=sys.stderr)
        return None
    out = pd.concat(frames, ignore_index=True)
    out["period"] = pd.Categorical([period] * len(out))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_parquet(path)
    print(f"[{period}] {len(out)} rows from {out['run'].nunique()} runs -> {path}")
    return path


def _parts(df, beam):
    init = (beam + M_P, 0.0, 0.0, beam)

    def p4(label, mass):
        return _p4({c: df[f"{label}_{c}"].to_numpy(np.float64) for c in ("px", "py", "pz")}, mass)

    def mm2(*parts):
        v = init
        for q in parts:
            v = _sub(v, q)
        return _m2(v)

    return p4, mm2


def mm2_ekp(df, beam):
    p4, mm2 = _parts(df, beam)
    return mm2(p4("e", M_E), p4("k", M_K), p4("p", M_P))


def panels(df, beam, channel):
    """List of (axis label, values, range, reference lines). Add a panel by appending here."""
    p4, mm2 = _parts(df, beam)
    e, k, p = p4("e", M_E), p4("k", M_K), p4("p", M_P)
    mm2_ek = (r"MM$^2$ $e'K^+$ [GeV$^2$]", mm2(e, k), (0.8, 2.5), LAM2)
    mm2_ekp_ = (r"MM$^2$ $e'K^+p$, $\pi^-$ missing [GeV$^2$]", mm2(e, k, p), (-0.3, 0.5), PI2)
    if channel == "ekppim":
        pim = p4("pim", M_PI)
        m2_ppim = _m2(_add(p, pim))
        return [
            (r"$M^2(p\pi^-)$ [GeV$^2$]", m2_ppim, (1.0, 2.0), LAM2[:1]),
            mm2_ek,
            (r"MM$^2$ $e'K^+p\pi^-$, full final state [GeV$^2$]", mm2(e, k, p, pim), (-0.3, 0.3), ZERO),
            mm2_ekp_,
        ]
    return [mm2_ek, mm2_ekp_]   # ekp


def plot_period(period, channel, indir, figdir, log, tag, recollect=False, cut=CUT, bins=100):
    path = collect(period, channel, indir, recollect)
    if path is None:
        return None
    need = ["run", "mass", "beam_energy"] + [f"{p}_{c}" for p in ("e", "k", "p", "pim") for c in ("px", "py", "pz")]
    df = read_table(path, need)       # only the columns the plots use
    beam = beam_energy[period]
    if "beam_energy" in df.columns and not np.allclose(df["beam_energy"], beam):
        print(f"[{period}] tables were written with beam energy {sorted(df['beam_energy'].unique())}, "
              f"using {beam} from periods.py", file=sys.stderr)

    keep = (mm2_ekp(df, beam) > cut[0]) & (mm2_ekp(df, beam) < cut[1])
    items = panels(df, beam, channel)
    nrows = math.ceil(len(items) / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(10.4, 3.9 * nrows), facecolor=SURFACE, squeeze=False)
    for ax, (label, values, rng, refs) in zip(axes.flat, items):
        ax.set_facecolor(SURFACE)
        for sel, color, alpha, name in ((slice(None), BEFORE, 0.35, "before cut"), (keep, SERIES, 0.35, "after cut")):
            v = values[sel]
            ax.hist(v, bins=bins, range=rng, histtype="stepfilled", color=color, alpha=alpha, linewidth=0)
            ax.hist(v, bins=bins, range=rng, histtype="step", color=color, linewidth=1.4,
                    label=f"{name}  ({len(v):,})")
        for x, text in refs:
            ax.axvline(x, color=INK2, linestyle=(0, (4, 3)), linewidth=0.9)
            ax.annotate(text, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2), textcoords="offset points",
                        va="bottom", ha="center", color=INK2, fontsize=9)
        if "e'K^+p$, $\\pi^-$ missing" in label:
            ax.axvspan(*cut, color=INK2, alpha=0.08, linewidth=0)
        ax.set_xlabel(label, color=INK, fontsize=10)
        ax.set_ylabel("rows", color=INK2, fontsize=9)
        ax.set_xlim(*rng)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=9)
    for ax in list(axes.flat)[len(items):]:
        ax.set_visible(False)

    title = (f"{period}  {channel}  beam {beam} GeV  {df['run'].nunique()} runs  "
             f"cut {cut[0]} < MM$^2(e'K^+p)$ < {cut[1]} GeV$^2$")
    fig.suptitle(title + (f"  ({tag})" if tag else ""), color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, f"lambda_{period}_{channel}_exclusivity.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[{period}] saved {out}  ({int(keep.sum()):,} of {len(df):,} rows pass the cut)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("period", help="period name from periods.py, or 'all'")
    ap.add_argument("--channel", default="ekppim", choices=("ekppim", "ekp"))
    ap.add_argument("--indir", default=DEFAULT_INDIR, help="where the per-run files are")
    ap.add_argument("--figdir", default=None, help="default <indir>/collected/figures")
    ap.add_argument("--log", action="store_true", help="logarithmic y axis")
    ap.add_argument("--recollect", action="store_true", help="rebuild the collected file even if it is up to date")
    ap.add_argument("--cut", nargs=2, type=float, default=CUT, metavar=("LO", "HI"),
                    help="window on MM^2(e'K+p) in GeV^2, default %(default)s")
    ap.add_argument("--tag", default="", help="extra text for the figure title")
    args = ap.parse_args()

    names = list(periods) if args.period == "all" else [args.period]
    unknown = [n for n in names if n not in periods]
    if unknown:
        sys.exit(f"unknown period {unknown}, choose from {list(periods)} or all")
    figdir = args.figdir or os.path.join(args.indir, "collected", "figures")
    done = [plot_period(n, args.channel, args.indir, figdir, args.log, args.tag, args.recollect, tuple(args.cut)) for n in names]
    if not any(done):
        sys.exit(1)


if __name__ == "__main__":
    main()
