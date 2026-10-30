"""RG-A nSidis train directories per data-taking period, shared by the analysis scripts.

Usage
  python periods.py                  list every nSidis HIPO file as 'period run path', one per line
  python periods.py sp18_inb ...     only the given periods
  python periods.py --beam sp18_inb  nominal beam energy of one period in GeV

  from periods import periods, list_files, run_number
"""
import glob
import os
import re
import sys

periods = {
    "sp19_inb":     "/cache/clas12/rg-a/production/recon/spring2019/torus-1/pass2/dst/train/nSidis/",
    "sp18_inb":     "/cache/clas12/rg-a/production/recon/spring2018/10.59gev/torus-1/pass1/dst/train/nSidis/",
    "sp18_outb":    "/cache/clas12/rg-a/production/recon/spring2018/10.59gev/torus+1/pass1/dst/train/nSidis/",
    "fall2018_inb": "/cache/clas12/rg-a/production/recon/fall2018/torus-1/pass2/main/train/nSidis/",
    "fall2018_outb":"/cache/clas12/rg-a/production/recon/fall2018/torus+1/pass2/train/nSidis/",
}


# Nominal beam energy in GeV, needed for missing masses. From the directory names and the RG-A
# announcements (spring 2019 10.2, fall 2018 10.604 for inbending). Use the RCDB value for precision work.
beam_energy = {
    "sp19_inb":      10.2,
    "sp18_inb":      10.59,
    "sp18_outb":     10.59,
    "fall2018_inb":  10.604,
    "fall2018_outb": 10.6,
}


def list_files(period):
    """Sorted HIPO files of one period (recursive glob catches directory-layout differences)."""
    files = glob.glob(os.path.join(periods[period], "**", "*.hipo"), recursive=True)
    return sorted(f for f in files if "nSidis" in f)


def run_number(path):
    """Run number as written in the file name, e.g. '003306' for nSidis_003306.hipo."""
    m = re.search(r"_(\d+)\.hipo$", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot find a run number in {path}")
    return m.group(1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--beam"]:
        if len(sys.argv) != 3 or sys.argv[2] not in beam_energy:
            sys.exit(f"usage: periods.py --beam PERIOD, PERIOD is one of {list(beam_energy)}")
        print(beam_energy[sys.argv[2]])
        sys.exit(0)
    names = sys.argv[1:] or list(periods)
    unknown = [n for n in names if n not in periods]
    if unknown:
        sys.exit(f"unknown period(s) {unknown}, choose from {list(periods)}")
    seen = {}
    for name in names:
        files = list_files(name)
        if not files:
            print(f"[{name}] WARNING: no files found under {periods[name]}", file=sys.stderr)
        for f in files:
            run = run_number(f)
            if run in seen:  # outputs are named by run number, so a repeat would overwrite
                sys.exit(f"run {run} appears twice: {seen[run]} and {f}")
            seen[run] = f
            print(name, run, f)
