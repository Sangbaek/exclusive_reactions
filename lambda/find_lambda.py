import ROOT
import pandas as pd
import awkward as ak
import numpy as np
import glob
import os
import random


periods = {
    "sp19_inb":     "/cache/clas12/rg-a/production/recon/spring2019/torus-1/pass2/dst/train/nSidis/",
    "sp18_inb":     "/cache/clas12/rg-a/production/recon/spring2018/10.59gev/torus-1/pass1/dst/train/nSidis/",
    "sp18_outb":    "/cache/clas12/rg-a/production/recon/spring2018/10.59gev/torus+1/pass1/dst/train/nSidis/",
    "fall2018_inb": "/cache/clas12/rg-a/production/recon/fall2018/torus-1/pass2/main/train/nSidis/",
    "fall2018_outb":"/cache/clas12/rg-a/production/recon/fall2018/torus+1/pass2/train/nSidis/",
}
periods = {"sp18_inb": periods["sp18_inb"]}
N_RUNS_PER_PERIOD = 1#999999999
random.seed(42)  # reproducible sample selection

LAMBDA_MIN, LAMBDA_MAX = 0.8, 1.3       # GeV, p pi- window applied to events


ROOT.gInterpreter.AddIncludePath("/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/hipo/4.5.0/root_6.38.04/include")
ROOT.gInterpreter.ProcessLine('#include "hipo4/RHipoDS.hxx"')
ROOT.DisableImplicitMT()

os.makedirs("dfs", exist_ok=True)

def process_file(fname):
    """Returns a particle-level DataFrame (all REC particles) for events with e' pi- p K+ and a p pi- pair in the Lambda window, or None if empty."""
    df = ROOT.MakeHipoDataFrame(fname)
    cols = ["REC_Particle_px", "REC_Particle_py", "REC_Particle_pz", "REC_Particle_pid", "REC_Particle_status"]
    npdict = df.AsNumpy(columns=cols)
    ak_cols = {c: ak.from_iter(npdict[c]) for c in cols}
    particles = ak.zip(ak_cols)

    pdf = ak.to_dataframe(particles)
    pdf.columns = ["px", "py", "pz", "pid", "status"]

    # final state e' pi- p K+ (Lambda -> p pi-), photons are optional extras
    selections = {
        "electron": (pdf.pid == 11) & (pdf.status < 0),  # trigger electron
        "pim":      (pdf.pid == -211),
        "proton":   (pdf.pid == 2212),
        "kaon":     (pdf.pid == 321),
    }
    good_entries = None
    for label, mask in selections.items():
        entries = set(pdf.loc[mask].index.get_level_values("entry"))
        if not entries:
            return None
        good_entries = entries if good_entries is None else good_entries & entries
    if not good_entries:
        return None

    # p pi- invariant mass for every pair in these events, then cut on the window
    M_P, M_PI = 0.938272, 0.139570
    pr = pdf.loc[selections["proton"]].reset_index()
    pi = pdf.loc[selections["pim"]].reset_index()
    pr = pr[pr["entry"].isin(good_entries)]
    pi = pi[pi["entry"].isin(good_entries)]
    pair = pr.merge(pi, on="entry", suffixes=("_p", "_pi"))
    Ep  = np.sqrt(pair.px_p**2  + pair.py_p**2  + pair.pz_p**2  + M_P**2)
    Epi = np.sqrt(pair.px_pi**2 + pair.py_pi**2 + pair.pz_pi**2 + M_PI**2)
    psq = (pair.px_p + pair.px_pi)**2 + (pair.py_p + pair.py_pi)**2 + (pair.pz_p + pair.pz_pi)**2
    lam_mass = np.sqrt(np.clip((Ep + Epi)**2 - psq, 0, None))
    in_window = (lam_mass >= LAMBDA_MIN) & (lam_mass <= LAMBDA_MAX)
    good_entries = set(pair.loc[in_window, "entry"])
    if not good_entries:
        return None

    # keep every particle (photons included) of the selected events
    out = pdf[pdf.index.get_level_values("entry").isin(good_entries)].reset_index()
    out["n_gamma"] = (out.pid == 22).groupby(out["entry"]).transform("sum")
    out["file"] = os.path.basename(fname)
    return out

for period_name, base_dir in periods.items():
  # recursive glob catches directory-layout differences between periods
  all_files = glob.glob(os.path.join(base_dir, "**", "*.hipo"), recursive=True)
  # keep only nSidis-trained files if the dir isn't already scoped to that train
  all_files = [f for f in all_files if "nSidis" in f]

  if not all_files:
    print(f"[{period_name}] WARNING: no files found under {base_dir}")
    continue

  n_sample = min(N_RUNS_PER_PERIOD, len(all_files))
  sample_files = random.sample(all_files, n_sample)

  print(f"[{period_name}] found {len(all_files)} files, sampling {n_sample}")

  frames = []
  for fname in sample_files:
    out = process_file(fname)
    if out is None:
      print(f"  {fname}: no selected events, skipping")
      continue
    frames.append(out)
    print(f"  {fname}: {out['entry'].nunique()} events")

  if not frames:
    print(f"[{period_name}] no events accumulated, nothing saved")
    continue

  period_df = pd.concat(frames, ignore_index=True)
  period_df["period"] = period_name
  period_df.to_parquet(f"dfs/{period_name}.parquet")
  print(f"[{period_name}] saved {len(period_df)} particles to dfs/{period_name}.parquet")

print("Saved per-period dataframes to dfs/")
