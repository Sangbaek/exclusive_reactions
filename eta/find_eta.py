import ROOT
import pandas as pd
import awkward as ak
import numpy as np
import glob
import os
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm as LogNorm


periods = {
    "sp19_inb":     "/cache/clas12/rg-a/production/recon/spring2019/torus-1/pass2/dst/train/nSidis/",
    "sp18_inb":     "/cache/clas12/rg-a/production/recon/spring2018/10.59gev/torus-1/pass1/dst/train/nSidis/",
    "sp18_outb":    "/cache/clas12/rg-a/production/recon/spring2018/10.59gev/torus+1/pass1/dst/train/nSidis/",
    "fall2018_inb": "/cache/clas12/rg-a/production/recon/fall2018/torus-1/pass2/main/train/nSidis/",
    "fall2018_outb":"/cache/clas12/rg-a/production/recon/fall2018/torus+1/pass2/train/nSidis/",
}

N_RUNS_PER_PERIOD = 999999999
random.seed(42)  # reproducible sample selection


ROOT.gInterpreter.AddIncludePath("/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/hipo/4.5.0/root_6.38.04/include")
ROOT.gInterpreter.ProcessLine('#include "hipo4/RHipoDS.hxx"')
ROOT.DisableImplicitMT()  

os.makedirs("figures", exist_ok=True)

# --- fixed binning, defined once so histograms from every run sum cleanly ---
mass_bins  = np.linspace(0.0, 1.0, 1001)     # 100 bins
angle_bins = np.linspace(0, 70, 71)     # 100 bins, degrees

# running totals
hists = {
    name: {
        "h_mass":  np.zeros(len(mass_bins) - 1),
        "h_angle": np.zeros(len(angle_bins) - 1),
        "h_2d":    np.zeros((len(angle_bins) - 1, len(mass_bins) - 1)),
        "files_used": [],
        "eta_candidate": 0,
        "pi0_candidate": 0,
        "eta_sideband": 0,
        "pi0_sideband": 0
    }
    for name in periods
}

def process_file(fname):
    """Returns inv_mass, opening_angle arrays for gamma-gamma multiplicity-2 events, or (None, None) if empty."""
    df = ROOT.MakeHipoDataFrame(fname)
    cols = ["REC_Particle_px", "REC_Particle_py", "REC_Particle_pz", "REC_Particle_pid"]
    npdict = df.AsNumpy(columns=cols)
    ak_cols = {c: ak.from_iter(npdict[c]) for c in cols}
    particles = ak.zip(ak_cols)

    pdf = ak.to_dataframe(particles)
    pdf.columns = ["px", "py", "pz", "pid"]
    df_gamma = pdf.loc[pdf.pid == 22, :]
    df_gamma.reset_index(inplace=True)
    if df_gamma.empty:
        return None, None

    g = df_gamma.groupby("entry")
    df_gamma["multiplicity"] = g["subentry"].transform("count")
    df_gamma.loc[:, "subentry"] = g.cumcount()

    df2 = df_gamma[df_gamma.multiplicity == 2].copy()
    if df2.empty:
        return None, None

    piv = df2.pivot(index="entry", columns="subentry", values=["px", "py", "pz"])
    px1, px2 = piv["px"][0], piv["px"][1]
    py1, py2 = piv["py"][0], piv["py"][1]
    pz1, pz2 = piv["pz"][0], piv["pz"][1]

    E1 = np.sqrt(px1**2 + py1**2 + pz1**2)
    E2 = np.sqrt(px2**2 + py2**2 + pz2**2)
    dot = px1 * px2 + py1 * py2 + pz1 * pz2
    cos_theta = np.clip(dot / (E1 * E2), -1.0, 1.0)

    opening_angle = np.degrees(np.arccos(cos_theta))
    inv_mass = np.sqrt(np.clip(2.0 * E1 * E2 * (1.0 - cos_theta), 0, None))
    return inv_mass, opening_angle

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

  for fname in sample_files:
    inv_mass, opening_angle = process_file(fname)
    if inv_mass is None:
      print(f"  {fname}: no gamma-gamma mult-2 events, skipping")
      continue

    hists[period_name]["h_mass"]  += np.histogram(inv_mass, bins=mass_bins)[0]
    hists[period_name]["h_angle"] += np.histogram(opening_angle, bins=angle_bins)[0]
    hists[period_name]["h_2d"]    += np.histogram2d(opening_angle, inv_mass, bins=[angle_bins, mass_bins])[0]
    hists[period_name]["files_used"].append(fname)

    hists[period_name]["pi0_candidate"] += len(inv_mass[(inv_mass<0.2) & (inv_mass>=0.1)])
    hists[period_name]["pi0_sideband"]  += len(inv_mass[(inv_mass<0.1) & (inv_mass>=0.05)])
    hists[period_name]["pi0_sideband"]  += len(inv_mass[(inv_mass<0.25) & (inv_mass>=0.2)])
    hists[period_name]["eta_candidate"] += len(inv_mass[(inv_mass<0.65) & (inv_mass>=0.45)])
    hists[period_name]["eta_sideband"]  += len(inv_mass[(inv_mass<0.4) & (inv_mass>=0.45)])
    hists[period_name]["eta_sideband"]  += len(inv_mass[(inv_mass<0.7) & (inv_mass>=0.65)])
    print(f"  {fname}: {len(inv_mass)} pairs")


  h_mass  = hists[period_name]["h_mass"]
  h_angle = hists[period_name]["h_angle"]
  h_2d    = hists[period_name]["h_2d"]

  if h_mass.sum() == 0:
    print(f"[{period_name}] no events accumulated, skipping plot")
    continue

  fig, axs = plt.subplots(1, 2, figsize=(12, 4))
  fig.suptitle(period_name)

  axs[0].stairs(h_mass, mass_bins, color = 'k')
  axs[0].set_xlabel("Invariant mass [GeV]")
  axs[0].set_ylabel("Counts")
  axs[0].set_title(r"$\gamma\gamma$ invariant mass")
  axs[0].axvline(0.050, color="red", linestyle=":", linewidth=1)
  axs[0].axvline(0.100, color="red", linestyle="--", linewidth=1)
  axs[0].axvline(0.200, color="red", linestyle="--", linewidth=1)
  axs[0].axvline(0.250, color="red", linestyle=":", linewidth=1)
  axs[0].axvline(0.400, color="blue", linestyle=":", linewidth=1)
  axs[0].axvline(0.450, color="blue", linestyle="--", linewidth=1)
  axs[0].axvline(0.650, color="blue", linestyle="--", linewidth=1)
  axs[0].axvline(0.70, color="blue", linestyle=":", linewidth=1)


  #axs[1].stairs(h_angle, angle_bins, color = 'k')
  #axs[1].set_xlabel("Opening angle [deg]")
  #axs[1].set_ylabel("Counts")
  #axs[1].set_title(r"$\gamma\gamma$ opening angle")

  nonzero = h_2d[h_2d > 0]
  im = axs[1].pcolormesh(mass_bins, angle_bins, h_2d, norm=LogNorm(vmin=nonzero.min(), vmax=h_2d.max()), shading="auto", rasterized =True)
  axs[1].set_ylabel("Opening angle [deg]")
  axs[1].set_xlabel("Invariant mass [GeV]")
  axs[1].set_title("Mass vs opening angle")
  fig.colorbar(im, ax=axs[1])

  plt.tight_layout()
  plt.savefig(f"figures/gammagamma_{period_name}.png", dpi=150)
  plt.close(fig)  # close each fig so 5 periods don't pile up open figures in memory

np.savez("gammagamma_hists_by_period.npz",
         mass_bins=mass_bins, angle_bins=angle_bins,
         **{f"{p}_{k}": v for p in hists for k, v in hists[p].items() if k != "files_used"})

for period_name in periods:
  hists[period_name]["pi0_only"] = hists[period_name]["pi0_candidate"] - hists[period_name]["pi0_sideband"]
  hists[period_name]["eta_only"] = hists[period_name]["eta_candidate"] - hists[period_name]["eta_sideband"]
  print("{} {} {} {:.3f}".format(period_name, hists[period_name]["pi0_only"], hists[period_name]["eta_only"], hists[period_name]["eta_only"]/hists[period_name]["pi0_only"]))

print("Saved per-period figures to figures/")
