"""Collect pi0 and eta candidates from one CLAS12 HIPO file.

Usage
  python find_eta.py <input.hipo> <output>

Writes one parquet file per candidate type (pickle if pyarrow is missing)
  <output>_pi0.parquet            2 photons, m(gg) in 0.05 to 0.25 GeV
  <output>_eta_gg.parquet         2 photons, m(gg) in 0.40 to 0.70 GeV
  <output>_eta_3pi0.parquet       6 photons, m(6 gamma) in 0.40 to 0.70 GeV
  <output>_eta_pippimpi0.parquet  2 photons + pi+ + pi-, m(gg) in 0.05 to 0.25 GeV, m(system) in 0.40 to 0.70 GeV
  <output>_eta_pippimg.parquet    1 photon + pi+ + pi-, m(system) in 0.40 to 0.70 GeV
All five files are always written, empty ones included. The eta_pippimg file is written last.

Columns in every file
  file       HIPO file name
  event      RUN::config event number (integer), use it to cross-reference with other outputs
  entry      position of the event in the file
  region     signal, sideband_low or sideband_high
  mass       invariant mass of the candidate
  px py pz E four-momentum of the candidate (sum of its particles)
  n_gamma n_pip n_pim   photon and pion multiplicities of the whole event, to veto overlapping topologies
  g1_px g1_py g1_pz ... measured momentum of each particle in the candidate
                        (g1..gN photons, pip, pim), no corrections applied
Extra columns per file
  pi0, eta_gg     opening_angle (deg)
  eta_3pi0        m_pi0_a/b/c and pairing_chi2 of the best pairing of the 6 photons into 3 pi0.
                  Photons are ordered by pair, (g1,g2) (g3,g4) (g5,g6), with the pair masses ascending,
                  so m(g1,g2) = m_pi0_a and so on.
  eta_pippimpi0   m_gg, m_pipi
  eta_pippimg     m_pipi
"""
import argparse
import os

import numpy as np
import pandas as pd
import awkward as ak


M_PI0 = 0.134977
M_PIC = 0.139570

# (low, high) of the signal window, and the outer edge of the sideband on each side
PI0_SIGNAL = (0.10, 0.20)
PI0_OUTER = (0.05, 0.25)
ETA_SIGNAL = (0.45, 0.65)
ETA_OUTER = (0.40, 0.70)


def _cols(*labels):
    return [f"{lab}_{c}" for lab in labels for c in ("px", "py", "pz")]


_BASE = ["file", "event", "entry", "region", "mass", "px", "py", "pz", "E", "n_gamma", "n_pip", "n_pim"]

# one entry per output file, in the order they are written
COLUMNS = {
    "pi0":           _BASE + _cols("g1", "g2") + ["opening_angle"],
    "eta_gg":        _BASE + _cols("g1", "g2") + ["opening_angle"],
    "eta_3pi0":      _BASE + _cols("g1", "g2", "g3", "g4", "g5", "g6")
                     + ["m_pi0_a", "m_pi0_b", "m_pi0_c", "pairing_chi2"],
    "eta_pippimpi0": _BASE + _cols("g1", "g2", "pip", "pim") + ["m_gg", "m_pipi"],
    "eta_pippimg":   _BASE + _cols("g1", "pip", "pim") + ["m_pipi"],
}
STRING_COLUMNS = {"file", "region"}
INT_COLUMNS = {"event", "entry", "n_gamma", "n_pip", "n_pim"}


def _pairings(items):
    """All ways to split a list of even length into unordered pairs."""
    if not items:
        yield []
        return
    first = items[0]
    for i in range(1, len(items)):
        rest = items[1:i] + items[i + 1:]
        for p in _pairings(rest):
            yield [(first, items[i])] + p


PAIRINGS_6 = list(_pairings(list(range(6))))                 # 15 pairings of 6 photons
PAIR_A = np.array([[a for a, b in p] for p in PAIRINGS_6])   # (15, 3) first photon of each pair
PAIR_B = np.array([[b for a, b in p] for p in PAIRINGS_6])   # (15, 3) second photon of each pair


def _minv(E, px, py, pz):
    return np.sqrt(np.clip(E**2 - px**2 - py**2 - pz**2, 0, None))


def _region(mass, signal, outer):
    """Label as signal / sideband_low / sideband_high, or None outside the outer window."""
    out = np.full(mass.shape, None, dtype=object)
    out[(mass >= outer[0]) & (mass < signal[0])] = "sideband_low"
    out[(mass >= signal[0]) & (mass < signal[1])] = "signal"
    out[(mass >= signal[1]) & (mass < outer[1])] = "sideband_high"
    return out


def _take(arr, evsel, n):
    """Rows of a jagged array with exactly n entries per selected event, as (nev, n)."""
    flat = np.asarray(ak.flatten(arr[evsel]))
    return flat.reshape(-1, n)


def _empty(name):
    """Zero-row table with the right column types, so files from different inputs always concatenate."""
    def dtype(c):
        return "string" if c in STRING_COLUMNS else "int64" if c in INT_COLUMNS else "float64"
    return pd.DataFrame({c: pd.Series(dtype=dtype(c)) for c in COLUMNS[name]})


def _table(name, fname, keep, event, entry, region, mass, vec, parts, counts, extra=None):
    """Rows where keep is True. parts maps a label (g1, pip, ...) to its (px, py, pz) arrays."""
    idx = np.nonzero(keep)[0]
    if not len(idx):
        return _empty(name)
    d = {
        "file": fname, "event": event[idx], "entry": entry[idx],
        "region": region[idx], "mass": mass[idx],
        "px": vec[0][idx], "py": vec[1][idx], "pz": vec[2][idx], "E": vec[3][idx],
        "n_gamma": counts[0][idx], "n_pip": counts[1][idx], "n_pim": counts[2][idx],
    }
    for label, (a, b, c) in parts.items():
        d[label + "_px"], d[label + "_py"], d[label + "_pz"] = a[idx], b[idx], c[idx]
    for key, v in (extra or {}).items():
        d[key] = v[idx]
    return pd.DataFrame(d)[COLUMNS[name]]


def _photon_parts(gpx, gpy, gpz):
    return {f"g{i + 1}": (gpx[:, i], gpy[:, i], gpz[:, i]) for i in range(gpx.shape[1])}


def build_candidates(px, py, pz, pid, event, fname):
    """Return {name: DataFrame} with one table per output file (the keys of COLUMNS).

    px, py, pz, pid are jagged awkward arrays, one list per event.
    event is the RUN::config event number, one integer per event.

    Selections (all on REC::Particle, no other particle restrictions)
      pi0, eta_gg     exactly 2 photons
      eta_3pi0        exactly 6 photons
      eta_pippimpi0   exactly 2 photons, 1 pi+, 1 pi-
      eta_pippimg     exactly 1 photon, 1 pi+, 1 pi-
    n_gamma, n_pip, n_pim are kept so overlapping topologies can be vetoed later
    (a 2 photon + pi+ pi- event is in eta_pippimpi0 and also in pi0 and eta_gg if the masses fit).
    """
    event = np.asarray(event, dtype=np.int64)
    is_g = pid == 22
    is_pp = pid == 211
    is_pm = pid == -211
    n_g = np.asarray(ak.sum(is_g, axis=1))
    n_pp = np.asarray(ak.sum(is_pp, axis=1))
    n_pm = np.asarray(ak.sum(is_pm, axis=1))

    tables = {name: _empty(name) for name in COLUMNS}

    def photons(evsel, n):
        gpx, gpy, gpz = (_take(a[is_g], evsel, n) for a in (px, py, pz))
        return gpx, gpy, gpz, np.sqrt(gpx**2 + gpy**2 + gpz**2)

    def charged(mask, evsel):
        cpx, cpy, cpz = (_take(a[mask], evsel, 1)[:, 0] for a in (px, py, pz))
        return cpx, cpy, cpz, np.sqrt(cpx**2 + cpy**2 + cpz**2 + M_PIC**2)

    # ---- two photons: pi0 candidates and eta -> gg ----
    ev = n_g == 2
    if ev.any():
        entry, evn = np.nonzero(ev)[0], event[ev]
        gpx, gpy, gpz, gE = photons(ev, 2)
        vec = (gpx.sum(1), gpy.sum(1), gpz.sum(1), gE.sum(1))
        mass = _minv(vec[3], vec[0], vec[1], vec[2])
        cos_t = np.clip((gpx[:, 0] * gpx[:, 1] + gpy[:, 0] * gpy[:, 1] + gpz[:, 0] * gpz[:, 1])
                        / (gE[:, 0] * gE[:, 1]), -1.0, 1.0)
        opening = np.degrees(np.arccos(cos_t))
        counts = (n_g[ev], n_pp[ev], n_pm[ev])
        parts = _photon_parts(gpx, gpy, gpz)
        for name, signal, outer in (("pi0", PI0_SIGNAL, PI0_OUTER), ("eta_gg", ETA_SIGNAL, ETA_OUTER)):
            reg = _region(mass, signal, outer)
            tables[name] = _table(name, fname, reg != None, evn, entry, reg, mass, vec,  # noqa: E711
                                  parts, counts, {"opening_angle": opening})

    # ---- 3 pi0 (6 photons) ----
    ev = n_g == 6
    if ev.any():
        entry, evn = np.nonzero(ev)[0], event[ev]
        gpx, gpy, gpz, gE = photons(ev, 6)
        n = len(entry)
        vec = (gpx.sum(1), gpy.sum(1), gpz.sum(1), gE.sum(1))
        mass = _minv(vec[3], vec[0], vec[1], vec[2])
        reg = _region(mass, ETA_SIGNAL, ETA_OUTER)
        # pairwise photon masses, m_ij^2 = 2 E_i E_j (1 - cos theta_ij) = 2 (E_i E_j - p_i . p_j)
        dot = (gpx[:, :, None] * gpx[:, None, :] + gpy[:, :, None] * gpy[:, None, :]
               + gpz[:, :, None] * gpz[:, None, :])
        mij = np.sqrt(np.clip(2.0 * (gE[:, :, None] * gE[:, None, :] - dot), 0, None))
        chi2 = ((mij[:, PAIR_A, PAIR_B] - M_PI0) ** 2).sum(axis=2)      # (n, 15)
        best = chi2.argmin(1)
        rows = np.arange(n)
        sel_a, sel_b = PAIR_A[best], PAIR_B[best]                       # (n, 3) photon indices
        pair_mass = mij[rows[:, None], sel_a, sel_b]                    # (n, 3)
        order = np.argsort(pair_mass, axis=1)                           # pairs by ascending mass
        sel_a, sel_b = np.take_along_axis(sel_a, order, 1), np.take_along_axis(sel_b, order, 1)
        pair_mass = np.take_along_axis(pair_mass, order, 1)
        idx6 = np.stack([sel_a, sel_b], axis=2).reshape(n, 6)           # g1 g2 | g3 g4 | g5 g6
        gpx, gpy, gpz = (np.take_along_axis(a, idx6, 1) for a in (gpx, gpy, gpz))
        tables["eta_3pi0"] = _table(
            "eta_3pi0", fname, reg != None, evn, entry, reg, mass, vec,  # noqa: E711
            _photon_parts(gpx, gpy, gpz), (n_g[ev], n_pp[ev], n_pm[ev]),
            {"m_pi0_a": pair_mass[:, 0], "m_pi0_b": pair_mass[:, 1], "m_pi0_c": pair_mass[:, 2],
             "pairing_chi2": chi2[rows, best]})

    # ---- pi+ pi- pi0 (2 photons + pi+ + pi-) ----
    ev = (n_g == 2) & (n_pp == 1) & (n_pm == 1)
    if ev.any():
        entry, evn = np.nonzero(ev)[0], event[ev]
        gpx, gpy, gpz, gE = photons(ev, 2)
        ppx, ppy, ppz, pE = charged(is_pp, ev)
        mpx, mpy, mpz, mE = charged(is_pm, ev)
        m_gg = _minv(gE.sum(1), gpx.sum(1), gpy.sum(1), gpz.sum(1))
        vec = (gpx.sum(1) + ppx + mpx, gpy.sum(1) + ppy + mpy,
               gpz.sum(1) + ppz + mpz, gE.sum(1) + pE + mE)
        mass = _minv(vec[3], vec[0], vec[1], vec[2])
        m_pipi = _minv(pE + mE, ppx + mpx, ppy + mpy, ppz + mpz)
        reg = _region(mass, ETA_SIGNAL, ETA_OUTER)
        keep = (reg != None) & (m_gg >= PI0_OUTER[0]) & (m_gg < PI0_OUTER[1])  # noqa: E711
        parts = {**_photon_parts(gpx, gpy, gpz), "pip": (ppx, ppy, ppz), "pim": (mpx, mpy, mpz)}
        tables["eta_pippimpi0"] = _table(
            "eta_pippimpi0", fname, keep, evn, entry, reg, mass, vec, parts,
            (n_g[ev], n_pp[ev], n_pm[ev]), {"m_gg": m_gg, "m_pipi": m_pipi})

    # ---- pi+ pi- gamma ----
    ev = (n_g == 1) & (n_pp == 1) & (n_pm == 1)
    if ev.any():
        entry, evn = np.nonzero(ev)[0], event[ev]
        gpx, gpy, gpz, gE = photons(ev, 1)
        ppx, ppy, ppz, pE = charged(is_pp, ev)
        mpx, mpy, mpz, mE = charged(is_pm, ev)
        vec = (gpx[:, 0] + ppx + mpx, gpy[:, 0] + ppy + mpy,
               gpz[:, 0] + ppz + mpz, gE[:, 0] + pE + mE)
        mass = _minv(vec[3], vec[0], vec[1], vec[2])
        m_pipi = _minv(pE + mE, ppx + mpx, ppy + mpy, ppz + mpz)
        reg = _region(mass, ETA_SIGNAL, ETA_OUTER)
        parts = {**_photon_parts(gpx, gpy, gpz), "pip": (ppx, ppy, ppz), "pim": (mpx, mpy, mpz)}
        tables["eta_pippimg"] = _table(
            "eta_pippimg", fname, reg != None, evn, entry, reg, mass, vec, parts,  # noqa: E711
            (n_g[ev], n_pp[ev], n_pm[ev]), {"m_pipi": m_pipi})

    return tables


def init_root():
    import ROOT
    ROOT.gInterpreter.AddIncludePath("/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/hipo/4.5.0/root_6.38.04/include")
    ROOT.gInterpreter.ProcessLine('#include "hipo4/RHipoDS.hxx"')
    ROOT.DisableImplicitMT()
    return ROOT


def _per_event_int(col):
    """One integer per event, whether the column comes as plain numbers or as one-element vectors."""
    a = np.asarray(col)
    if a.dtype != object:
        return a.astype(np.int64)
    a = ak.from_iter(col)
    if a.ndim > 1:
        a = ak.fill_none(ak.firsts(a, axis=1), -1)
    return np.asarray(ak.to_numpy(a)).astype(np.int64)


def process_file(ROOT, fname):
    """Returns {name: DataFrame} candidate tables for one file."""
    df = ROOT.MakeHipoDataFrame(fname)
    cols = ["REC_Particle_px", "REC_Particle_py", "REC_Particle_pz", "REC_Particle_pid"]
    npdict = df.AsNumpy(columns=cols + ["RUN_config_event"])
    px, py, pz, pid = (ak.from_iter(npdict[c]) for c in cols)
    event = _per_event_int(npdict["RUN_config_event"])
    if len(event) != len(px):
        raise RuntimeError(f"{fname}: {len(event)} RUN::config events but {len(px)} REC::Particle events")
    return build_candidates(px, py, pz, pid, event, os.path.basename(fname))


def save_df(df, base):
    try:
        path = base + ".parquet"
        df.to_parquet(path)
    except ImportError:  # no pyarrow/fastparquet
        path = base + ".pkl"
        df.to_pickle(path)
    print(f"saved {len(df)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input HIPO file")
    ap.add_argument("output", help="output prefix, gives <output>_pi0.parquet, <output>_eta_gg.parquet, ...")
    args = ap.parse_args()

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    ROOT = init_root()
    tables = process_file(ROOT, args.input)
    print(f"{args.input}: " + ", ".join(f"{name} {len(df)}" for name, df in tables.items()))
    for name, df in tables.items():  # dict order, eta_pippimg last
        save_df(df, f"{args.output}_{name}")


if __name__ == "__main__":
    main()
