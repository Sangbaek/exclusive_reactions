"""Collect deeply virtual K+ production candidates, ep -> e' K+ Lambda (Lambda -> p pi-), with and without
an extra photon, from one CLAS12 HIPO file.

Usage
  python find_lambda.py <input.hipo> <output> --beam-energy <GeV>

One parquet file per final state (pickle if pyarrow is missing). A particle in brackets is not required,
the Lambda is then taken from a missing mass.
  <output>_ekppim.parquet    (1) e' K+ pi- p           Lambda = p pi- invariant mass
  <output>_ekp.parquet       (2) e' K+ p (pi-)         Lambda = missing mass of e' K+
  <output>_ek.parquet        (3) e' K+ (pi- p)         Lambda = missing mass of e' K+
  <output>_ekppimg.parquet   (4) e' K+ pi- p gamma     Lambda = p pi- invariant mass
  <output>_ekpg.parquet      (5) e' K+ gamma p (pi-)   Lambda = missing mass of e' K+ gamma
  <output>_ekg.parquet       (6) e' K+ gamma (pi- p)   Lambda = missing mass of e' K+ gamma
All six files are always written, empty ones included. The ekg file is written last.
The files overlap, a (1) event is also in (2) and (3). Use n_kp, n_pr, n_pim, n_gamma to separate them.
Events with several candidates (two protons, several photons, ...) give one row per combination.

Selection
  e'   pid 11 with status < 0 (the trigger electron),  K+ pid 321,  p pid 2212,  pi- pid -211,  gamma pid 22
  (2) (3) (5) (6)   missing mass of the Lambda in 0.9 to 1.5 GeV (Lambda 1.116, Sigma0 1.193)
  (1) (4)           m(p pi-) in 0.8 to 1.3 GeV
  The target is a proton at rest and the beam travels along +z, so the beam energy is needed.
  Nothing else is cut, in particular no photon energy or angle requirement.

Columns in every file
  file         HIPO file name
  event        RUN::config event number (integer), use it to cross-reference with other outputs
  entry        position of the event in the file
  beam_energy  value given on the command line
  mass         Lambda candidate mass, from m(p pi-) for (1) (4), else the missing mass named above
  px py pz E   Lambda candidate four-momentum, same definition
  n_kp n_pr n_pim n_gamma   K+, proton, pi- and photon multiplicities of the whole event
  e_px e_py e_pz  k_px k_py k_pz   measured momenta of e' and K+, no corrections applied
  p_*, pim_*, g_*   the same for the proton, pi- and photon where the channel has them
Extra columns
  (1) mm_ek = missing mass of e' K+,  mm2_ekppim = missing mass squared of e' K+ p pi-, expect 0
  (2) mm2_ekp = missing mass squared of e' K+ p, expect the pi- mass squared 0.0195
  (4) mm_ekg = missing mass of e' K+ gamma,  mm2_ekppimg = missing mass squared of e' K+ gamma p pi-, expect 0
  (5) mm2_ekpg = missing mass squared of e' K+ gamma p, expect 0.0195
"""
import argparse
import os

import numpy as np
import pandas as pd
import awkward as ak


M_E, M_P, M_K, M_PI = 0.000511, 0.938272, 0.493677, 0.139570
MASS = {"e": M_E, "k": M_K, "p": M_P, "pim": M_PI, "g": 0.0}

MM_WINDOW = (0.9, 1.5)       # GeV, missing mass of the Lambda
LAMBDA_WINDOW = (0.8, 1.3)   # GeV, p pi- invariant mass


def _cols(*labels):
    return [f"{lab}_{c}" for lab in labels for c in ("px", "py", "pz")]


_BASE = (["file", "event", "entry", "beam_energy", "mass", "px", "py", "pz", "E",
          "n_kp", "n_pr", "n_pim", "n_gamma"] + _cols("e", "k"))

# one entry per output file, in the order they are written
COLUMNS = {
    "ekppim":  _BASE + _cols("p", "pim") + ["mm_ek", "mm2_ekppim"],
    "ekp":     _BASE + _cols("p") + ["mm2_ekp"],
    "ek":      _BASE,
    "ekppimg": _BASE + _cols("p", "pim", "g") + ["mm_ekg", "mm2_ekppimg"],
    "ekpg":    _BASE + _cols("p", "g") + ["mm2_ekpg"],
    "ekg":     _BASE + _cols("g"),
}
STRING_COLUMNS = {"file"}
INT_COLUMNS = {"event", "entry", "n_kp", "n_pr", "n_pim", "n_gamma"}


def _empty(name):
    """Zero-row table with the right column types, so files from different inputs always concatenate."""
    def dtype(c):
        return "string" if c in STRING_COLUMNS else "int64" if c in INT_COLUMNS else "float64"
    return pd.DataFrame({c: pd.Series(dtype=dtype(c)) for c in COLUMNS[name]})


def _flat(x):
    return np.asarray(ak.flatten(x))


def _unpack(group):
    """Flat numpy arrays of one particle in every combination, momenta in double precision."""
    d = {f: _flat(group[f]).astype(np.float64) for f in ("px", "py", "pz")}
    d.update({f: _flat(group[f]) for f in ("entry", "event")})
    return d


def _p4(d, m):
    """(E, px, py, pz)"""
    return (np.sqrt(d["px"]**2 + d["py"]**2 + d["pz"]**2 + m**2), d["px"], d["py"], d["pz"])


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _m2(v):
    return v[0]**2 - v[1]**2 - v[2]**2 - v[3]**2


def _msqrt(m2):
    return np.sqrt(np.clip(m2, 0, None))


def _missing(init, c, labels):
    """Beam + target minus the listed particles."""
    v = init
    for lab in labels:
        v = _sub(v, _p4(c[lab], MASS[lab]))
    return v


def _table(name, fname, keep, ent, evn, beam, mass, cand, counts, parts, extra=None):
    """Rows where keep is True. cand is the (E, px, py, pz) of the Lambda candidate,
    counts the per-event (n_kp, n_pr, n_pim, n_gamma), parts maps a label (e, k, ...) to its flat momentum dict."""
    idx = np.nonzero(keep)[0]
    if not len(idx):
        return _empty(name)
    ev = ent[idx]
    d = {
        "file": fname, "event": evn[idx], "entry": ev, "beam_energy": beam, "mass": mass[idx],
        "px": cand[1][idx], "py": cand[2][idx], "pz": cand[3][idx], "E": cand[0][idx],
        "n_kp": counts[0][ev], "n_pr": counts[1][ev], "n_pim": counts[2][ev], "n_gamma": counts[3][ev],
    }
    for label, p in parts.items():
        d[label + "_px"], d[label + "_py"], d[label + "_pz"] = p["px"][idx], p["py"][idx], p["pz"][idx]
    for key, v in (extra or {}).items():
        d[key] = v[idx]
    return pd.DataFrame(d)[COLUMNS[name]]


def build_candidates(px, py, pz, pid, status, event, beam_energy, fname):
    """Return {name: DataFrame} with one table per output file (the keys of COLUMNS).

    px, py, pz, pid, status are jagged awkward arrays, one list per event.
    event is the RUN::config event number, one integer per event.
    """
    n = len(px)
    tables = {name: _empty(name) for name in COLUMNS}
    if n == 0:
        return tables
    event = np.asarray(event, dtype=np.int64)
    entry_j = ak.broadcast_arrays(ak.Array(np.arange(n)), px)[0]
    event_j = ak.broadcast_arrays(ak.Array(event), px)[0]
    P = ak.zip({"px": px, "py": py, "pz": pz, "pid": pid, "status": status,
                "entry": entry_j, "event": event_j})
    counts = tuple(np.asarray(ak.sum(pid == code, axis=1)) for code in (321, 2212, -211, 22))

    groups = {
        "e":   P[(P["pid"] == 11) & (P["status"] < 0)],
        "k":   P[P["pid"] == 321],
        "p":   P[P["pid"] == 2212],
        "pim": P[P["pid"] == -211],
        "g":   P[P["pid"] == 22],
    }
    init = (beam_energy + M_P, 0.0, 0.0, beam_energy)     # beam electron + target proton at rest

    def lam_missing(miss_labels, mm2_name=None, mm2_labels=None):
        """Lambda is what is missing after the particles in miss_labels."""
        def define(c):
            miss = _missing(init, c, miss_labels)
            extra = {mm2_name: _m2(_missing(init, c, mm2_labels))} if mm2_name else {}
            return _msqrt(_m2(miss)), miss, extra, MM_WINDOW
        return define

    def lam_invariant(mm_name, mm_labels, mm2_name, mm2_labels):
        """Lambda is the p pi- pair."""
        def define(c):
            lam = _add(_p4(c["p"], M_P), _p4(c["pim"], M_PI))
            extra = {mm_name: _msqrt(_m2(_missing(init, c, mm_labels))),
                     mm2_name: _m2(_missing(init, c, mm2_labels))}
            return _msqrt(_m2(lam)), lam, extra, LAMBDA_WINDOW
        return define

    def channel(name, labels, define):
        c = {}
        combos = ak.cartesian({lab: groups[lab] for lab in labels}, axis=1)
        for lab in labels:
            c[lab] = _unpack(combos[lab])
        mass, cand, extra, window = define(c)
        keep = (mass >= window[0]) & (mass < window[1])
        tables[name] = _table(name, fname, keep, c["e"]["entry"], c["e"]["event"], beam_energy, mass, cand,
                              counts, c, extra)

    channel("ekppim",  ["e", "k", "p", "pim"],
            lam_invariant("mm_ek", ["e", "k"], "mm2_ekppim", ["e", "k", "p", "pim"]))
    channel("ekp",     ["e", "k", "p"],
            lam_missing(["e", "k"], "mm2_ekp", ["e", "k", "p"]))
    channel("ek",      ["e", "k"],
            lam_missing(["e", "k"]))
    channel("ekppimg", ["e", "k", "p", "pim", "g"],
            lam_invariant("mm_ekg", ["e", "k", "g"], "mm2_ekppimg", ["e", "k", "g", "p", "pim"]))
    channel("ekpg",    ["e", "k", "p", "g"],
            lam_missing(["e", "k", "g"], "mm2_ekpg", ["e", "k", "g", "p"]))
    channel("ekg",     ["e", "k", "g"],
            lam_missing(["e", "k", "g"]))
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


def process_file(ROOT, fname, beam_energy):
    """Returns {name: DataFrame} candidate tables for one file."""
    df = ROOT.MakeHipoDataFrame(fname)
    cols = ["REC_Particle_px", "REC_Particle_py", "REC_Particle_pz", "REC_Particle_pid", "REC_Particle_status"]
    npdict = df.AsNumpy(columns=cols + ["RUN_config_event"])
    px, py, pz, pid, status = (ak.from_iter(npdict[c]) for c in cols)
    event = _per_event_int(npdict["RUN_config_event"])
    if len(event) != len(px):
        raise RuntimeError(f"{fname}: {len(event)} RUN::config events but {len(px)} REC::Particle events")
    return build_candidates(px, py, pz, pid, status, event, beam_energy, os.path.basename(fname))


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
    ap.add_argument("output", help="output prefix, gives <output>_ekppim.parquet, <output>_ekp.parquet, ...")
    ap.add_argument("--beam-energy", type=float, required=True, help="beam energy in GeV, see periods.py")
    args = ap.parse_args()

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    ROOT = init_root()
    tables = process_file(ROOT, args.input, args.beam_energy)
    print(f"{args.input}: " + ", ".join(f"{name} {len(df)}" for name, df in tables.items()))
    for name, df in tables.items():  # dict order, ekg last
        save_df(df, f"{args.output}_{name}")


if __name__ == "__main__":
    main()
