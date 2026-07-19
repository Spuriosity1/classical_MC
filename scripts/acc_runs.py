#!/usr/bin/env python3

import re
import sys
import os
import glob
import argparse
from collections import defaultdict

import numpy as np
import h5py

# Usage: acc_runs.py <dir_or_file> [<dir_or_file> ...]
#
# Walks the given directories (and/or individual files), collects every
# *.out.h5 run file it finds, and groups them by their filename tags
# (key=value segments, e.g. "L=8", "J1=1.0", "spiral_axis=2") ignoring
# "seed" — runs that agree on every other tag are assumed to be repeats of
# the same physical run with different RNG seeds. Each group is merged
# across seeds and the cross-seed variance of the energy per primitive
# cell, e = E_total / N_primitive_cells, is computed.
#
# Assumes all seeds use identical --n_sample (uniform n_per_seed per temperature).
#
# Writes one <params>_mergeK_.avg.h5 per group (merged over K seeds), into
# the same directory as the source files. Any pre-existing *.avg.h5 files
# found among the given paths are flagged and, with confirmation, deleted
# before new merges are written.
#
# Datasets in /energy:
#   T_list    — temperature grid
#   E         — sum of e over seeds (e = energy per primitive cell)
#   E2        — sum of e² over seeds
#   n_samples — total sample count (= K, since each seed contributes 1 per T)
#   e_mean    — cross-seed mean energy per primitive cell
#   e2_mean   — cross-seed mean of e²
#   var       — cross-seed variance: e2_mean - e_mean²
#
# /geometry:
#   index_cell
#   recip_vectors
#
# Datasets in /ssf:
#   static_corr      — Σ_seeds raw sum of per-sample correlators [n_corr,n_T,n_k,2]
#   static_corr_2    — Σ_seeds (per-seed raw sum)²  [same shape]  ← inter-seed 2nd moment
#   n_samples        — total sample count across seeds [n_T]
#   var_inter        — inter-seed variance of C(k): Var_seeds[mean_C_seed] [same shape]
#   var_intra        — sum of per-seed intrinsic variances of C(k) [same shape]
#                      (absent if no seed file contained static_corr_2)
#   T_list, corr_lookup, sl_positions, attrs — copied from first file
#
# Sublattice contraction is done in C++ (ssf_manager), so on-disk corr arrays are
# [n_corr, n_T, n_k, 2]; the last dim is [Re, Im], stored independently.
#
# Heat capacity per spin: C/N = var * N_prim / (T² * N_spins)
#                              = var / (T² * 16)
# is left to post-processing.


def load_energy_raw(fname):
    """Return T_list, E_sum, E2_sum, n_samples (raw sums from energy_manager)."""
    with h5py.File(fname, "r") as f:
        g = f["/energy"]
        assert isinstance(g, h5py.Group)
        T  = np.array(g["T_list"])
        E  = np.array(g["E"])
        E2 = np.array(g["E2"])
        n  = np.array(g["n_samples"])
    return T, E, E2, n

def load_geometry(fname):
    with h5py.File(fname, "r") as f:
        g= f["/geometry"]
        assert isinstance(g, h5py.Group)
        idx_cell = np.array(g["index_cell"])
        recip_vectors = np.array(g["recip_vectors"])

    return idx_cell, recip_vectors


"""
expected layout:
/ssf/T_list              Dataset {1}
/ssf/corr_lookup         Dataset {3}
/ssf/n_samples           Dataset {1}
/ssf/sl_positions        Dataset {16, 3}
/ssf/static_corr         Dataset {3, 1, 512, 16, 16, 2}
"""
def load_ssf_raw(fname):
    with h5py.File(fname, "r") as f:
        g= f["/ssf"]
        assert isinstance(g, h5py.Group)
        T  = np.array(g["T_list"])
        corr_lookup = np.array(g["corr_lookup"])
        n_samples = np.array(g["n_samples"])
        sl_pos = np.array(g["sl_positions"])
        static_corr  = np.array(g["static_corr"])
        static_corr_2 = np.array(g["static_corr_2"]) if "static_corr_2" in g else None
        attrs = {k: v for k, v in g.attrs.items()}

    return T, corr_lookup, n_samples, sl_pos, static_corr, static_corr_2, attrs

# key=value segments in run filenames. Keys are normally a single
# identifier word ("L", "J1", "Tc", ...), but "spiral_axis" is the one
# multi-word exception emitted by cli_bits.hpp, so it's special-cased here.
# Values never contain "_" (that's the segment separator), so a value runs
# up to the next "_" (or end of string).
_TAG_RE = re.compile(r"(spiral_axis|[A-Za-z][A-Za-z0-9]*)=([^_]*)")


def parse_tags(fname):
    """Parse all key=value tags present in a run filename into a dict of
    raw strings. Free text that isn't part of a key=value pair (e.g. the
    user-supplied --prefix) is ignored."""
    base = os.path.basename(fname)
    base = re.sub(r"\.(out|avg)\.h5$", "", base)
    tags = {m.group(1): m.group(2) for m in _TAG_RE.finditer(base)}
    return tags


def _normalize(value):
    """Numeric tags should compare equal regardless of formatting
    (e.g. "0.10" vs "0.1"); everything else compares as a raw string."""
    try:
        return float(value)
    except ValueError:
        return value


def group_key(tags, ignore=("seed",)):
    """A hashable key identifying runs that differ only in `ignore` tags."""
    return tuple(sorted(
        (k, _normalize(v)) for k, v in tags.items() if k not in ignore
    ))


def group_runs(fnames):
    """Group run files by their non-seed tags. Files without a seed= tag
    are skipped (they're not seed-repeated runs)."""
    groups = defaultdict(list)
    for fname in fnames:
        tags = parse_tags(fname)
        if "seed" not in tags:
            print(f"Warning: no seed= tag in {os.path.basename(fname)}, skipping",
                  file=sys.stderr)
            continue
        groups[group_key(tags)].append(fname)
    return groups


def merged_name(representative_file, n_seeds):
    """Replace seed=<hex>_ with mergeN_ and change extension to .avg.h5."""
    base = os.path.basename(representative_file)
    out = re.sub(r"seed=[0-9a-fA-F]+_", f"merge{n_seeds}_", base)
    out = re.sub(r"\.out\.h5$", ".avg.h5", out)
    if out == base:
        raise ValueError(f"Could not generate output name from: {base}")
    return os.path.join(os.path.dirname(representative_file), out)


def merge_group(fnames):
    """Merge one group of seed-repeated run files into a single .avg.h5.
    Returns the output path."""
    T_ref    = None
    E_total  = None
    E2_total = None
    n_total  = None

    # accumulate the energies
    for fname in fnames:
        T, E, E2, n = load_energy_raw(fname)
        if T_ref is None:
            T_ref    = T
            E_total  = E.copy()
            E2_total = E2.copy()
            n_total  = n.copy()
        else:
            if not np.allclose(T, T_ref):
                raise ValueError(f"T_list mismatch in {fname}")
            E_total  += E
            E2_total += E2
            n_total  += n

    # accumulate the ssf's
    T_ref_ssf = None
    corr_lookup_ref=None
    sl_positions_ref=None
    ssf_attrs_ref = None

    total_static_corr = None
    total_static_corr_2 = None   # Σ_seeds C_s²  (inter-seed 2nd moment)
    total_intrinsic_corr_2 = None  # Σ_seeds static_corr_2[seed]  (intra-seed 2nd moment)
    total_static_samp = None

    for fname in fnames:
        T, corr_lookup, n_samp, sl_pos, static_corr, static_corr_2, attrs = load_ssf_raw(fname)
        if T_ref_ssf is None:
            T_ref_ssf = T
            corr_lookup_ref = corr_lookup
            sl_positions_ref = sl_pos
            ssf_attrs_ref = attrs

            total_static_corr = static_corr.copy()
            total_static_corr_2 = static_corr**2
            total_static_samp = n_samp.copy()
            total_intrinsic_corr_2 = static_corr_2.copy() if static_corr_2 is not None else None
        else:
            if not np.allclose(T_ref_ssf, T):
                raise RuntimeError(f"Temperature range of file {fname} incompatible")
            if np.any(corr_lookup_ref != corr_lookup):
                raise RuntimeError(f"Stored correlators of file {fname} incompatible")
            if not np.allclose(sl_positions_ref, sl_pos):
                raise RuntimeError(f"Sublattice convention incompatible in file {fname}")
            attr_mismatches = [
                f"{k}: {ssf_attrs_ref[k]!r} vs {attrs[k]!r}"
                for k in ssf_attrs_ref
                if k not in attrs or not np.array_equal(ssf_attrs_ref[k], attrs[k])
            ] + [f"{k}: missing in reference" for k in attrs if k not in ssf_attrs_ref]
            if attr_mismatches:
                raise RuntimeError(
                    f"SSF group attribute mismatch in {fname}: "
                    + ", ".join(attr_mismatches)
                )

            total_static_corr += static_corr
            total_static_corr_2 += static_corr**2
            total_static_samp += n_samp
            if total_intrinsic_corr_2 is not None:
                if static_corr_2 is not None:
                    total_intrinsic_corr_2 += static_corr_2
                else:
                    print(f"Warning: {fname} has no static_corr_2; dropping var_intra",
                          file=sys.stderr)
                    total_intrinsic_corr_2 = None
            

    K = len(fnames)
    e_mean  = E_total  / n_total
    e2_mean = E2_total / n_total
    var     = e2_mean - e_mean**2

    # SSF variance terms.
    # Assumes uniform n_per_seed across seeds; n_per_seed[t] = total_samp[t] / K.
    # Both var arrays have shape [n_corr, n_T, n_k, 2] where the last dimension
    # stores [Re variance, Im variance] independently.
    n_per_seed = total_static_samp / K   # [n_T]
    # broadcast over [n_corr, n_T, n_k, 2]
    n_ = n_per_seed[np.newaxis, :, np.newaxis, np.newaxis]

    # Inter-seed variance: Var_seeds[mean_C_seed]
    #   = (1/K) Σ_s (C_s/n)² − ((1/K) Σ_s C_s/n)²
    #   = total_corr2/(K·n²) − (total_corr/(K·n))²
    ssf_var_inter = (total_static_corr_2 / (K * n_**2)
                     - (total_static_corr / (K * n_))**2)

    # Sum of per-seed intrinsic variances: Σ_s Var_intra_s
    #   = Σ_s [C2_s/n − (C_s/n)²]
    #   = total_intr_C2/n − total_corr2/n²
    if total_intrinsic_corr_2 is not None:
        ssf_var_intra = total_intrinsic_corr_2 / n_ - total_static_corr_2 / n_**2
    else:
        ssf_var_intra = None

    idx, rcv = load_geometry(fnames[0])

    out = merged_name(fnames[0], K)
    with h5py.File(out, "w") as f:
        g = f.create_group("energy")
        g.create_dataset("T_list",    data=T_ref)
        g.create_dataset("E",         data=E_total)
        g.create_dataset("E2",        data=E2_total)
        g.create_dataset("n_samples", data=n_total)
        g.create_dataset("e_mean",    data=e_mean)
        g.create_dataset("e2_mean",   data=e2_mean)
        g.create_dataset("var",       data=var)
        g.create_dataset("n_seeds",   data=np.uint64(K))

        gg = f.create_group("geometry")
        gg.create_dataset("index_cell",  data=idx)
        gg.create_dataset("recip_vectors", data=rcv)

        sg = f.create_group("ssf")
        sg.create_dataset("T_list",        data=T_ref_ssf)
        sg.create_dataset("corr_lookup",   data=corr_lookup_ref)
        sg.create_dataset("n_samples",     data=total_static_samp)
        sg.create_dataset("static_corr",   data=total_static_corr)
        sg.create_dataset("static_corr_2", data=total_static_corr_2)
        sg.create_dataset("var_inter",     data=ssf_var_inter)
        if ssf_var_intra is not None:
            sg.create_dataset("var_intra", data=ssf_var_intra)
        sg.create_dataset("sl_positions",  data=sl_positions_ref)
        for k, v in ssf_attrs_ref.items():
            sg.attrs[k] = v

    print(f"{K} seed(s) merged → {out}")
    return out


def collect_inputs(paths):
    """Split given paths (directories and/or files) into the set of
    *.out.h5 run files and *.avg.h5 files found among them."""
    out_files = set()
    avg_files = set()
    for p in paths:
        if os.path.isdir(p):
            out_files.update(glob.glob(os.path.join(p, "*.out.h5")))
            avg_files.update(glob.glob(os.path.join(p, "*.avg.h5")))
        elif p.endswith(".out.h5"):
            out_files.add(p)
        elif p.endswith(".avg.h5"):
            avg_files.add(p)
        else:
            print(f"Warning: ignoring unrecognized path: {p}", file=sys.stderr)
    return sorted(out_files), sorted(avg_files)


def prompt_delete_avg_files(avg_files):
    if not avg_files:
        return
    print(f"Found {len(avg_files)} pre-existing .avg.h5 file(s):")
    for f in avg_files:
        print(f"  {f}")
    reply = input("Delete these before regenerating? [y/N] ").strip().lower()
    if reply in ("y", "yes"):
        for f in avg_files:
            os.remove(f)
        print(f"Deleted {len(avg_files)} file(s).")
    else:
        print("Leaving pre-existing .avg.h5 files in place.")


def main(argv):
    parser = argparse.ArgumentParser(
        description="Accumulate seed-repeated run files into per-parameter "
                     ".avg.h5 files."
    )
    parser.add_argument("paths", nargs="+",
                         help="Directories and/or .out.h5 files to process")
    args = parser.parse_args(argv)

    out_files, avg_files = collect_inputs(args.paths)

    prompt_delete_avg_files(avg_files)

    if not out_files:
        print("No .out.h5 files found.", file=sys.stderr)
        sys.exit(1)

    groups = group_runs(out_files)
    if not groups:
        print("No seed-tagged run files to merge.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(out_files)} run file(s) in {len(groups)} group(s).")
    for fnames in groups.values():
        merge_group(sorted(fnames))


if __name__ == "__main__":
    main(sys.argv[1:])
