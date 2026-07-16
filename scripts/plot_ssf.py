#!/usr/bin/env python3
"""Plot static structure factor from anneal HDF5 output."""

import argparse
import os
import re
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_file(path):
    with h5py.File(path, "r") as f:
        recip      = f["/geometry/recip_vectors"][:]   # (3,3) rows are b0,b1,b2
        index_cell = f["/geometry/index_cell"][:]

        T_list = f["/energy/T_list"][:]
        E      = f["/energy/E"][:]
        E2     = f["/energy/E2"][:]
        n_E    = f["/energy/n_samples"][:]

        raw          = f["/ssf/static_corr"][:]   # [n_corr, n_T, n_k, 2]
        corr_lookup  = [s.decode() if isinstance(s, bytes) else s
                        for s in f["/ssf/corr_lookup"][:]]
        sl_positions = f["/ssf/sl_positions"][:]  # [n_sl, 3] int64 (metadata)
        k_dims       = f["/ssf"].attrs["k_dims"][:].astype(int)   # [3]
        n_spins      = int(f["/ssf"].attrs["n_spins"])
        ssf_T        = f["/ssf/T_list"][:]        # sorted ascending
        n_ssf        = f["/ssf/n_samples"][:]

    # static_corr is already the sublattice-contracted scalar S(q) (the phase
    # fold Σ_{mu,nu} exp(+i q·(r_mu-r_nu)) conj(Ã_mu) Ã_nu is done in the C++
    # writer), summed over MC samples.
    corr = raw[..., 0] + 1j * raw[..., 1]  # [n_corr, n_T, n_k]
    return (recip, index_cell,
            T_list, E, E2, n_E,
            corr, corr_lookup, sl_positions, k_dims, n_spins, ssf_T, n_ssf)


def normalize_ssf(corr, corr_lookup, k_dims, n_ssf):
    """Sample-normalise the contracted correlator and reshape to a k-grid.

    The sublattice phase contraction is performed in the C++ writer, so
    ``corr`` is already the scalar S(q) summed over samples.  Here we take the
    real part (S(q) is real for diagonal correlators), divide by the per-T
    sample count, and reshape flat k into (k0, k1, k2).

    Returns dict label -> array of shape [n_T, k0, k1, k2], real-valued.
    """
    contracted = np.real(corr)                       # [n_corr, n_T, n_k]
    contracted = contracted / n_ssf[np.newaxis, :, np.newaxis]

    k0, k1, k2 = k_dims
    contracted = contracted.reshape(len(corr_lookup), -1, k0, k1, k2)
    return {label: contracted[i] for i, label in enumerate(corr_lookup)}

def parse_params(path):
    """Extract key=value tokens from a filename stem into an ordered dict.

    Strips all extensions (e.g. .avg.h5).  Bare tokens are split at the
    first digit boundary: 'b512' -> {'b': '512'}, 'merge64' -> {'merge': '64'}.
    Pure-alpha bare tokens like 'ec' are stored as {'ec': ''}.
    """
    stem = re.sub(r'(\.[a-zA-Z][a-zA-Z0-9]*)+$', '', os.path.basename(path))

    params = {}
    for token in stem.split("_"):
        if not token:
            continue
        if "=" in token:
            k, v = token.split("=", 1)
            params[k] = v
        else:
            m = re.match(r'^([A-Za-z]+)(\d.*)$', token)
            if m:
                params[m.group(1)] = m.group(2)
            else:
                params[token] = ''
    return params


def split_fixed_varying(files):
    """Return (fixed, varying_keys) across all filenames.

    fixed: dict of params whose value is identical in every file.
    varying_keys: list of params that differ across files (in order of first appearance).
    """
    all_params = [parse_params(f) for f in files]
    all_keys = dict.fromkeys(k for p in all_params for k in p)
    fixed = {
        k: all_params[0][k]
        for k in all_keys
        if all(p.get(k) == all_params[0].get(k) for p in all_params)
    }
    varying_keys = [k for k in all_keys if k not in fixed]
    return fixed, varying_keys, all_params


def label_axes(ax, slice_axis=2):
    axes_l = ['yz', 'xz', 'xy']
    a1, a2 = tuple(axes_l[slice_axis])

    ax.set_xlabel(rf"$k_{a1}$")
    ax.set_ylabel(rf"$k_{a2}$")

def kgrid_xy(k_dims, recip, slice_axis=2):
    """Cartesian x,y coordinates for a 2D slice of the k-grid.

    slice_axis: which of 0,1,2 to fix (default 2 → hk0 plane).
    Returns x,y arrays of shape [na0, na1].
    """
    axes = [a for a in range(3) if a != slice_axis]
    a0, a1 = axes
    na0, na1 = k_dims[a0], k_dims[a1]

    h = np.arange(na0) - na0 // 2
    k = np.arange(na1) - na1 // 2
    hh, kk = np.meshgrid(h, k, indexing="ij")

    q = (hh[..., None] / na0 * recip[a0]
       + kk[..., None] / na1 * recip[a1])  # [na0, na1, 3]
    return q[..., a0], q[..., a1]


def plot_ssf(ax, file, args, title=""):
    (recip, index_cell,
     T_list, E, E2, n_E,
     corr, corr_lookup, sl_positions,
     k_dims, n_spins, ssf_T, n_ssf) = load_file(file)

    n_T = corr.shape[1]
    t_idx = args.t_index if args.t_index is not None else n_T - 1
    if not (0 <= t_idx < n_T):
        sys.exit(f"--t-index must be in [0, {n_T - 1}]")
    t_val = ssf_T[t_idx]

    S = normalize_ssf(corr, corr_lookup, k_dims, n_ssf)

    # Build derived observables from whatever components are present
    diag = [c for c in ("xx", "yy", "zz") if c in S]
    data_3d = sum(S[c] for c in diag)

    sa = args.slice_axis
    sl = args.slice_idx
    x, y = kgrid_xy(k_dims, recip, slice_axis=sa)

    # Build index tuple for the 2D slice
    idx = [slice(None), slice(None), slice(None)]
    idx[sa] = sl
    idx = tuple(idx)  # applied as data_3d[t_idx][idx]

    ax.set_title(title, fontsize=6)

    data = np.fft.fftshift(data_3d[t_idx][idx])
    data /= n_spins

    vmax = 10**args.vmax
    if args.vmin:
        vmin = 10**args.vmin
    else:
        vmin = max(data.min(), 1e-6 * vmax) if args.log else 0
    norm = (mcolors.LogNorm(vmin=vmin, vmax=vmax) if args.log
            else mcolors.Normalize(vmin=0, vmax=vmax))
    a0, a1 = [a for a in range(3) if a != sa]
    # For FCC: tile with the 4 nearest in-plane BCC shifts (±1,±1) so the
    # full FCC reciprocal lattice is visible beyond one conventional cubic BZ.
    shifts = ((0,0), (1,1), (1,-1), (-1,1), (-1,-1)) if args.fcc else ((0,0),)
    for p, q in shifts:
        xs = x + p * recip[a0][a0] + q * recip[a1][a0]
        ys = y + p * recip[a0][a1] + q * recip[a1][a1]
        c = ax.pcolormesh(xs, ys, data, cmap=args.cmap, norm=norm, shading="auto")

    label_axes(ax, slice_axis=sa)
    ax.set_aspect("equal")
    return c



def plot_energy(args):
    _, _, T_list, E, E2, n_E, *_ = load_file(args.file)

    E_mean  = E / n_E
    E2_mean = E2 / n_E
    C = (E2_mean - E_mean**2) / T_list**2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(T_list, E_mean, "o-", ms=3)
    ax1.set_ylabel("⟨E⟩ / site")
    ax1.invert_xaxis()

    ax2.plot(T_list, C, "o-", ms=3, color="tab:orange")
    ax2.set_ylabel("C / site")
    ax2.set_xlabel("T")

    fig.suptitle(args.file)
    plt.tight_layout()

    if args.output:
        out = (args.output.replace(".png", "_energy.png")
               if args.output.endswith(".png") else args.output + "_energy.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(
        description="Plot SSF (and optionally energy) from anneal HDF5 output.")
    p.add_argument("file", help="Path(s) to HDF5 file", nargs='+')
    p.add_argument("-t", "--t-index", type=int, default=None,
                   help="Temperature index into the SSF array (default: last = coldest)")
    p.add_argument("--slice-axis", type=int, default=2, choices=[0, 1, 2],
                   help="Which k-axis to fix for the 2D slice (default: 2 → hk0 plane)")
    p.add_argument("--slice-idx", type=int, default=0,
                   help="Index along --slice-axis to plot (default: 0)")
    p.add_argument("--log", action="store_true", help="Use logarithmic colour scale")
    p.add_argument("--cmap", default="inferno", help="Matplotlib colormap (default: inferno)")

    p.add_argument("--fcc", action="store_true",
                   help="Tile with BCC-type shifts to show full FCC reciprocal lattice")
    p.add_argument("--vmin", type=float)
    p.add_argument("--vmax", default=1, type=float)

    p.add_argument("-o", "--output", default=None,
                   help="Save figure to file instead of displaying")
    args = p.parse_args()
    
    files = args.file
    n_panels = len(files)

    fixed, varying_keys, all_params = split_fixed_varying(files)

    print(varying_keys)
    panel_titles = [
        "  ".join(f"{k}={p.get(k, '?')}" for k in varying_keys) or os.path.basename(f)
        for f, p in zip(files, all_params)
    ]
    fixed_str = "  ".join(f"{k}={v}" for k, v in fixed.items())
    suptitle = r"$S\cdot S$ (xx+yy+zz)"
    if fixed_str:
        suptitle += f"\n{fixed_str}"

    fig, axes = plt.subplots(1, n_panels, figsize=(2 * n_panels, 3), sharex=True, sharey=True)

    if n_panels == 1:
        axes = [axes]

    fig.suptitle(suptitle)

    c = [plot_ssf(ax, f, args, title=t) for ax, f, t in zip(axes, files, panel_titles)]
    cax = fig.add_axes([0.05,0.9,0.3,0.02])
    plt.colorbar(c[0], cax=cax, orientation='horizontal')
        

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
