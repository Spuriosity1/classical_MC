#!/usr/bin/env python3
"""Plot SSF intensity at (0,0,Qz) and symmetry-equivalent q-points vs a scan parameter."""

from plot_ssf import load_file, normalize_ssf, split_fixed_varying
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import sys


def qz_to_idx(qz_frac, k_dim):
    return int(round(qz_frac * k_dim)) % k_dim


def load_ssf_variance(path):
    """Load var_inter / var_intra / n_seeds from a merged HDF5 file (acc_runs.py).

    var_inter : (biased, /K) variance of the per-seed mean S(q) across seeds.
    var_intra : sum over seeds of the per-seed (within-run) single-sample
                variance of S(q).
    Both are already sublattice-contracted scalars (the phase fold is done in
    the C++ writer / propagated linearly through acc_runs), stored as
    [n_corr, n_T, n_k, 2] with the last axis carrying the Re/Im-part variances.
    Returns (var_inter, var_intra, n_seeds) as dicts label -> [n_T, k0,k1,k2]
    (real, = Re-part variance), or None for a variance whose dataset is absent.
    """
    def reshape_var(raw, corr_lookup, k_dims):
        # Re-part variance is what matters for the real observable S(q).
        real = raw[..., 0]                            # [n_corr, n_T, n_k]
        k0, k1, k2 = k_dims
        real = real.reshape(len(corr_lookup), -1, k0, k1, k2)
        return {label: real[i] for i, label in enumerate(corr_lookup)}

    var_inter = var_intra = None
    with h5py.File(path, "r") as f:
        ssf = f["/ssf"]
        k_dims = ssf.attrs["k_dims"][:].astype(int)
        corr_lookup = [s.decode() if isinstance(s, bytes) else s
                       for s in ssf["corr_lookup"][:]]
        if "var_inter" in ssf:
            var_inter = reshape_var(ssf["var_inter"][:], corr_lookup, k_dims)
        if "var_intra" in ssf:
            var_intra = reshape_var(ssf["var_intra"][:], corr_lookup, k_dims)
        energy = f["/energy"]
        n_seeds = int(energy["n_seeds"][()]) if "n_seeds" in energy else None
    return var_inter, var_intra, n_seeds


def se_from_inter(W_inter, n_seeds, n_spins):
    """Standard error of the multi-seed mean SSF from var_inter.

    W_inter : var_inter[k_q], summed over diagonal components — the biased
              (/K) population variance of the per-seed mean S(q) across seeds.
    Bessel-corrected SE of the K-seed mean: sqrt(W_inter / (K-1)) / n_spins.
    """
    if n_seeds is None or n_seeds < 2:
        return np.nan
    return np.sqrt(max(float(W_inter), 0.0) / (n_seeds - 1)) / n_spins


def se_from_intra(W_intra, n_seeds, n_per_seed, n_spins):
    """Standard error contribution to the multi-seed mean SSF from
    within-run (intra-seed) sampling noise.

    W_intra    : var_intra[k_q], summed over diagonal components —
                 Σ_seeds Var[single MC sample of S(q) | seed].
    n_per_seed : MC samples per seed at this temperature (= n_ssf_t / K).

    Treats samples within a seed as independent (no autocorrelation
    correction) and seeds as independent, consistent with se_from_inter().
    """
    if n_seeds is None or n_seeds < 1 or not n_per_seed or n_per_seed <= 0:
        return np.nan
    return np.sqrt(max(float(W_intra), 0.0) / (n_seeds**2 * n_per_seed)) / n_spins


def main():
    p = argparse.ArgumentParser(
        description="Plot SSF intensity at Bragg/symmetry-equivalent q-points vs a scan parameter."
    )
    p.add_argument("file", help="Path(s) to HDF5 file", nargs='+')
    p.add_argument("-x", "--x-axis", default=None,
                   help="Tag to plot along the x-axis (default: temperature)")
    p.add_argument("-s", "--series-axis", help="Tag to use as a series label")
    p.add_argument("-t", "--t-index", type=int, default=None,
                   help="Temperature index into the SSF array (default: last = coldest); "
                        "ignored when -x is not given (temperature mode)")
    p.add_argument("--err-source", choices=["inter", "intra", "total", "both"],
                   default="inter",
                   help="Error bar source (default: inter). 'inter' = seed-to-seed "
                        "SE of the mean; 'intra' = within-run sampling-noise SE "
                        "only; 'total' = both combined in quadrature; 'both' = draw "
                        "intra as a faint wide band behind the solid inter-seed bars")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("-o", "--output", default=None,
                   help="Save figure to file instead of displaying")
    args = p.parse_args()

    files = args.file
    fixed, _, all_params = split_fixed_varying(files)

    temp_mode = args.x_axis is None
    x_label = "T" if temp_mode else args.x_axis

    if args.series_axis:
        series_vals = sorted(set(pm.get(args.series_axis, '?') for pm in all_params))
    else:
        series_vals = [None]

    def get_series(params):
        return params.get(args.series_axis, '?') if args.series_axis else None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    panel_labels = [
        r"$(0,0,Q_z)$  [Bragg]",
        r"$(Q_z,0,0)$",
        r"$(0,Q_z,0)$",
        r"$\Gamma = (0,0,0)$",
    ]
    for ax, title in zip(ax_flat, panel_labels):
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$S(\mathbf{q})$ / spin")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    series_color = {v: colors[i % len(colors)] for i, v in enumerate(series_vals)}

    series_data = {
        v: {'x': [], 'I': [[], [], [], []],
            'SE_inter': [[], [], [], []], 'SE_intra': [[], [], [], []]}
        for v in series_vals
    }

    for fpath, params in zip(files, all_params):
        if not temp_mode:
            x_str = params.get(args.x_axis)
            if x_str is None:
                print(f"Warning: '{args.x_axis}' not found in {os.path.basename(fpath)}, skipping",
                      file=sys.stderr)
                continue
            try:
                x_val = float(x_str)
            except ValueError:
                x_val = x_str

        qz_str = params.get('Q') or params.get('Qz')
        if qz_str is None:
            print(f"Warning: 'Q' or 'Qz' not found in {os.path.basename(fpath)}, skipping",
                  file=sys.stderr)
            continue
        qz = float(qz_str)

        (_, _, _, _, _, _,
         corr, corr_lookup, sl_positions, k_dims, n_spins, ssf_T, n_ssf) = load_file(fpath)


        n_T = corr.shape[1]
        if not temp_mode:
            t_idx = args.t_index if args.t_index is not None else n_T - 1
            if not (0 <= t_idx < n_T):
                sys.exit(f"--t-index {t_idx} out of range [0, {n_T - 1}]")
            t_indices = [t_idx]
        else:
            t_indices = range(n_T)

        S = normalize_ssf(corr, corr_lookup, k_dims, n_ssf)
        diag = [c for c in ("xx", "yy", "zz") if c in S]
        if not diag:
            print(f"Warning: no diagonal correlators in {os.path.basename(fpath)}, skipping",
                  file=sys.stderr)
            continue

        var_inter, var_intra, n_seeds = load_ssf_variance(fpath)
        # S_inter = (contract_corr2(var_inter, corr_lookup, sl_positions, k_dims)
        #            if var_inter is not None else None)
        # S_intra = (contract_corr2(var_intra, corr_lookup, sl_positions, k_dims)
        #            if var_intra is not None else None)
        S_inter = var_inter
        S_intra= var_intra


        # Qz is in units of 2π/a_cubic; k_dims[i] = L for cubic supercell
        qi = qz_to_idx(qz, k_dims[0])

        q_indices = [
            (0,  0,  qi),   # (0, 0, Qz)
            (qi, 0,  0),    # (Qz, 0, 0)
            (0,  qi, 0),    # (0, Qz, 0)
            (0,  0,  0),    # Gamma
        ]

        ser = get_series(params)
        for t_idx in t_indices:
            x_val = ssf_T[t_idx] if temp_mode else x_val  # noqa: F821 (x_val set above for non-temp)
            series_data[ser]['x'].append(x_val)
            n_per_seed = n_ssf[t_idx] / n_seeds if n_seeds else np.nan
            for panel, (i0, i1, i2) in enumerate(q_indices):
                intensity = sum(S[c][t_idx, i0, i1, i2] for c in diag) / n_spins
                series_data[ser]['I'][panel].append(intensity)

                if S_inter is not None and n_seeds is not None:
                    W_inter = sum(S_inter[c][t_idx, i0, i1, i2] for c in diag)
                    se_inter = se_from_inter(W_inter, n_seeds, n_spins)
                # elif S2 is not None and n_seeds is not None:
                #     W_q = sum(S2[c][t_idx, i0, i1, i2] for c in diag)
                #     se_inter = cross_seed_se(W_q, intensity, n_seeds, n_ssf[t_idx], n_spins)
                else:
                    se_inter = np.nan

                if S_intra is not None and n_seeds is not None:
                    W_intra = sum(S_intra[c][t_idx, i0, i1, i2] for c in diag)
                    se_intra = se_from_intra(W_intra, n_seeds, n_per_seed, n_spins)
                else:
                    se_intra = np.nan

                series_data[ser]['SE_inter'][panel].append(se_inter)
                series_data[ser]['SE_intra'][panel].append(se_intra)

    plotted_any = False
    for ser in series_vals:
        d = series_data[ser]
        if not d['x']:
            continue
        order = np.argsort(d['x'])
        x_sorted = np.array(d['x'])[order]
        label = str(ser) if ser is not None else None
        color = series_color[ser]
        for panel, ax in enumerate(ax_flat):
            I_sorted = np.array(d['I'][panel])[order]
            SEi_sorted = np.array(d['SE_inter'][panel])[order]
            SEa_sorted = np.array(d['SE_intra'][panel])[order]

            if args.err_source == "inter":
                se_main = SEi_sorted
            elif args.err_source == "intra":
                se_main = SEa_sorted
            else:  # total or both -> main bar is the combined/inter estimate
                have_either = np.isfinite(SEi_sorted) | np.isfinite(SEa_sorted)
                se_main = np.sqrt(np.nan_to_num(SEi_sorted)**2
                                   + np.nan_to_num(SEa_sorted)**2)
                se_main = np.where(have_either, se_main, np.nan)

            if args.err_source == "both" and np.any(np.isfinite(SEa_sorted)):
                # faint wide band behind the main bars shows intra-seed spread
                ax.errorbar(x_sorted, I_sorted, yerr=np.nan_to_num(SEa_sorted),
                            fmt='none', ecolor=color, alpha=0.3,
                            elinewidth=5, capsize=0, zorder=1)
                se_main = SEi_sorted

            if np.any(np.isfinite(se_main)):
                ax.errorbar(x_sorted, I_sorted, yerr=se_main,
                            fmt='o-', label=label, color=color, ms=4,
                            capsize=3, elinewidth=1, zorder=2)
            else:
                ax.plot(x_sorted, I_sorted, 'o-', label=label, color=color, ms=4, zorder=2)
        plotted_any = True

    if not plotted_any:
        if temp_mode:
            sys.exit("No data to plot — check that Qz tag exists in the filenames.")
        else:
            sys.exit("No data to plot — check that --x-axis and Qz tags exist in the filenames.")

    if args.series_axis:
        ax_flat[0].legend(title=args.series_axis, fontsize=8)

    if args.vmin is not None or args.vmax is not None:
        for ax in ax_flat:
            ax.set_ylim(args.vmin, args.vmax)

    fixed_str = "  ".join(f"{k}={v}" for k, v in fixed.items())

    if temp_mode:
        suptitle = r"$S(\mathbf{q})$ at symmetry-equivalent wavevectors vs $T$"
    else:
        suptitle = r"$S(\mathbf{q})$ at symmetry-equivalent wavevectors, T=" + str(ssf_T[t_idx])
    if fixed_str:
        suptitle += f"\n{fixed_str}"
    fig.suptitle(suptitle)

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
