#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotational Sweep for CPT Hemispheres Test
Author: CosmicThinker & Toko (2025-10-24)

Barrera φ en [0°,180°] cada Δφ=10°, mide Δ(φ) y p_perm(φ).
Genera gráficos de anisotropía azimutal.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hemispheres_cpt_test_v2_rotplane import (
    radec_to_unit, axis_unit, orthonormal_basis_from_axis,
    plane_normal, assign_hemisphere, bootstrap_delta_mean,
    permutation_pvalue_delta
)
from scipy.stats import mannwhitneyu

def sweep_rotational(frbcsv, outdir="results_sweep",
                     axis_ra=170, axis_dec=40,
                     abs_b_min=20, min_dm=800,
                     n_perm=5000, seed=42,
                     dphi=10, balance=False):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    df = pd.read_csv(frbcsv)
    if abs_b_min is not None and "galactic_lat" in df.columns:
        df = df.loc[df["galactic_lat"].abs() >= abs_b_min]
    if min_dm is not None:
        df = df.loc[df["dm"] >= min_dm]
    df = df.dropna(subset=["ra_deg","dec_deg","dm"])
    pts = radec_to_unit(df["ra_deg"].values, df["dec_deg"].values)
    axis = axis_unit(axis_ra, axis_dec)

    results = []
    phis = np.arange(0,181,dphi)
    for phi in phis:
        normal_vec = plane_normal(axis, mode="through-axis", phi_deg=phi)
        hemi, dots = assign_hemisphere(pts, normal_vec)
        dm_plus = df.loc[hemi==1, "dm"].values
        dm_minus = df.loc[hemi==-1, "dm"].values

        if balance:
            n = min(len(dm_plus), len(dm_minus))
            dm_plus = rng.choice(dm_plus, size=n, replace=False)
            dm_minus = rng.choice(dm_minus, size=n, replace=False)

        if len(dm_plus)<3 or len(dm_minus)<3:
            continue

        obs_delta, null, p_perm = permutation_pvalue_delta(df["dm"].values, hemi, n_perm=n_perm, rng=rng)
        U, p_mw = mannwhitneyu(dm_plus, dm_minus, alternative="two-sided")
        q16, q84, q025, q975 = bootstrap_delta_mean(dm_plus, dm_minus, n_boot=5000, rng=rng)
        results.append({
            "phi_deg": phi,
            "N_plus": len(dm_plus),
            "N_minus": len(dm_minus),
            "delta_mean": obs_delta,
            "p_perm": p_perm,
            "p_mw": p_mw,
            "CI95": [q025,q975]
        })
        print(f"φ={phi:3.0f}° | Δ={obs_delta:7.1f} | p_perm={p_perm:.3f}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(outdir,"rotational_sweep_results.csv"),index=False)

    # Plot Δ(φ)
    plt.figure(figsize=(7,4))
    plt.plot(df_out["phi_deg"], df_out["delta_mean"], "o-", label="Δ mean(DM+ − DM−)")
    plt.axhline(0,color="k",lw=0.8)
    plt.xlabel("φ (deg)")
    plt.ylabel("Δ mean DM [pc cm⁻³]")
    plt.title("Rotational Hemispheres Sweep – Δ(φ)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"delta_vs_phi.png"),dpi=150)
    plt.close()

    # Plot p(φ)
    plt.figure(figsize=(7,4))
    plt.semilogy(df_out["phi_deg"], df_out["p_perm"], "s-", color="tab:orange", label="p_perm")
    plt.axhline(0.05,color="r",ls="--",label="p=0.05")
    plt.xlabel("φ (deg)")
    plt.ylabel("Permutation p-value")
    plt.title("Rotational Hemispheres Sweep – p(φ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"pvalue_vs_phi.png"),dpi=150)
    plt.close()

    print(f"[✓] Sweep completed. Results saved to {outdir}/")
    return df_out

if __name__ == "__main__":
    sweep_rotational("data/frb_catalog.csv")
