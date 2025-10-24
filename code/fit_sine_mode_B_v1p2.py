import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Intentar aplicar un estilo limpio si seaborn está disponible
try:
    import seaborn as sns
    sns.set(style="whitegrid", context="paper")
except ImportError:
    plt.style.use("default")

def sine_func(phi, A, phi0, C):
    """Modelo sinusoidal Δ(φ) = A sin(φ − φ0) + C"""
    return A * np.sin(np.radians(phi - phi0)) + C

# --- Leer datos ---
try:
    df = pd.read_csv("results_sweep_B/sweep_B_results.csv")
except FileNotFoundError:
    raise SystemExit("❌ No se encontró results_sweep_B/sweep_B_results.csv")

phi = df["phi_deg"].to_numpy()
delta = df["delta_mean"].to_numpy()

# Manejar columnas de error flexiblemente
err_candidates = ["delta_std", "delta_err", "std", "error"]
delta_err = None
for c in err_candidates:
    if c in df.columns:
        delta_err = df[c].to_numpy()
        break
if delta_err is None:
    delta_err = np.ones_like(delta) * np.std(delta) * 0.05  # error genérico 5%

# --- Ajuste sinusoidal ---
from scipy.optimize import curve_fit
try:
    popt, pcov = curve_fit(sine_func, phi, delta, p0=[100, 130, 0])
except Exception as e:
    raise SystemExit(f"❌ Error durante el ajuste: {e}")

A, phi0, C = popt
Aerr, phi0err, Cerr = np.sqrt(np.diag(pcov))

# --- Graficar ---
plt.figure(figsize=(7, 4))
plt.errorbar(phi, delta, yerr=delta_err, fmt='o', color='tab:blue', capsize=3, label="Data")
phi_fit = np.linspace(0, 180, 500)
plt.plot(phi_fit, sine_func(phi_fit, *popt), '-', color='tab:blue',
         label=f"Fit: A={A:.1f}±{Aerr:.1f}, φ₀={phi0:.1f}°±{phi0err:.1f}°")

plt.xlabel("Rotation angle φ (deg)")
plt.ylabel("ΔDM (pc cm$^{-3}$)")
plt.title("Mode B – Orthogonal-axis rotation")
plt.legend()
plt.tight_layout()

# --- Guardar figura ---
out_path = "results_sweep_B_fit/delta_phi_sine_fit_B_v1p2.png"
plt.savefig(out_path, dpi=300)
print(f"✅ Figura guardada: {out_path}")
