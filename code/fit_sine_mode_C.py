# fit_sine_mode_C.py
import json, os, math, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==== CONFIG ====
INPUT_CSV = "rotational_sweep_results.csv"          # contiene barridos de φ para modos B y C
MODE_FIELD = "mode"                                  # columna con valores 'B' o 'C' (si no existe, usa filtro manual)
PHI_COL = "phi_deg"                                  # en grados [0..360)
Y_COL = "delta_metric"                               # métrica del modo C (p.ej., ΔDM_through(φ)); ajusta al nombre real
FILTER_MODE = "C"                                    # filtra filas del modo C
OUTDIR = "results_sweep_C_fit"
OUT_FIG = os.path.join(OUTDIR, "delta_phi_sine_fit_C.png")
OUT_JSON = "sine_fit_summary_C.json"
N_PERM = 20000                                       # permutaciones para p_perm(|A|); sube si quieres
SEED = 1234

# ==== CARGA CSV (robusta a separadores) ====
def smart_read_csv(path):
    import csv
    with open(path, "r", encoding="utf-8") as f:
        # detecta separador
        sample = f.read(2048); f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.DictReader(f, dialect=dialect)
        rows = [r for r in reader]
    return rows

rows = smart_read_csv(INPUT_CSV)

# si no hay columna 'mode', asume que el archivo ya es solo del modo C
if MODE_FIELD in (rows[0].keys()):
    rows = [r for r in rows if (r.get(MODE_FIELD,"").strip().upper()==FILTER_MODE)]
if not rows:
    raise RuntimeError("No se han encontrado filas del modo C. Revisa INPUT_CSV / FILTER_MODE / columnas.")

# parseo
def as_float(s):
    try: return float(str(s).strip())
    except: return np.nan

phi = np.array([as_float(r[PHI_COL]) for r in rows], dtype=float)
y   = np.array([as_float(r[Y_COL])   for r in rows], dtype=float)
mask = np.isfinite(phi) & np.isfinite(y)
phi, y = phi[mask], y[mask]
if phi.size < 5:
    raise RuntimeError("Muy pocos puntos válidos para ajustar.")

# diseña base sen/cos
phi_rad = np.deg2rad(phi)
S = np.sin(phi_rad)
C = np.cos(phi_rad)
X = np.column_stack([S, C, np.ones_like(S)])  # [a, b, c]

# ajuste lineal por mínimos cuadrados
beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)  # beta = [a,b,c]
a, b, c0 = beta

# métricas
y_hat = X @ beta
ss_res = np.sum((y - y_hat)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
n, p = len(y), X.shape[1]
sigma2 = ss_res / max(n - p, 1)

# covarianza de beta
XtX_inv = np.linalg.inv(X.T @ X)
cov_beta = sigma2 * XtX_inv
se_a, se_b, se_c = np.sqrt(np.diag(cov_beta))

# convierte a (A, phi0, C)
A = math.hypot(a, b)
phi0_rad = math.atan2(b, a)  # porque y = a*sinφ + b*cosφ = A*sin(φ + atan2(b,a))
# Queremos y = A*sin(φ - φ0) => φ0 = -atan2(b,a)
phi0_deg = (-np.rad2deg(phi0_rad)) % 360.0
Coff = c0

# errores de A y phi0 por propagación lineal (aprox.)
# dA/da = a/A, dA/db = b/A
if A > 0:
    var_A = (a/A)**2 * se_a**2 + (b/A)**2 * se_b**2 + 2*(a/A)*(b/A)*cov_beta[0,1]
    se_A = math.sqrt(max(var_A, 0.0))
else:
    se_A = np.nan
# φ0 = -atan2(b,a) -> derivadas: dφ0/da =  b/(a^2 + b^2), dφ0/db = -a/(a^2 + b^2), con signo por el '-'
den = a*a + b*b
if den > 0:
    dph0_da =  b/den
    dph0_db = -a/den
    var_ph0 = dph0_da**2 * se_a**2 + dph0_db**2 * se_b**2 + 2*dph0_da*dph0_db*cov_beta[0,1]
    se_phi0_deg = np.rad2deg(math.sqrt(max(var_ph0,0.0)))
else:
    se_phi0_deg = np.nan

# Permutación para p_perm(|A|)
rng = np.random.default_rng(SEED)
A_obs = A
As = []
for _ in range(N_PERM):
    y_perm = rng.permutation(y)
    beta_perm, *_ = np.linalg.lstsq(X, y_perm, rcond=None)
    a_p, b_p = beta_perm[0], beta_perm[1]
    As.append(math.hypot(a_p, b_p))
As = np.array(As)
p_perm = (np.sum(As >= abs(A_obs)) + 1) / (N_PERM + 1)

# salida
os.makedirs(OUTDIR, exist_ok=True)

# figura
grid_phi = np.linspace(0, 360, 720)
fit = A * np.sin(np.deg2rad(grid_phi - phi0_deg)) + Coff

plt.figure(figsize=(7,4))
plt.scatter(phi, y, s=20, alpha=0.8, label="Datos modo C")
plt.plot(grid_phi, fit, lw=2, label=f"Fit: A={A_obs:.1f}±{se_A:.1f}, φ0={phi0_deg:.1f}°±{se_phi0_deg:.1f}°, C={Coff:.1f}")
plt.xlabel("φ (grados)")
plt.ylabel("Δ(φ) modo C")
plt.title("Ajuste sinusoidal — Modo C (through-axis)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=200)
plt.close()

summary = {
    "mode": "C",
    "timestamp": datetime.utcnow().isoformat()+"Z",
    "n_points": int(n),
    "A": A_obs, "A_err": float(se_A),
    "phi0_deg": float(phi0_deg), "phi0_err_deg": float(se_phi0_deg),
    "C": float(Coff), "C_err": float(se_c),
    "R2": float(r2),
    "p_perm_absA": float(p_perm),
    "inputs": {
        "csv": INPUT_CSV, "phi_col": PHI_COL, "y_col": Y_COL, "mode_field": MODE_FIELD, "filter_mode": FILTER_MODE
    },
    "outputs": {
        "figure": OUT_FIG
    },
    "fit_form": "y(φ)=A·sin(φ−φ0)+C",
    "notes": "Ajuste linealizado (a sinφ + b cosφ + c) y conversión a (A, φ0, C). Permutaciones sobre y."
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[OK] Modo C ajustado. R2={r2:.3f}, A={A_obs:.2f}±{se_A:.2f}, φ0={phi0_deg:.1f}°±{se_phi0_deg:.1f}°, p_perm(|A|)={p_perm:.4g}")
print(f"Figura: {OUT_FIG}")
print(f"Resumen: {OUT_JSON}")
