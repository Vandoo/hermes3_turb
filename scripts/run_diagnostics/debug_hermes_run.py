"""
debug_cvode_run.py
------------------
Debug and inspect BOUT++ / Hermes-3 CVODE runs.

opt_plt options:
  time_var          — plot max/min of each variable over all saved timesteps
  field_aligned_var — convert variables to field-aligned coords, report max/min/location
  crash_seed        — localise critically-low / negative density (crash seed)
  inventory_time    — n_sep / particle inventory / min(Ne) across folders (depletion?)
  precon_coef       — conduction-precon κ/N conditioning (zero-pivot risk)
  neutral_diff      — neutral perp-diffusion stiffness (Dnn/Δ⊥²)
  field_autopsy     — per-flux-region field post-mortem: density contrast (EM Apar
                      Naulin risk), pressure blow-up, T=P/N excursions, and WHERE
  stiffness_map     — WHERE/HOW HARD the accuracy limit caps dt (local-error weight
                      map from saved ddt(*)) + 3-D structure probe at the worst cells
  log_autopsy       — classify a failure from BOUT.log.* : which solver failed +
                      CVODE trajectory → timeout / Newton-collapse / divergence / first-step

Triage order for a failed run: log_autopsy (how it died) → inventory_time
(depletion vs local holes) → field_autopsy (blow-up vs hole, and where).
"""

from __future__ import annotations

import logging
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import sys, os, fnmatch, configparser

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
cfg = configparser.ConfigParser()
cfg.read(os.path.join(os.getcwd(), 'config_analysis.ini'))
prepath  = cfg.get('debug', 'prepath',  fallback='./1/')
frame    = cfg.getint('debug', 'frame', fallback=-1)   # time index for stiffness_map
opt_plt  = cfg.get('debug', 'opt_plt',  fallback='field_aligned_var')
# opt_plt  = "field_aligned_var"
# opt_plt = "time_var"
# opt_plt = "crash_seed"
# opt_plt = "inventory_time"
# opt_plt = "precon_coef"
# opt_plt = "neutral_diff"
# opt_plt = "field_autopsy"
# opt_plt = "log_autopsy"
# opt_plt = "stiffness_map"

DATANAME = "boutdata.nc"

VAR_LIST = [
    "Ne", "Pe", "NVe",
    "Ni", "Pi", "NVi",
    "Nd+", "Pd+", "NVd+",
    "Nd", "Pd", "NVd",
    "Vort", "phi",
]
FVAR_LIST = [
    "Pi", "SPi", "ddt(Pi)",
    "efi_cond_ylow", "efi_kin_ylow", "efi_tot_ylow",
    "Pi_src", "Pi_cipar", "Pi_ciperp",
    "Ti", "Ni",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_file(pattern: str, search_path: str) -> str | list:
    """Return the first file matching *pattern* under *search_path*, or []."""
    for root, _dirs, files in os.walk(search_path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return []


def _open_xhermes(path: str, gridname: str, yguard: bool,
                  unnormalise: bool = True):
    """Open a Hermes dataset via xhermes, with or without a grid file."""
    import xhermes
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    if len(gridname) == 0:
        return xhermes.open_hermesdataset(path, inputfilepath="./BOUT.inp")
    return xhermes.open_hermesdataset(
        path,
        geometry="toroidal",
        gridfilepath=gridname,
        keep_xboundaries=True,
        keep_yboundaries=yguard,
        unnormalise=unnormalise,
        info=False,
    )
# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
# Modes that take every quantity (metric, topology, guard counts) from the dump
# itself and so must not fail when no grid file is reachable.
GRIDLESS_MODES = {"stiffness_map"}

gridname = find_file("*.nc", "../")
gr, MYG, yguard = None, 2, True
if opt_plt in GRIDLESS_MODES:
    log.info("%s reads geometry from the dump — skipping the grid file", opt_plt)
else:
    gr     = nc.Dataset(gridname)
    MYG    = int(gr["y_boundary_guards"][:])
    yguard = MYG == 2

# ---------------------------------------------------------------------------
# time_var
# ---------------------------------------------------------------------------
if opt_plt == "time_var":
    jyseps1_1 = int(gr["jyseps1_1"][:]) + MYG
    jyseps2_1 = int(gr["jyseps2_1"][:]) + MYG
    jyseps1_2 = int(gr["jyseps1_2"][:]) + MYG * 3
    jyseps2_2 = int(gr["jyseps2_2"][:]) + MYG * 3
    if jyseps2_1 != jyseps1_2:
        ny_inner = int(gr["ny_inner"][:])
    gr.close()

    log.info("netCDF data loads")
    ds = nc.Dataset(prepath + DATANAME)

    fig, ax = plt.subplots()
    for var in VAR_LIST:
        if var in ds.variables:
            data  = ds[var][:, 2:-2, :, :]   # (t, x, y, z)
            max_t = np.max(data, axis=(1, 2, 3))
            min_t = np.min(data, axis=(1, 2, 3))
            ax.plot(max_t, label=f"{var} max")
            ax.plot(min_t, label=f"{var} min")

    ax.legend(ncol=3)
    ax.set_xlabel("timestep")
    plt.tight_layout()
    plt.show()
    ds.close()

    log.info("--------------------------------------------------------")
    log.info("data = ds['Pi'][-1, 2:-2, :, :]")
    log.info("np.unravel_index(np.argmax(data), data.shape)")

# ---------------------------------------------------------------------------
# field_aligned_var
# ---------------------------------------------------------------------------
elif opt_plt == "field_aligned_var":
    jyseps1_1 = int(gr["jyseps1_1"][:]) + MYG
    jyseps2_1 = int(gr["jyseps2_1"][:]) + MYG
    jyseps1_2 = int(gr["jyseps1_2"][:]) + MYG * 3
    jyseps2_2 = int(gr["jyseps2_2"][:]) + MYG * 3
    if jyseps2_1 != jyseps1_2:
        ny_inner = int(gr["ny_inner"][:])
    ny_grid = gr.dimensions["y"].size
    ixseps1 = int(gr["ixseps1"][:]) - 2
    gr.close()

    ds = _open_xhermes(prepath + DATANAME, gridname, yguard, unnormalise=True)

    # Build ybndry_id using xarray dim name 'theta'
    if ds.dims["theta"] > ny_grid:
        log.info("*** y_guard_cells saved in boutdata but not in gridfile ***")
        ybndry_id = [1, ny_inner + 3, ny_inner + 5, ds.dims["theta"] - 1]
    else:
        if MYG == 2:
            ybndry_id = [1, ny_inner + 3, ny_inner + 5, ny_grid - 1]
        else:
            ybndry_id = [0, ny_inner + 1, ny_inner, ny_grid]

    x_slice  = slice(MYG, -MYG) if MYG > 0 else slice(None)
    x_offset = MYG if MYG > 0 else 0

    fds: dict = {}
    log.info("fds in field-aligned coord")

    for var in FVAR_LIST:
        if var not in list(ds.variables):
            continue

        fds[var]   = ds[var].bout.to_field_aligned()
        data_tmp   = np.concatenate(
            [fds[var][:, x_slice, ybndry_id[0]:ybndry_id[1], :],
             fds[var][:, x_slice, ybndry_id[2]:ybndry_id[3], :]],
            axis=2,
        )

        id_max = np.unravel_index(np.argmax(data_tmp), data_tmp.shape)
        id_min = np.unravel_index(np.argmin(data_tmp), data_tmp.shape)
        max_v  = float(data_tmp[id_max])
        min_v  = float(data_tmp[id_min])

        print(
            f"{var:6s}: "
            f"max={max_v:.2e} @(t={id_max[0]},x={id_max[1]+x_offset},y={id_max[2]},z={id_max[3]}), "
            f"min={min_v:.2e} @(t={id_min[0]},x={id_min[1]+x_offset},y={id_min[2]},z={id_min[3]})"
        )

    log.info("vars in f-aligned (fds): %s", list(fds.keys()))

# ---------------------------------------------------------------------------
# crash_seed — localise the seed of a crash: domain cells where density is
# critically low or T<0 (Te=Pe/Ne and V=NV/N blow up as Ne→0). Works on a
# single frame (current violators) or a multi-frame dump (first violation time).
# ---------------------------------------------------------------------------
elif opt_plt == "crash_seed":
    N_FLOOR = 1e-3   # normalised density threshold (× Nnorm) for "critically low"

    if gr is not None:
        gr.close()
    ds = nc.Dataset(prepath + DATANAME)
    nxf = ds.dimensions["x"].size
    nyf = ds.dimensions["y"].size
    # Guard layout MUST come from the DUMP: the Hypnotoad grid is guardless
    # (y_boundary_guards = 0), whereas the data here retains y-guards (MYG=2).
    MXG  = int(np.array(ds["MXG"][...]))      if "MXG" in ds.variables else 2
    MYGd = int(np.array(ds["MYG"][...]))      if "MYG" in ds.variables else 2
    ny_d = int(np.array(ds["ny"][...]))       if "ny"  in ds.variables else nyf
    nyi  = int(np.array(ds["ny_inner"][...])) if "ny_inner" in ds.variables else 0
    g    = MYGd if nyf > ny_d else 0          # y-guards present in the dump?
    is_dn = 0 < nyi < ny_d

    # Domain y-mask (drop target + inter-region guard bands)
    ym = np.zeros(nyf, bool)
    if is_dn:                                 # double-null: two regions
        ym[g:nyi + g] = True
        ym[nyi + 3 * g:nyf - g] = True
    else:                                     # single-null
        ym[g:nyf - g] = True
    xm = np.zeros(nxf, bool); xm[MXG:nxf - MXG] = True
    dom = (xm[:, None] & ym[None, :])[:, :, None]    # (x, y, 1) → broadcasts over z

    Ne = np.array(ds["Ne"][:]); Pe = np.array(ds["Pe"][:])
    if Ne.ndim == 3:                          # add a length-1 time axis
        Ne, Pe = Ne[None], Pe[None]
    Te = Pe / np.where(Ne != 0, Ne, np.nan)
    nt = Ne.shape[0]

    print(f"frames={nt}  N_FLOOR={N_FLOOR} (norm)  domain-only "
          f"(x[{MXG}:{nxf-MXG}], y-guards & inter-region band excluded)")
    first_t = None
    for t in range(nt):
        ne = np.where(dom, Ne[t], np.nan); te = np.where(dom, Te[t], np.nan)
        nlow = int(np.nansum(ne < N_FLOOR)); nnegT = int(np.nansum(te < 0))
        if (nlow or nnegT) and first_t is None:
            first_t = t
        print(f"  t={t:3d}: Ne<floor: {nlow:6d} cells (min {np.nanmin(ne):.2e});"
              f"  Te<0: {nnegT:5d} (min {np.nanmin(te):.2e})")
    if first_t is not None:
        print(f"\nfirst frame with a domain violation: t={first_t}")

    # Worst (lowest-Ne) domain cell at the last frame + its neighbourhood
    t = nt - 1
    ne = np.where(dom, Ne[t], np.nan)
    x0, y0, z0 = (int(i) for i in np.unravel_index(np.nanargmin(ne), ne.shape))
    print(f"\nworst low-Ne domain cell @ t={t}: (x={x0}, y={y0}, z={z0})")
    for v in ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort", "phi"):
        if v in ds.variables:
            print(f"   {v:5s} = {float(np.array(ds[v][t, x0, y0, z0])):.3e}")
    print(f"   Te = Pe/Ne = {Te[t, x0, y0, z0]:.3e}")
    print("   radial   Ne[x-2..x+2] =", np.array2string(Ne[t, max(x0-2,0):x0+3, y0, z0], precision=4))
    print("   parallel Ne[y-2..y+2] =", np.array2string(Ne[t, x0, max(y0-2,0):y0+3, z0], precision=4))
    print("   toroidal Ne[z-2..z+2] =", np.array2string(Ne[t, x0, y0, max(z0-2,0):z0+3], precision=4))
    print("   → a single-cell dip = grid-scale (resolution/dissipation); a broad "
          "dip = physical depletion (source/transport).")
    ds.close()

# ---------------------------------------------------------------------------
# inventory_time — track n_sep, particle inventory and min(Ne) across the
# numbered output folders (auto-discovered) to see whether the run is settling
# to its tuned steady state or draining. A flat n_sep at the target + a levelling
# SOL inventory = healthy; a monotonic drain = depleting.
# ---------------------------------------------------------------------------
elif opt_plt == "inventory_time":
    import glob
    gr.close()
    N_TARGET = 5e18   # tuned separatrix density [m^-3] for the reference line

    parent = os.path.dirname(os.path.normpath(prepath)) or "."
    folders = [d for d in glob.glob(os.path.join(parent, "[0-9]*"))
               if os.path.isdir(d) and os.path.basename(d).isdigit()]

    rows = []
    for folder in folders:
        fn = next((c for c in (os.path.join(folder, "boutdata_guards.nc"),
                               os.path.join(folder, "boutdata.nc"))
                   if os.path.exists(c)), None)
        if fn is None:
            continue
        d = nc.Dataset(fn)
        if "Ne" not in d.variables:
            d.close(); continue
        Nn = float(d["Nnorm"][...]); Om = float(d["Omega_ci"][...])
        nxf = d.dimensions["x"].size; nyf = d.dimensions["y"].size
        MYGd = int(d["MYG"][...]); nyi = int(d["ny_inner"][...])
        nyd = int(d["ny"][...]); MXG = int(d["MXG"][...]); ix1 = int(d["ixseps1"][...])
        g = MYGd if nyf > nyd else 0
        yomp = ((int(d["jyseps1_2"][...]) + 3 * g) + (int(d["jyseps2_2"][...]) + 3 * g)) // 2
        ym = np.zeros(nyf, bool)
        if 0 < nyi < nyd:
            ym[g:nyi + g] = True; ym[nyi + 3 * g:nyf - g] = True
        else:
            ym[g:nyf - g] = True
        xm = np.zeros(nxf, bool); xm[MXG:nxf - MXG] = True
        dom = xm[:, None] & ym[None, :]
        core = dom & (np.arange(nxf)[:, None] < ix1)
        sol = dom & (np.arange(nxf)[:, None] >= ix1)

        Ne = np.array(d["Ne"][:]); Ne = Ne[-1] if Ne.ndim == 4 else Ne
        dV = np.array(d["J"][:]) * np.array(d["dx"][:]) * np.array(d["dy"][:]) \
            * float(np.array(d["dz"]).flat[0])
        inv = lambda mk: float(np.nansum(np.where(mk[:, :, None], Ne, 0.0) * dV[:, :, None]))
        nsep = float(Ne[ix1, yomp, :].mean()) * Nn
        minNe = float(np.nanmin(np.where(dom[:, :, None], Ne, np.nan))) * Nn
        tms = (float(np.array(d["t_array"][-1])) / Om * 1e3) if "t_array" in d.variables else np.nan
        rows.append([int(os.path.basename(folder)), tms, nsep,
                     inv(dom), inv(core), inv(sol), minNe])
        d.close()

    rows.sort(key=lambda r: (r[1] if np.isfinite(r[1]) else r[0]))
    a = np.array(rows, float)
    print(f"{'dir':>5}{'t[ms]':>9}{'n_sep[m^-3]':>14}{'I_tot':>12}{'I_core':>12}"
          f"{'I_SOL':>12}{'minNe[m^-3]':>14}")
    for r in rows:
        print(f"{int(r[0]):>5}{r[1]:>9.3f}{r[2]:>14.3e}{r[3]:>12.3e}{r[4]:>12.3e}"
              f"{r[5]:>12.3e}{r[6]:>14.3e}")

    t = a[:, 1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(t, a[:, 2], "o-")
    ax[0].axhline(N_TARGET, color="r", ls="--", label=f"target {N_TARGET:.0e}")
    ax[0].set(xlabel="t [ms]", ylabel="n_sep [m$^{-3}$]", title="separatrix density (OMP)")
    ax[0].legend()
    ax[1].plot(t, a[:, 3], "o-", label="total")
    ax[1].plot(t, a[:, 4], "s-", label="core")
    ax[1].plot(t, a[:, 5], "^-", label="SOL")
    ax[1].set(xlabel="t [ms]", ylabel="particle inventory [norm]", title="inventory")
    ax[1].legend()
    ax[2].semilogy(t, np.clip(a[:, 6], 1e10, None), "o-")
    ax[2].set(xlabel="t [ms]", ylabel="min domain Ne [m$^{-3}$]", title="min density")
    fig.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# precon_coef — diagnose the parallel-conduction preconditioner (InvertParDiv,
# evolve_pressure::precon). It inverts  (1 + ∂_y[coef·∂_y]) with
#   coef = (2/3)·γ·kappa_par / softFloor(N, density_floor).
# A "Zero pivot in CyclicReduce" appears when |coef|/Δs∥² ≫ 1 so the diffusion
# term swamps the coefA=1 identity and the operator approaches the singular
# parallel Laplacian. This mode reports the γ-independent factor κ/N and the
# discrete diffusion number (κ/N)/Δs∥²; multiply by the CVODE internal step γ
# (~the "Last step size" in BOUT.log, ~1e-2) to get the actual number. A κ flux
# limiter should drop the max by orders of magnitude — re-run to confirm.
# ---------------------------------------------------------------------------
elif opt_plt == "precon_coef":
    DENSITY_FLOOR = 1e-7   # match the run's density_floor (raise to test mitigations)
    GAMMA_SCAN = [0.1, 1.0, 10.0]   # CVODE precon γ values to scan (≈ attempted step
                           # sizes). Absolute parallel diffusion number = γ·(κ/N)/Δs∥²;
                           # zero-pivot risk in cells where this > ~1.
    gr.close()
    ds = nc.Dataset(prepath + DATANAME)
    nxf, nyf = ds.dimensions["x"].size, ds.dimensions["y"].size
    MXG = int(np.array(ds["MXG"][...])); MYGd = int(np.array(ds["MYG"][...]))
    ny_d = int(np.array(ds["ny"][...])); nyi = int(np.array(ds["ny_inner"][...]))
    g = MYGd if nyf > ny_d else 0
    ym = np.zeros(nyf, bool)
    if 0 < nyi < ny_d:
        ym[g:nyi + g] = True; ym[nyi + 3 * g:nyf - g] = True
    else:
        ym[g:nyf - g] = True
    xm = np.zeros(nxf, bool); xm[MXG:nxf - MXG] = True
    dom = (xm[:, None] & ym[None, :])[:, :, None]

    g22 = np.array(ds["g_22"][:]); g22 = g22[0] if g22.ndim == 3 else g22
    dy = np.array(ds["dy"][:]);   dy = dy[0] if dy.ndim == 3 else dy
    dspar2 = (g22 * dy ** 2)[:, :, None]            # Δs∥² = g_22·dy²  (parallel cell length²)

    def report(species, kname, nname):
        if kname not in ds.variables:
            print(f"  {species}: {kname} absent"); return
        kap = np.array(ds[kname][:]); kap = kap[-1] if kap.ndim == 4 else kap
        N = np.array(ds[nname][:]);  N = N[-1] if N.ndim == 4 else N
        coef = (2.0 / 3.0) * kap / np.maximum(N, DENSITY_FLOOR)   # = |coefB|/γ
        cm = np.where(dom, coef, np.nan)
        dn = np.where(dom, coef / dspar2, np.nan)                 # (κ/N)/Δs∥²  (=diffnum/γ)
        ic = np.unravel_index(np.nanargmax(dn), dn.shape)
        ndom = int(np.isfinite(dn).sum())
        print(f"  {species}:  κ/N max={np.nanmax(cm):.2e}   (κ/N)/Δs∥² max={np.nanmax(dn):.2e}"
              f" @ (x={ic[0]},y={ic[1]},z={ic[2]})   [{ndom} domain cells]")
        for gamma in GAMMA_SCAN:
            ad = gamma * dn
            nrisk = int(np.nansum(ad > 1.0)); nwarn = int(np.nansum(ad > 0.1))
            print(f"        γ={gamma:>6g}:  max γ·diffnum={np.nanmax(ad):.2e}   "
                  f"cells>1 (zero-pivot risk): {nrisk:>6d}   cells>0.1 (marginal): {nwarn:>6d}")

    print(f"Conduction-precon coefficient diagnosis (density_floor={DENSITY_FLOOR}):")
    report("e", "kappa_par_e", "Ne")
    report("i", "kappa_par_i", "Ni" if "Ni" in ds.variables else "Ne")
    print("Mitigations to bring max(κ/N) down → better-conditioned precon: raise density_floor")
    print("(clamps N in the coef), or reduce the attempted step. CAUTION: kappa_limit_alpha is")
    print("double-edged — it lowers κ but TRAPS heat in steep-gradient legs → P blow-up (see")
    print("field_autopsy), and the limiter's q_fl=α·N·T·√(floor(T,0)) re-NaNs κ_∥ guards. Prefer")
    print("density_floor / step control first; only use a κ-limit if the precon is the sole issue.")
    ds.close()

# ---------------------------------------------------------------------------
# neutral_diff — neutral-diffusion stiffness (neutral_mixed). The neutral
# density/pressure diffusion ∝ Dnn∇²n is usually the stiffest term in a run
# with neutrals; its diffusion number per CVODE step is γ·Dnn/Δ², largest in
# the SMALL (perpendicular/radial) cells. CVODE needs γ·Dnn/Δ² ≲ 1 unless the
# term is preconditioned, so this predicts the stable step and shows whether a
# Dnn limiter / a neutral-diffusion preconditioner brings it into range.
#   Δ⊥² = dx²/g11   (perp/radial)      Δ∥² = g_22·dy²   (parallel)
# ---------------------------------------------------------------------------
elif opt_plt == "neutral_diff":
    GAMMA_SCAN = [1e-3, 3.6e-3, 1e-2]   # γ values to scan (neutrals need small γ)
    gr.close()
    ds = nc.Dataset(prepath + DATANAME)
    nxf, nyf = ds.dimensions["x"].size, ds.dimensions["y"].size
    MXG = int(np.array(ds["MXG"][...])); MYGd = int(np.array(ds["MYG"][...]))
    ny_d = int(np.array(ds["ny"][...])); nyi = int(np.array(ds["ny_inner"][...]))
    g = MYGd if nyf > ny_d else 0
    ym = np.zeros(nyf, bool)
    if 0 < nyi < ny_d:
        ym[g:nyi + g] = True; ym[nyi + 3 * g:nyf - g] = True
    else:
        ym[g:nyf - g] = True
    xm = np.zeros(nxf, bool); xm[MXG:nxf - MXG] = True
    dom = (xm[:, None] & ym[None, :])[:, :, None]

    g11 = np.array(ds["g11"][:]); g11 = g11[0] if g11.ndim == 3 else g11
    g22 = np.array(ds["g_22"][:]); g22 = g22[0] if g22.ndim == 3 else g22
    dx = np.array(ds["dx"][:]); dx = dx[0] if dx.ndim == 3 else dx
    dy = np.array(ds["dy"][:]); dy = dy[0] if dy.ndim == 3 else dy
    dperp2 = (dx ** 2 / g11)[:, :, None]    # perp cell size² = (dx/√g11)²
    dpar2 = (g22 * dy ** 2)[:, :, None]     # parallel cell size²

    # Main neutral diffusion coefficient(s): Dnn<species> (not the P/T/N variants)
    dnn_vars = [v for v in ds.variables
                if v.startswith("Dnn") and not v.startswith(("DnnP", "DnnT", "DnnN"))]
    if not dnn_vars:
        print("No Dnn* (neutral diffusion) variable found — not a neutral_mixed run?")
    for dv in dnn_vars:
        Dnn = np.array(ds[dv][:]); Dnn = Dnn[-1] if Dnn.ndim == 4 else Dnn
        perp = np.where(dom, Dnn / dperp2, np.nan)   # = diffusion number / γ (perp)
        par = np.where(dom, Dnn / dpar2, np.nan)
        ip = np.unravel_index(np.nanargmax(perp), perp.shape)
        ndom = int(np.isfinite(perp).sum())
        print(f"{dv}:  Dnn max={np.nanmax(np.where(dom,Dnn,np.nan)):.2e}")
        print(f"   Dnn/Δ⊥² max={np.nanmax(perp):.2e} @ (x={ip[0]},y={ip[1]},z={ip[2]});"
              f"   Dnn/Δ∥² max={np.nanmax(par):.2e}   [{ndom} cells]  (perp usually binds)")
        for gamma in GAMMA_SCAN:
            ap = gamma * perp
            nrisk = int(np.nansum(ap > 1.0)); nwarn = int(np.nansum(ap > 0.1))
            print(f"      γ={gamma:>8g}:  max γ·Dnn/Δ⊥²={np.nanmax(ap):.2e}   "
                  f"cells>1: {nrisk:>6d}   cells>0.1: {nwarn:>6d}")
    print("Stable explicit step needs γ·Dnn/Δ⊥² ≲ 1 (≈ where max→1). A neutral-diffusion")
    print("preconditioner removes this limit; a Dnn limiter lowers max(Dnn/Δ⊥²) directly.")
    ds.close()

# ---------------------------------------------------------------------------
# field_autopsy — single-dump post-mortem of the field state resolved by flux
# region (core / SOL / lower-PFR / upper-PFR via jyseps). Answers "blow-up vs
# hole vs depletion, and WHERE" in one shot. Reports per region:
#   • density min and contrast max(Ne)/min(Ne) — the predictor of an EM Apar
#     Naulin divergence (>~1e4-1e5 ⇒ won't converge);
#   • pressure max — heat-trap blow-up (healthy ~5, runaway ≫);
#   • #cells with T=P/N < 0 or non-finite (Ne→0 ⇒ Ti→∞).
# ---------------------------------------------------------------------------
elif opt_plt == "field_autopsy":
    APAR_CONTRAST_WARN = 1e4   # max(Ne)/min(Ne) above this → Naulin Apar risk
    BLOWUP_WARN        = 8.0   # P max above this (norm) → heat-trap blow-up (healthy ~5)
    DENSITY_FLOOR      = 1e-5  # the run's density_floor (norm), for the "#<floor" count

    if gr is not None:
        gr.close()
    ds = nc.Dataset(prepath + DATANAME)
    nxf = ds.dimensions["x"].size; nyf = ds.dimensions["y"].size
    MXG  = int(np.array(ds["MXG"][...]))      if "MXG" in ds.variables else 2
    MYGd = int(np.array(ds["MYG"][...]))      if "MYG" in ds.variables else 2
    ny_d = int(np.array(ds["ny"][...]))       if "ny"  in ds.variables else nyf
    nyi  = int(np.array(ds["ny_inner"][...])) if "ny_inner" in ds.variables else 0
    ix1  = int(np.array(ds["ixseps1"][...]))  if "ixseps1" in ds.variables else nxf
    g     = MYGd if nyf > ny_d else 0
    is_dn = 0 < nyi < ny_d
    # jyseps in dump y-index (same +g / +3g guard convention as field_aligned_var)
    j11 = int(np.array(ds["jyseps1_1"][...])) + g
    j21 = int(np.array(ds["jyseps2_1"][...])) + g
    j12 = int(np.array(ds["jyseps1_2"][...])) + 3 * g
    j22 = int(np.array(ds["jyseps2_2"][...])) + 3 * g

    yy = np.arange(nyf)
    confined = ((yy > j11) & (yy <= j21)) | ((yy > j12) & (yy <= j22))
    lower_leg = ((yy >= g) & (yy <= j11)) | ((yy > j22) & (yy < nyf - g))
    upper_leg = (((yy > j21) & (yy < nyi + g)) | ((yy >= nyi + 3 * g) & (yy <= j12))) \
        if is_dn else np.zeros(nyf, bool)
    ydom = np.zeros(nyf, bool)
    if is_dn:
        ydom[g:nyi + g] = True; ydom[nyi + 3 * g:nyf - g] = True
    else:
        ydom[g:nyf - g] = True
    xx = np.arange(nxf)
    xdom = (xx >= MXG) & (xx < nxf - MXG)
    inx  = xx < ix1                     # inside separatrix (closed / PFR radially)

    def _load(v):
        a = np.array(ds[v][:]) if v in ds.variables else None
        return a[-1] if (a is not None and a.ndim == 4) else a
    Ne = _load("Ne"); Pe = _load("Pe"); Pi = _load("Pi")
    Ti = Pi / np.where(Ne != 0, Ne, np.nan) if (Pi is not None and Ne is not None) else None

    def _rmask(ymask, radial):
        ym = (ymask & ydom)[None, :, None]
        xm = (xdom & (~inx if radial == "sol" else inx))[:, None, None]
        return xm & ym

    regions = [("core", confined, "core"),
               ("SOL", np.ones(nyf, bool), "sol"),
               ("lower-PFR", lower_leg, "pfr")]
    if is_dn:
        regions.append(("upper-PFR", upper_leg, "pfr"))

    print(f"field_autopsy ({DATANAME}, last frame)  "
          f"APAR_CONTRAST_WARN={APAR_CONTRAST_WARN:.0e}  BLOWUP_WARN={BLOWUP_WARN}")
    print(f"{'region':>11}{'#cells':>8}{'Ne_min':>11}{'Ne_max':>11}{'contrast':>10}"
          f"{'Pe_max':>9}{'Pi_max':>9}{'#Ti<0/inf':>11}")
    worst_hole = [None, np.inf]; worst_blow = [None, 0.0]; worst_contrast = [None, 1.0]
    for name, ymask, radial in regions:
        m = _rmask(ymask, radial)
        if not m.any():
            continue
        ne = np.where(m, Ne, np.nan)
        nemin = float(np.nanmin(ne)); nemax = float(np.nanmax(ne))
        contrast = nemax / nemin if nemin > 0 else np.inf
        pemax = float(np.nanmax(np.where(m, Pe, np.nan))) if Pe is not None else np.nan
        pimax = float(np.nanmax(np.where(m, Pi, np.nan))) if Pi is not None else np.nan
        nti = int(np.nansum(np.where(m, (Ti < 0) | ~np.isfinite(Ti), False))) if Ti is not None else 0
        print(f"{name:>11}{int(m.sum()):>8}{nemin:>11.2e}{nemax:>11.2e}{contrast:>10.1e}"
              f"{pemax:>9.2e}{pimax:>9.2e}{nti:>11d}")
        if nemin < worst_hole[1]: worst_hole = [name, nemin]
        if max(pemax, pimax) > worst_blow[1]: worst_blow = [name, max(pemax, pimax)]
        if contrast > worst_contrast[1]: worst_contrast = [name, contrast]

    # Global domain contrast drives the Apar warning: the Naulin solve spans the
    # whole domain, so it's max(Ne)/min(Ne) over the domain (not per-region) that
    # conditions it. The per-region table above localises WHERE the min sits.
    dm = xdom[:, None, None] & ydom[None, :, None]
    ne_dom = np.where(dm, Ne, np.nan)
    gmin = float(np.nanmin(ne_dom)); gmax = float(np.nanmax(ne_dom))
    gcontrast = gmax / gmin if gmin > 0 else np.inf

    print("\nverdict:")
    print(f"  lowest density : {worst_hole[0]} (min Ne = {worst_hole[1]:.2e} norm)")
    print(f"  global domain contrast max/min Ne = {gcontrast:.1e}")
    if gcontrast > APAR_CONTRAST_WARN:
        print(f"  WARN  EM Apar Naulin risk: domain contrast {gcontrast:.1e} > "
              f"{APAR_CONTRAST_WARN:.0e} (min in {worst_hole[0]}) → raise density_floor "
              f"(clamp the holes) or switch Apar to Boussinesq.")
    if worst_blow[1] > BLOWUP_WARN:
        print(f"  WARN  pressure BLOW-UP: {worst_blow[0]} P_max {worst_blow[1]:.1e} ≫ ~5 → "
              f"heat-trapping (check kappa_limit) / thermal runaway.")
    print(f"  domain cells below density_floor({DENSITY_FLOOR:.0e}): "
          f"{int(np.nansum(np.where(dm, Ne < DENSITY_FLOOR, False)))}")
    ds.close()

# ---------------------------------------------------------------------------
# log_autopsy — classify how a run ended from BOUT.log.* (all ranks): which
# solver/term failed, and the CVODE trajectory (step, local-error fails LEF,
# nonlinear-convergence fails NCF) → timeout vs Newton-collapse vs solver-
# divergence vs first-step. Run this FIRST on any failed case.
# ---------------------------------------------------------------------------
elif opt_plt == "log_autopsy":
    import glob, re
    gr.close()
    logs = sorted(glob.glob(os.path.join(prepath, "BOUT.log.*")))
    if not logs:
        log.error("no BOUT.log.* under %s", prepath); raise SystemExit(1)

    patterns = {
        "EM Apar Naulin not converged": r"LaplaceNaulin error: Not converged",
        "conduction-precon zero pivot": r"Zero pivot in CyclicReduce",
        "ddt non-finite (RHS term)":    r"ddt\([^)]*\) non-finite[^\n]*",
        "wall-limit timeout":           r"User signalled to quit|Quitting",
    }
    print(f"log_autopsy: scanning {len(logs)} rank logs under {prepath}")
    hits = {}; example = {}
    for fn in logs:
        try:
            txt = open(fn, errors="ignore").read()
        except OSError:
            continue
        for label, pat in patterns.items():
            m = re.search(pat, txt)
            if m:
                hits[label] = hits.get(label, 0) + 1
                example.setdefault(label, (os.path.basename(fn), m.group(0)[:80]))
    print("\nexception scan (rank-log count):")
    for label in patterns:
        if label in hits:
            print(f"  {hits[label]:>4d} ranks  {label}   e.g. {example[label][1]!r} ({example[label][0]})")
    if not hits:
        print("  (no exception/timeout pattern matched — clean finish or killed externally)")

    log0 = os.path.join(prepath, "BOUT.log.0")
    traj = []
    if os.path.exists(log0):
        t = open(log0, errors="ignore").read()
        for ns, step, order, lef, ncf in re.findall(
                r"CVODE: nsteps\s+(\d+).*?Last step size:\s+([\d.eE+-]+), order:\s+(\d+)"
                r".*?Local error fails:\s+(\d+), nonlinear convergence fails:\s+(\d+)",
                t, re.DOTALL):
            traj.append((int(ns), float(step), int(order), int(lef), int(ncf)))
        print(f"\nCVODE trajectory (rank 0, {len(traj)} outputs):")
        print(f"  {'nsteps':>8}{'step':>11}{'order':>6}{'LEF':>8}{'NCF':>8}")
        for ns, step, order, lef, ncf in traj:
            print(f"  {ns:>8}{step:>11.2e}{order:>6}{lef:>8}{ncf:>8}")

    print("\nverdict:")
    if "wall-limit timeout" in hits:
        print("  → wall-limit TIMEOUT (ran out of time, not a crash). If step is pinned at"
              " max_timestep → raise it; if step is collapsing → fix the stiffness first.")
    if not hits and not traj:
        print("  → NO BOUT exception and NO completed CVODE step. Backtraces are enabled, so a"
              " zero-pivot / Naulin / non-finite WOULD have printed — this is a SILENT death:"
              " external kill (OOM / scheduler time / node fault / MPI launch) or a hang, NOT a"
              " Hermes physics/solver crash. Check the scheduler stderr/out (SLURM .err/.out),"
              " and memory (InvertParDiv allocates on the first precon setup → classic OOM spot).")
    elif hits and not traj:
        print("  → exception with NO completed CVODE step: failed on the FIRST solve"
              " (restart-state / config), not after evolution.")
    elif len(traj) >= 2:
        k = max(0, len(traj) - 3)
        d_ncf = traj[-1][4] - traj[k][4]
        smax = max(s for _, s, _, _, _ in traj)
        step_drop = traj[-1][1] < 0.3 * smax
        dn = max(1, traj[-1][0] - traj[k][0])
        lef_rate = (traj[-1][3] - traj[k][3]) / dn
        if d_ncf > 5 and step_drop:
            print("  → Newton COLLAPSE (NCF climbing + step collapsing): stiff state —"
                  " depletion / floor-machinery / undershoots. inventory_time + field_autopsy next.")
        elif lef_rate > 0.05 and d_ncf <= 5:
            print("  → stiff but CONVERGING (high LEF, NCF≈0, step holding): expensive/slow →"
                  " timeout risk, not a crash. Chunk it.")
        else:
            print("  → advancing; the exception above (if any) is the trigger — field_autopsy"
                  " the crash dump to localise it.")

# ---------------------------------------------------------------------------
# stiffness_map — WHERE the accuracy (local-error) limit bites, HOW hard, and
# WHAT the field looks like there.
#
# CVODE caps dt by the weighted local error
#       w = |ddt(f)| / (rtol*|f| + atol)
# For an unsigned field (N, P) rtol*|f| >> atol, so
#       w = (1/rtol) * |d ln f/dt|        and        dt_limit ~ 1/w
# one number per cell, directly comparable with "Last step size" in BOUT.log.
#
# TRAP: a SIGNED variable near a zero crossing is atol-dominated
# (NVe ~ 5e-8 -> rtol*|f| = 5.3e-13 < atol = 1e-12) and fakes w ~ 1e8, owning
# the maximum. N/P are ranked; NV/Vort are reported and flagged, never ranked.
#
# Structure probe at the worst cells answers "is this a real feature?":
#   zigzag  = |f[i-1] - 2 f[i] + f[i+1]| / (|f[i-1]| + 2|f[i]| + |f[i+1]|)
#             ~0 smooth, -> 1 for a pure 2*dx grid-scale oscillation
#   nyq_z   = Nyquist share of the toroidal spectrum at that (x,y)
#   M, dM   = parallel Mach number and its jump across the y-faces (shock)
# ---------------------------------------------------------------------------
elif opt_plt == "stiffness_map":
    RTOL, ATOL = 1e-5, 1e-12   # must match [solver] rtol/atol (BOUT++ defaults)
    RANK_VARS  = ["Ne", "Pe", "Ni", "Pi"]   # unsigned only — see TRAP above
    NTOP       = 5             # worst cells to probe for 3-D structure
    FRAME      = frame         # time index within the dump ([debug] frame in the ini)
    AA_ION     = 2.0           # NV = A*N*V (verified: (NVi/Ni)/Vi = 2.0)

    if gr is not None:
        gr.close()
    ds = nc.Dataset(prepath + DATANAME)
    nxf = ds.dimensions["x"].size
    nyf = ds.dimensions["y"].size
    nzf = ds.dimensions["z"].size
    MXG = int(np.array(ds["MXG"][...]))
    MYGd = int(np.array(ds["MYG"][...]))
    ny_d = int(np.array(ds["ny"][...]))
    nyi = int(np.array(ds["ny_inner"][...]))
    ix1 = int(np.array(ds["ixseps1"][...]))
    j11, j21, j12, j22 = (int(np.array(ds[k][...])) for k in
                          ("jyseps1_1", "jyseps2_1", "jyseps1_2", "jyseps2_2"))

    # Domain mask: drop x guards and the y-boundary guard rows (target +
    # X-point branch-cut rows), which carry extrapolated garbage.
    g = MYGd if nyf > ny_d else 0
    ym = np.zeros(nyf, bool)
    if 0 < nyi < ny_d:
        ym[g:nyi + g] = True
        ym[nyi + 3 * g:nyf - g] = True
    else:
        ym[g:nyf - g] = True
    xm = np.zeros(nxf, bool)
    xm[MXG:nxf - MXG] = True
    dom = (xm[:, None] & ym[None, :])[:, :, None]

    def dumpy(ygrid):
        """Grid y-index -> dump y-index (guard offsets differ per region)."""
        return ygrid + g if ygrid < nyi else ygrid + 3 * g

    def same_seg(ya, yb):
        """True if two dump y-indices lie in the SAME poloidal segment.

        In a double null the inner (y < ny_inner) and outer segments are
        physically disconnected — they meet only at the upper targets. A
        stencil or difference straddling that join is meaningless (it compares
        two different divertor legs), as is one crossing a target guard row.
        """
        if not (ym[ya] and ym[yb]):
            return False
        inner_a = ya < nyi + g
        inner_b = yb < nyi + g
        return inner_a == inner_b

    def region_of(y):
        # The band between jyseps2_1 and jyseps1_2 holds BOTH upper legs, split
        # at ny_inner: below it the inner-upper leg, above it the outer-upper
        # leg. They are physically separate divertor legs (see same_seg), so
        # they are reported separately.
        yy = y - g if y < nyi + g else y - 3 * g
        if yy <= j11:  return "inner-lower leg"
        if yy <= j21:  return "inner core/SOL"
        if yy <  j12:  return "upper-inner leg" if yy < nyi else "upper-outer leg"
        if yy <= j22:  return "outer core/SOL"
        return "outer-lower leg"

    t_arr = np.array(ds["t_array"][:])
    nt = len(t_arr)
    print(f"stiffness_map: {prepath + DATANAME}  frame {FRAME} of {nt}  "
          f"t={t_arr[FRAME]:.1f}")
    print(f"  rtol={RTOL:g} atol={ATOL:g}   domain cells={int(dom.sum())}"
          f"   (nx={nxf} ny={nyf} nz={nzf}, y-guards {'kept' if g else 'absent'})")

    # |phi|max per frame. A polarisation_drift blowup drags the stiffness
    # maximum out of the divertor and into the core, so the map is only telling
    # you about the *divertor* while phi is still healthy (~1-2, Dudson).
    if "phi" in ds.variables:
        pm = np.array([np.nanmax(np.where(dom, np.abs(np.array(ds["phi"][k])), np.nan))
                       for k in range(nt)])
        sel = pm[FRAME]
        print("  |phi|max per frame: "
              + "  ".join(f"{v:.1f}" + ("*" if k == (FRAME % nt) else "")
                          for k, v in enumerate(pm)))
        if sel > 5.0:
            first_ok = next((k for k in range(nt) if pm[k] <= 5.0), None)
            print(f"  ** WARNING: |phi|max = {sel:.1f} at the selected frame "
                  f"(healthy is ~1-2). **")
            print( "  ** The phi runaway dominates ddt in the CORE and will mask the "
                   "divertor. **")
            if first_ok is not None:
                print(f"  ** For the divertor picture set FRAME = {first_ok} "
                      f"(|phi|max = {pm[first_ok]:.1f}, t = {t_arr[first_ok]:.0f}). **")
            else:
                print( "  ** No frame in this dump is healthy — use an earlier folder. **")
    print()

    # --- weight per variable -------------------------------------------------
    print(f"  {'variable':>9} {'max w':>11} {'dt_limit':>10}  {'location':>22}  note")
    wmax_all, best = None, None
    for v in VAR_LIST:
        dname = f"ddt({v})"
        if dname not in ds.variables or v not in ds.variables:
            continue
        f0 = np.array(ds[v][FRAME])
        d0 = np.array(ds[dname][FRAME])
        den = RTOL * np.abs(f0)
        w = np.where(dom, np.abs(d0) / (den + ATOL), np.nan)
        i = np.unravel_index(np.nanargmax(w), w.shape)
        wm = float(np.nanmax(w))
        ranked = v in RANK_VARS
        natol = int(np.nansum(np.where(dom, den < ATOL, False)))
        note = "ranked" if ranked else "NOT ranked (signed)"
        if natol:
            note += f"; {natol} atol-dominated cells"
        print(f"  {v:>9} {wm:11.3e} {1.0 / wm:10.2e}  "
              f"x={i[0]:3d} y={i[1]:2d} z={i[2]:2d}  {note}")
        if ranked and (wmax_all is None or wm > wmax_all):
            wmax_all, best = wm, (v, w, f0)

    if best is None:
        print("  no ranked variable has a saved ddt(*) — nothing to map.")
        ds.close()
        raise SystemExit(0)

    vname, w, fld = best
    print(f"\n  -> dt is set by {vname}: max w = {wmax_all:.3e}  =>  dt_limit ~ "
          f"{1.0 / wmax_all:.2e} (normalised)")

    # A density or pressure has no business being negative. When it is, the
    # weight at that cell is set by atol rather than by any rate of change, so
    # the huge w is partly an artifact — but the sign is a real red flag, and
    # the cell is usually ringing at the grid scale beside a steep front.
    iw = np.unravel_index(np.nanargmax(w), w.shape)
    if RTOL * abs(fld[iw]) < ATOL:
        print(f"     ** the winning cell is ATOL-DOMINATED: {vname} = {fld[iw]:.3e} "
              f"(rtol*|f| = {RTOL * abs(fld[iw]):.2e} < atol = {ATOL:.0e}) **")
        print( "     ** w there is set by the atol floor, not by a rate. Treat the "
               "VALUE, not w, as the finding. **")
    neg = np.where(dom, fld < 0, False)
    if neg.any():
        ys = np.unique(np.argwhere(neg)[:, 1])
        print(f"     ** {int(neg.sum())} domain cells have {vname} < 0 "
              f"(min {np.nanmin(np.where(dom, fld, np.nan)):.3e}), at y = "
              f"{', '.join(str(int(v)) for v in ys)} "
              f"[{', '.join(sorted({region_of(int(v)) for v in ys}))}] **")
    print("     NOTE: dt_limit is a FIRST-ORDER, PER-CELL proxy and runs ~100x below the\n"
          "     step CVODE actually takes (measured: 94x on SXD es_v2_88, 158x on CD 166).\n"
          "     Two reasons: CVODE integrates at order 2-3, and its test is a weighted RMS\n"
          "     over all cells, so one bad cell out of N is diluted by ~sqrt(N). Read w as a\n"
          "     RELATIVE ranking (which cell, which region, how much worse) — not as a dt.")

    # --- regional breakdown --------------------------------------------------
    print(f"\n  {'region':>17} {'max w':>10} {'dt_limit':>9} {'99th pct':>10} "
          f"{'max/p99':>8} {'min N':>9}  worst cell")
    print("     max/p99 ~1-3 = the whole region is this stiff (no local fix will help);\n"
          "     >>10 = a few rogue cells carry it, worth chasing individually.")
    for r in ["inner-lower leg", "inner core/SOL", "upper-inner leg",
              "upper-outer leg", "outer core/SOL", "outer-lower leg"]:
        sel = np.array([ym[y] and region_of(y) == r for y in range(nyf)])
        if not sel.any():
            continue
        m = (xm[:, None] & sel[None, :])[:, :, None]
        ww = np.where(m, w, np.nan)
        if not np.isfinite(ww).any():
            continue
        i = np.unravel_index(np.nanargmax(ww), ww.shape)
        Ne0 = np.where(m, np.array(ds["Ne"][FRAME]), np.nan)
        side = "core" if i[0] < ix1 else "SOL"
        p99 = np.nanpercentile(ww, 99)
        ratio = np.nanmax(ww) / p99 if p99 > 0 else np.nan
        print(f"  {r:>17} {np.nanmax(ww):10.2e} {1 / np.nanmax(ww):9.1e} "
              f"{p99:10.2e} {ratio:8.1f} {np.nanmin(Ne0):9.2e}  "
              f"x={i[0]:3d}({side}) y={i[1]:2d}")

    # --- 3-D structure at the worst cells ------------------------------------
    def zigzag(a):
        """2*dx oscillation share of a 3-point stencil (0 smooth, 1 pure zigzag)."""
        lo, mid, hi = a
        den = abs(lo) + 2 * abs(mid) + abs(hi)
        return abs(lo - 2 * mid + hi) / den if den > 0 else np.nan

    Ne_f = np.array(ds["Ne"][FRAME])
    Pe_f = np.array(ds["Pe"][FRAME])
    Pi_f = np.array(ds["Pi"][FRAME]) if "Pi" in ds.variables else Pe_f
    Ni_f = np.array(ds["Ni"][FRAME]) if "Ni" in ds.variables else Ne_f
    cs = np.sqrt(np.maximum(Pe_f + Pi_f, 0) / np.maximum(AA_ION * Ni_f, 1e-10))
    if "Vi" in ds.variables:
        Vpar = np.array(ds["Vi"][FRAME])
    elif "NVi" in ds.variables:
        Vpar = np.array(ds["NVi"][FRAME]) / np.maximum(AA_ION * Ni_f, 1e-10)
    else:
        Vpar = None
    Mach = Vpar / np.maximum(cs, 1e-10) if Vpar is not None else None

    flat = np.where(np.isfinite(w), w, -np.inf).ravel()
    tops = np.array(np.unravel_index(np.argsort(flat)[::-1][:NTOP], w.shape)).T
    print(f"\n  structure at the {NTOP} stiffest cells ({vname}); "
          f"zigzag: ~0 smooth, ->1 grid-scale 2dx oscillation")
    print(f"  {'x':>4}{'y':>4}{'z':>4} {'w':>10} {'zz_x':>6}{'zz_y':>6}{'zz_z':>6}"
          f"{'nyq_z':>7} {'M':>7}{'dM_y':>7}  verdict")
    for (ix, iy, iz) in tops:
        zx = zigzag(fld[ix - 1:ix + 2, iy, iz]) if 0 < ix < nxf - 1 else np.nan
        okp = iy > 0 and same_seg(iy, iy - 1)
        okn = iy < nyf - 1 and same_seg(iy, iy + 1)
        zy = zigzag(fld[ix, iy - 1:iy + 2, iz]) if (okp and okn) else np.nan
        zz = zigzag(np.take(fld[ix, iy], [iz - 1, iz, (iz + 1) % nzf], mode="wrap"))
        sp = np.abs(np.fft.rfft(fld[ix, iy] - fld[ix, iy].mean()))
        nyq = sp[-1] / sp[1:].sum() if sp[1:].sum() > 0 else np.nan
        if Mach is not None:
            M = Mach[ix, iy, iz]
            lo = iy - 1 if okp else iy
            hi = iy + 1 if okn else iy
            # only a within-segment difference is a real parallel gradient
            dM = Mach[ix, hi, iz] - Mach[ix, lo, iz] if hi != lo else np.nan
        else:
            M = dM = np.nan
        bits = []
        if max(np.nan_to_num(zx), np.nan_to_num(zy), np.nan_to_num(zz)) > 0.3:
            bits.append("GRID-SCALE zigzag")
        if np.nan_to_num(nyq) > 0.25:
            bits.append("Nyquist-heavy in z")
        if abs(np.nan_to_num(dM)) > 0.5:
            bits.append("parallel shock/jump")
        if not bits:
            bits.append("smooth — genuinely fast physics")
        print(f"  {ix:>4}{iy:>4}{iz:>4} {w[ix, iy, iz]:10.2e} {zx:6.2f}{zy:6.2f}{zz:6.2f}"
              f"{nyq:7.2f} {M:7.2f}{dM:7.2f}  {', '.join(bits)}  [{region_of(iy)}]")

    # --- plots ---------------------------------------------------------------
    with np.errstate(invalid="ignore"):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)   # x-guard rows are all-NaN
            wmap = np.nanmax(w, axis=2)                 # worst over toroidal angle
            wrad = np.nanmax(wmap, axis=1)
    bx, by, bz = tops[0]
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.5))

    im = ax[0, 0].pcolormesh(np.arange(nyf), np.arange(nxf), np.log10(wmap),
                             shading="nearest", cmap="inferno")
    fig.colorbar(im, ax=ax[0, 0], label=r"$\log_{10}\,w$")
    ax[0, 0].axhline(ix1, color="cyan", lw=1.2, ls="--", label="separatrix")
    for jy in (j11, j21, j12, j22):
        ax[0, 0].axvline(dumpy(jy) + 0.5, color="w", lw=0.8, ls=":")
    ax[0, 0].plot(tops[:, 1], tops[:, 0], "co", ms=5, mfc="none", label=f"{NTOP} stiffest")
    ax[0, 0].set_xlabel("y (poloidal index)")
    ax[0, 0].set_ylabel("x (radial index)")
    ax[0, 0].set_title(f"local-error weight, max over z  ({vname}, t={t_arr[FRAME]:.0f})")
    ax[0, 0].legend(fontsize=8, loc="upper right")

    ax[0, 1].semilogy(np.arange(nxf), wrad, lw=1.3)
    ax[0, 1].axvline(ix1, color="k", ls="--", lw=1, label="separatrix")
    ax[0, 1].set_xlabel("x (radial index)")
    ax[0, 1].set_ylabel("max w over (y, z)")
    ax[0, 1].set_title("radial profile of the accuracy limit")
    ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(np.arange(nzf), fld[bx, by], ".-", lw=1)
    ax[1, 0].set_xlabel("z (toroidal index)")
    ax[1, 0].set_ylabel(vname)
    ax[1, 0].set_title(f"toroidal cut at the stiffest cell (x={bx}, y={by}) — "
                       f"zigzag here?")
    ax[1, 0].grid(alpha=0.3)

    segs = [np.array([y for y in range(nyf) if ym[y] and y < nyi + g]),
            np.array([y for y in range(nyf) if ym[y] and y >= nyi + g])]
    a2 = ax[1, 1].twinx() if Mach is not None else None
    for k, seg in enumerate(segs):
        if seg.size == 0:
            continue
        # segments are physically disconnected — never join them with a line
        ax[1, 1].plot(seg, fld[bx, seg, bz], ".-", lw=1, color="C0",
                      label=vname if k == 0 else None)
        if a2 is not None:
            a2.plot(seg, Mach[bx, seg, bz], "r.-", lw=1, alpha=0.7,
                    label="Mach" if k == 0 else None)
    if a2 is not None:
        a2.set_ylabel("parallel Mach number", color="r")
        a2.axhline(1, color="r", ls=":", lw=0.8)
        a2.axhline(-1, color="r", ls=":", lw=0.8)
    if segs[0].size and segs[1].size:
        ax[1, 1].axvspan(segs[0][-1], segs[1][0], color="0.85", zorder=0)
        ax[1, 1].text(0.5 * (segs[0][-1] + segs[1][0]), ax[1, 1].get_ylim()[1],
                      "upper\ntargets", ha="center", va="top", fontsize=7, color="0.35")
    ax[1, 1].axvline(by, color="k", ls="--", lw=0.8)
    ax[1, 1].set_xlabel("y (poloidal index, guards removed)")
    ax[1, 1].set_ylabel(vname)
    ax[1, 1].set_title(f"parallel cut at (x={bx}, z={bz}) — inner | outer segments "
                       f"are disconnected")
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
    ds.close()

else:
    log.error("Unknown opt_plt=%r — nothing to do.", opt_plt)
    raise SystemExit(1)
