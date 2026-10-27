#!/usr/bin/env python3
"""
Poloidal flux diagnostic for a Hermes-3 3D turbulence simulation.

Toroidal- and time-mean poloidal flux density split into its physical
contributions, for both particles and energy:

    parallel flow         :  n · v_{∥,pol}      (dominant in SOL)
    ExB poloidal drift    :  n · v_E,pol
    magnetic-drift pol.   :  ± n·T · curl(b/B)_pol     (electron / ion)
    parallel conduction   :  −κ_∥ · (∂T/∂s_∥)·(Bp/B)   (heat only; if κ_∥ saved)

Two figures per quantity (particle, energy):
    A) Stacked 2-D (x, θ) panels of each contribution + total.
    B) 1-D radially-integrated profile vs poloidal index, integrated over the
       open-field-line band (θ-dependent x_in: ixseps1 in the main SOL,
       inner wall in the legs/PFR) — useful for tracking the build-up of flux
       from the midplane down to the targets.

Caching
-------
The heavy load + flux computation is cached to ./cache/v3_poloidal_flux_*.pkl.
On rerun with the same dir_range / species the plot is regenerated from the
pickle in seconds. Set use_cache = False to force recomputation.

Edit the CONFIG block and run.
"""

import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import hermes3_turb as h3t
from hermes3_turb.load import get_parallel_velocity
from xhermes import open_hermesdataset


# ── CONFIG ────────────────────────────────────────────────────────────────────
data_root    = "./"
#gridfile     = "../exp45470_x132_y28.nc"
gridfile     = data_root+"../new_sxd_nx132_ny36.nc"
dir_range    = range(140, 141)
keep_yguards = True
species      = "e"
use_cache    = True
# ─────────────────────────────────────────────────────────────────────────────
dir_range = h3t.existing_dirs(data_root, dir_range, "boutdata_guards.nc")


# ── IPython-rerun safety: close stale handles before reloading ───────────────
plt.close("all")
for _name in ("ds", "datasets", "ds_i"):
    _v = globals().pop(_name, None)
    if _v is None:
        continue
    if hasattr(_v, "close"):
        try:    _v.close()
        except Exception: pass
    elif isinstance(_v, list):
        for _d in _v:
            if hasattr(_d, "close"):
                try:    _d.close()
                except Exception: pass
    del _v
gc.collect()


def _compute():
    """Load datasets, compute every 2-D poloidal flux density and 1-D
    radially-integrated profile. Returns a flat dict ready for pickling."""
    # ── Load (NO geometry / gridfile) ────────────────────────────────────────
    # The xhermes geometry="toroidal" + gridfilepath path fails when the grid
    # was generated without y-guards but the dump retains them (Hermes-3
    # extrapolates into the guards, so dump_ny > grid_ny). Skip geometry —
    # all metric tensor / Bp / Bxy are read directly from the dump's own
    # variables, so neither the analysis nor the magnetic-drift load below
    # depends on the grid being applied. load_grid_vars(gridfile, …) is still
    # used for bxcvx / bxcvy because those are not always written to the dump.
    datasets = []
    for i, dirnum in enumerate(dir_range):
        print(f"loading {data_root}{dirnum}")
        ds_i = open_hermesdataset(
            #f"{data_root}/{dirnum}/boutdata.nc",
            f"{data_root}/{dirnum}/boutdata_guards.nc",
            inputfilepath="BOUT.inp",
            keep_xboundaries=True,
            keep_yboundaries=keep_yguards,
            unnormalise=False,
            info=False,
        )
        datasets.append(ds_i)
    # NB do NOT drop each folder's first frame: that assumes BOUT++'s default
    # dump_at_restart=true, and with dump_at_restart=false the first frame is a
    # genuinely NEW time, so the positional rule deletes real data (and a folder
    # squashed to a single output time contributes nothing at all). Restart
    # overlap is removed after concatenation by time VALUE -- see
    # load.dedup_restart_times, which handles both settings and mixed ones.
    ds = xr.concat(datasets, dim="t")
    ds = h3t.dedup_restart_times(ds)

    zdim = h3t.detect_toroidal_dim(ds)
    pdim = h3t.detect_poloidal_dim(ds)

    # ── Drop y-boundary guard rows if the load kept them ──────────────────────
    # xBOUT honours the GRID's `y_boundary_guards`, so a grid built with
    # y_boundary_guards=2 returns a guard-retaining dataset even when this
    # script asks for keep_yboundaries=False. That is not merely untidy:
    # `.bout.ddy()` returns only the domain rows but RE-LABELS them onto the
    # FIRST n_domain theta values rather than the domain rows' own labels.
    # xarray then inner-joins on those labels inside `ExB_radial_velocity`, so
    # each domain cell's ∂φ/∂θ is paired with the ∂φ/∂ζ and metric of a
    # *different* cell — offset by the guards, and wrong everywhere past the
    # first guard pair, not just at the ends.
    # Visible symptom: Fig 1 dies in pcolormesh with
    #   Dimensions of C (244, 112) should be one smaller than X(120)
    # Invisible symptom: a mis-registered ExB flux, plus `Gamma_mag_*` left on
    # the full 120 axis (load_grid_vars pads to the dataset) and `j1/j2`
    # computed for the guarded layout — so the θ-band of every 1-D profile is
    # taken from the wrong rows.
    # Measured on from_archer_6kW/data/396 (grid ny=112, MYG=2 → 120 rows):
    # ddy returns theta == full[:112] exactly, confirming the re-labelling.
    # Guard layout is this codebase's usual one: region A at +MYG, B at +3·MYG.
    import netCDF4 as _nc_g
    with _nc_g.Dataset(gridfile) as _g0:
        _ny_g = int(_g0["ny"][:])
    _MYG0 = int(ds.metadata.get("MYG", 2))
    _nyi0 = int(ds.metadata.get("ny_inner", 0) or 0)
    _npol = ds.sizes[pdim]
    if _nyi0 > 0 and _npol == _ny_g + 4 * _MYG0:            # double null
        _guard_rows = (list(range(_MYG0))
                       + list(range(_nyi0 + _MYG0, _nyi0 + 3 * _MYG0))
                       + list(range(_ny_g + 3 * _MYG0, _ny_g + 4 * _MYG0)))
    elif _npol == _ny_g + 2 * _MYG0:                        # single null
        _guard_rows = (list(range(_MYG0))
                       + list(range(_ny_g + _MYG0, _ny_g + 2 * _MYG0)))
    else:
        _guard_rows = []
    if _guard_rows:
        _keep_rows = [i for i in range(_npol) if i not in set(_guard_rows)]
        print(f"  dataset kept y-guards despite keep_yguards={keep_yguards} "
              f"(grid y_boundary_guards>0): dropping rows {_guard_rows}, "
              f"{_npol} -> {len(_keep_rows)} poloidal cells")
        ds = ds.isel({pdim: _keep_rows})
    m    = ds.metadata
    print(f"Loaded {ds.sizes['t']} time steps")

    # ── Topology (X-points + OMP/IMP + targets) ──────────────────────────────
    import netCDF4 as _nc
    with _nc.Dataset(gridfile) as _grd:
        _ny_grid = int(_grd["ny"][:])
    _has_yguards = ds.sizes[pdim] > _ny_grid
    MYG = int(m.get("MYG", 2)) if _has_yguards else 0
    MXG = int(m.get("MXG", 2))

    ixsep1 = m.get("ixseps1", ds.sizes["x"] // 2)
    j1_1   = m.get("jyseps1_1", MYG) + MYG + 1
    j2_1   = m.get("jyseps2_1", ds.sizes[pdim] // 4) + MYG + 1
    j1_2   = m.get("jyseps1_2", ds.sizes[pdim] // 2) + MYG * 3 + 1
    j2_2   = m.get("jyseps2_2", 3 * ds.sizes[pdim] // 4) + MYG * 3 + 1
    ny_inn = m.get("ny_inner", None)
    if ny_inn is not None:
        ny_inn = ny_inn + MYG * 2

    y_omp = (j1_2 + j2_2) // 2
    y_imp = (j1_1 + j2_1) // 2

    y_targets = {
        "inner_lower": MYG,
        "outer_lower": ds.sizes[pdim] - 1 - MYG,
    }
    if ny_inn is not None and ny_inn > MYG and ny_inn < ds.sizes[pdim] - MYG:
        y_targets["inner_upper"] = ny_inn - 1
        y_targets["outer_upper"] = ny_inn

    print(f"ixseps1={ixsep1}, OMP y={y_omp}, IMP y={y_imp}, targets={y_targets}")
    print(f"X-points: j1_1={j1_1}, j2_1={j2_1}, j1_2={j1_2}, j2_2={j2_2}")

    # ── Fields ────────────────────────────────────────────────────────────────
    n_name = f"N{species}"
    P_name = f"P{species}"
    is_electron = (species == "e")
    ion_species = "i" if "Ni" in ds.data_vars else "d+"

    n   = ds[n_name]
    P   = ds[P_name]
    phi = ds["phi"]
    Ne  = ds["Ne"]; Ni = ds[f"N{ion_species}"]
    Pe  = ds["Pe"]; Pi = ds[f"P{ion_species}"]
    # Te/Ti are in the dump only if the input asks for them; a folder that
    # did not leaves whole NaN frames after the concat (nx32 dirs 23-28:
    # folder 23 has no Ti), so fill those frames from P/N.
    Te_eV = Pe / Ne.clip(min=1e-10)
    Ti_eV = Pi / Ni.clip(min=1e-10)
    if "Te" in ds:
        Te_eV = ds["Te"].fillna(Te_eV)
    if f"T{ion_species}" in ds:
        Ti_eV = ds[f"T{ion_species}"].fillna(Ti_eV)
    v_par = get_parallel_velocity(ds, species)

    grid_vars = h3t.load_grid_vars(gridfile, ds=ds, pdim=pdim)
    bxcvx, bxcvy, Bxy = grid_vars["bxcvx"], grid_vars["bxcvy"], grid_vars["Bxy"]

    # ── Poloidal flux contributions (2-D, x×θ after time+toroidal mean) ──────
    def tmean(da):
        return da.mean("t") if "t" in getattr(da, "dims", ()) else da

    def zmean(da):
        return h3t.toroidal_mean(da, dim=zdim) if zdim in getattr(da, "dims", ()) else da

    print("Computing poloidal flux contributions ...")
    v_par_pol = h3t.parallel_to_poloidal(ds, v_par)
    v_ExB_pol = h3t.ExB_poloidal_velocity(ds, phi)

    F_par_pol = tmean(zmean(n * v_par_pol))
    F_ExB_pol = tmean(zmean(n * v_ExB_pol))
    Q_par_pol = tmean(zmean(2.5 * P * v_par_pol))
    Q_ExB_pol = tmean(zmean(2.5 * P * v_ExB_pol))

    Gmag_pol_e, Gmag_pol_i = h3t.poloidal_magnetic_drift_particle_flux(
        ds, Ne, Ni, bxcvx, bxcvy, Bxy, Te_eV, Ti_eV)
    qmag_pol_e, qmag_pol_i = h3t.poloidal_magnetic_drift_heat_flux(
        ds, Pe, Pi, bxcvx, bxcvy, Bxy, Te_eV, Ti_eV)
    F_mag_pol = tmean(Gmag_pol_e if is_electron else Gmag_pol_i)
    Q_mag_pol = tmean(qmag_pol_e if is_electron else qmag_pol_i)

    # ── Parallel conduction ──────────────────────────────────────────────────
    # Manual centered FD on the field-aligned T (matches the pattern used in
    # target_poloidal_check.py). We can't use `T.bout.ddy()` here because
    # without geometry="toroidal" applied at load, the bout accessor's
    # field-aligned check (`direction_y != "Aligned"`) raises. So shift T
    # explicitly via h3t.to_field_aligned (FFT-based, geometry-independent)
    # and difference it.
    kpar_name = f"kappa_par_{species}"
    T_name    = f"T{species}"
    if kpar_name in ds and T_name in ds:
        T_aligned = h3t.to_field_aligned(ds[T_name].clip(min=0.1), ds)
        # Centered ∂T/∂y on the field-aligned grid, divided by 2·dy.
        DTY = (T_aligned.roll({pdim: -1}, roll_coords=False)
               - T_aligned.roll({pdim:  1}, roll_coords=False)
              ) / (2.0 * ds["dy"])
        q_cond = (
            -ds[kpar_name] * DTY / ds["g_22"]
            * (ds["g_33"] * ds["g_22"] - ds["g_23"] ** 2)
            / (ds["J"] * np.sqrt(ds["g11"] * ds["g_33"]))
        )
        Q_cond_pol = tmean(zmean(q_cond))
    else:
        print(f"  ({kpar_name}/{T_name} absent — poloidal conduction omitted)")
        Q_cond_pol = xr.zeros_like(Q_par_pol)

    F_pol = F_par_pol + F_ExB_pol + F_mag_pol
    Q_pol = Q_par_pol + Q_ExB_pol + Q_mag_pol + Q_cond_pol

    # ── 1-D radially-integrated profiles vs θ (open-field-line band) ─────────
    ny        = ds.sizes[pdim]
    x_out_loc = ds.sizes["x"] - 1 - MXG
    x_in_arr  = np.full(ny, MXG, dtype=int)
    x_in_arr[j1_1:j2_1] = ixsep1
    x_in_arr[j1_2:j2_2] = ixsep1

    _elem_pol = h3t.poloidal_area_element_dx(ds)

    def _radial_sum_open_field(F_pol_density):
        integ = F_pol_density * _elem_pol
        extra = [d for d in integ.dims if d not in ("x", pdim)]
        if extra:
            integ = integ.mean(extra)
        col = integ.transpose("x", pdim).values
        prof = np.array([col[x_in_arr[j]:x_out_loc + 1, j].sum() for j in range(ny)])
        return prof * 2 * np.pi

    F_par_prof = _radial_sum_open_field(F_par_pol)
    F_ExB_prof = _radial_sum_open_field(F_ExB_pol)
    F_mag_prof = _radial_sum_open_field(F_mag_pol)
    F_tot_prof = _radial_sum_open_field(F_pol)
    Q_par_prof  = _radial_sum_open_field(Q_par_pol)
    Q_ExB_prof  = _radial_sum_open_field(Q_ExB_pol)
    Q_mag_prof  = _radial_sum_open_field(Q_mag_pol)
    Q_cond_prof = _radial_sum_open_field(Q_cond_pol)
    Q_tot_prof  = _radial_sum_open_field(Q_pol)

    return dict(
        # 2-D (x, θ)
        F_par_pol=np.asarray(F_par_pol.transpose("x", pdim).values),
        F_ExB_pol=np.asarray(F_ExB_pol.transpose("x", pdim).values),
        F_mag_pol=np.asarray(F_mag_pol.transpose("x", pdim).values),
        F_pol=np.asarray(F_pol.transpose("x", pdim).values),
        Q_par_pol=np.asarray(Q_par_pol.transpose("x", pdim).values),
        Q_ExB_pol=np.asarray(Q_ExB_pol.transpose("x", pdim).values),
        Q_mag_pol=np.asarray(Q_mag_pol.transpose("x", pdim).values),
        Q_cond_pol=np.asarray(Q_cond_pol.transpose("x", pdim).values),
        Q_pol=np.asarray(Q_pol.transpose("x", pdim).values),
        # 1-D
        F_par_prof=F_par_prof, F_ExB_prof=F_ExB_prof,
        F_mag_prof=F_mag_prof, F_tot_prof=F_tot_prof,
        Q_par_prof=Q_par_prof, Q_ExB_prof=Q_ExB_prof,
        Q_mag_prof=Q_mag_prof, Q_cond_prof=Q_cond_prof,
        Q_tot_prof=Q_tot_prof,
        # coords + topology
        theta_coord=ds[pdim].values,
        ny=int(ny), MXG=int(MXG),
        ixsep1=int(ixsep1),
        j1_1=int(j1_1), j2_1=int(j2_1),
        j1_2=int(j1_2), j2_2=int(j2_2),
        y_omp=int(y_omp), y_imp=int(y_imp),
        y_targets={k: int(v) for k, v in y_targets.items()},
        species=species,
    )


# ── Cache load or compute ────────────────────────────────────────────────────
_cache_pkl = h3t.cache_path("v3_poloidal_flux_d2g", dir_range, species,
                            "t2")
                                    # t2: Te/Ti NaN frames filled from P/N
if use_cache and _cache_pkl.exists():
    print(f"Cache hit: {_cache_pkl}")
    _C = h3t.load_cache(_cache_pkl)
else:
    _C = _compute()
    h3t.save_cache(_cache_pkl, **_C)
    print(f"Saved cache: {_cache_pkl}")
globals().update(_C)


# ── 2D (x, θ) plots ──────────────────────────────────────────────────────────
def _sym_lim(arr):
    a = float(np.nanmax(np.abs(arr)))
    return -a, a

def plot_panel(ax, arr, title):
    """arr is already in (x, θ) order from the cache."""
    vmin, vmax = _sym_lim(arr)
    m = ax.contourf(arr, levels=21, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(m, ax=ax, shrink=0.85)
    x_outer = arr.shape[0] - 1 - MXG
    ax.axhline(ixsep1,  color="white", lw=1.2)
    ax.axhline(x_outer, color="white", lw=1.0)
    for jj in (j1_1, j2_1, j1_2, j2_2):
        if 0 <= jj < arr.shape[1]:
            ax.axvline(jj, color="black", ls=":", lw=1.0, alpha=0.9)
    for jj in y_targets.values():
        if 0 <= jj < arr.shape[1]:
            ax.axvline(jj, color="black", ls="-", lw=1.0, alpha=0.7)
    ax.set_xlabel("poloidal index θ")
    ax.set_ylabel("radial index x")
    ax.set_title(title, fontsize=10)


def make_2d_figure(qty_name, F_par, F_ExB, F_mag, F_tot, unit, F_cond=None):
    npan = 5 if F_cond is not None else 4
    fig, ax = plt.subplots(1, npan, figsize=(4.5 * npan, 5))
    plot_panel(ax[0], F_par, f"parallel  [{unit}]")
    plot_panel(ax[1], F_ExB, f"ExB poloidal  [{unit}]")
    plot_panel(ax[2], F_mag, f"mag-drift poloidal  [{unit}]")
    k = 3
    if F_cond is not None:
        plot_panel(ax[3], F_cond, f"conduction  [{unit}]")
        k = 4
    plot_panel(ax[k], F_tot, f"total  [{unit}]")
    fig.suptitle(f"Poloidal {qty_name} flux density — species '{species}' "
                 "(toroidal- and time-mean)")
    fig.tight_layout()
    return fig


fig_part_2d = make_2d_figure("particle", F_par_pol, F_ExB_pol, F_mag_pol, F_pol, "norm")
fig_heat_2d = make_2d_figure("heat",     Q_par_pol, Q_ExB_pol, Q_mag_pol, Q_pol, "norm",
                             F_cond=Q_cond_pol)


# ── 1D profile plots (track from midplane down to targets) ───────────────────
y_idx = np.arange(ny)

def mark_landmarks(ax):
    labeled = {"X": False, "tgt": False}
    for jj in (j1_1, j2_1, j1_2, j2_2):
        if 0 <= jj < ny:
            ax.axvline(y_idx[jj], color="gray", ls=":", lw=0.8,
                       label="X-points" if not labeled["X"] else None)
            labeled["X"] = True
    for jj, color, name in [(y_omp, "tab:blue", "OMP"), (y_imp, "tab:red", "IMP")]:
        if 0 <= jj < ny:
            ax.axvline(y_idx[jj], color=color, ls="--", lw=1.0, label=name)
    for _name, jj in y_targets.items():
        if 0 <= jj < ny:
            ax.axvline(y_idx[jj], color="black", ls="-", lw=1.0, alpha=0.5,
                       label="targets" if not labeled["tgt"] else None)
            labeled["tgt"] = True

fig_prof, ax_prof = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (par, exb, mag, cond, label, unit) in zip(
        ax_prof,
        [(F_par_prof, F_ExB_prof, F_mag_prof, None,        "Particle", "particles/s"),
         (Q_par_prof, Q_ExB_prof, Q_mag_prof, Q_cond_prof, "Heat",     "W")]):
    total = par + exb + mag + (0.0 if cond is None else cond)
    ax.plot(y_idx, par,   "-",  lw=1.5, label="parallel")
    ax.plot(y_idx, exb,   "--", lw=1.5, label="ExB")
    ax.plot(y_idx, mag,   ":",  lw=1.5, label="mag-drift")
    if cond is not None:
        ax.plot(y_idx, cond, "-.", lw=1.5, label="conduction")
    ax.plot(y_idx, total, "k-", lw=2.5, label="total")
    mark_landmarks(ax)
    ax.set_xlabel("poloidal index θ")
    ax.set_ylabel(f"{label} flux  [{unit}, normalised]")
    ax.set_title(f"{label} — radially-integrated poloidal flux  "
                 f"(open-field-line band, full torus)")
    ax.legend(loc="best", fontsize=8)

fig_prof.tight_layout()
plt.show()
