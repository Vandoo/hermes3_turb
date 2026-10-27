#!/usr/bin/env python3
"""
Interpolate a Hermes-3 run onto a finer grid and write new restart files.

Consolidates the four ad-hoc scripts that did this (`interpolatex_dmp_hm.py`,
`interpolatey_dmp_hm.py`, `interpolatey_dmp_hm_yguards.py`,
`copy_restart_hermes.py`) into one driver with a CONFIG block and a single
`mode`, in the style of the other scripts here.

The workflow
------------
Refining a run is a four-step job, and each step is a `mode` below:

  1. Squash the old run                       -> `<src>/boutdata.nc`
     (`squash_hermes.py`, or `boutdata.squashoutput` directly)
  2. Run the NEW grid for one output step and squash it. That gives a dataset
     of the right shape to write into — copy it to `<dst>/boutdata_write.nc`,
     which is what this script fills. **The template is never the file you
     read**, so a half-finished run cannot corrupt your source.
  3. `mode='interp_y'` and/or `'interp_x'` — interpolate old -> template.
     To go the other way (a cheaper grid for a neutrals run) use
     `mode='coarsen_x'`, one halving per run: 256 -> 128 -> 64.
  4. `mode='to_restart'` — write `BOUT.restart.*` from the filled template.

`mode='check'` reports the topology, region map and index ranges without
touching anything; run it first.

How the interpolation works
---------------------------
**Poloidal (`interp_y`).** Done **region by region** (`ds.regions`), because a
single global interpolation would smear across the X-point branch cuts where
the poloidal index is discontinuous. Within a region the data is first put
into **field-aligned** coordinates (`to_field_aligned`), so the interpolation
follows field lines rather than cutting across them, then a `CubicSpline` in
normalised poloidal index maps old to new, then `from_field_aligned` undoes
the shift before writing. Two guard cells are taken on the source
(`with_guards=2`) so the spline has support at the region ends.

**Radial, refining (`interp_x`).** A `CubicSpline` in normalised poloidal flux
`ψ_n = (ψ − ψ_axis)/(ψ_bdry − ψ_axis)`, per poloidal column, so flux surfaces
map to flux surfaces even when the two grids pack cells differently.

**Radial, coarsening (`coarsen_x`).** Going the other way — 256 → 128 → 64, to
make a neutrals run affordable — is *not* the same operation with a different
target. Sampling a spline at fewer points aliases: everything above the new
Nyquist folds back into the resolved band instead of being removed, and a
cubic can overshoot outside the range of its data. Measured on the SXD
`flutter_111` case at 2:1, the spline overshoots the source maximum of `Ne` by
**1.8 %** — an extremum that was not in the data — and differs from the
conservative answer by 0.4 % of mean `Ne`, most of it cell-to-cell, i.e. the
aliased part.

So `coarsen_x` does a **volume-weighted average over cell overlaps in ψ**:
each destination cell takes `Σ f_i·J_i·Δψ_i / Σ J_i·Δψ_i` over the source
cells it covers. That is a box filter (it removes the sub-grid scales rather
than folding them in), it is bounded by the source values (so `positive_vars`
never needs the `abs()` clamp that spline overshoot forces), and it conserves
the column integral to machine precision — measured `-0.0000 %` against the
spline's `-0.0011 %`. It handles non-nested grids and the strongly non-uniform
radial packing these meshes use (`dψ` varies 15× across the SXD grid), because
the overlap is computed in ψ rather than by index.

Cell **edges** come from `dx`, which is exactly the ψ width of a cell: rebuilt
edges reproduce `psixy` to 3e-12 relative, and `coarsen_x` verifies that on
every column before using it.

`uniform_theta=False` switches the poloidal coordinate from index to true
parallel arc length `∫dy·√g_22`, which matters when the two grids have
different poloidal packing rather than a uniform refinement.

What changed from the original scripts
--------------------------------------
* **The region → array-index map is derived, not hard-coded.** The originals
  carried a `y_beg`/`y_end`/`xN_beg`/`xO_beg` table written out by hand for a
  connected double null, with a magic `xO_beg['SOL'] = 2`. Here the
  destination slice is recovered by matching each region's own coordinate
  values back into the full array (`np.searchsorted`), which is topology-
  independent and cannot silently drift if the grid changes.
* **One `mode`.** `interpolatey_dmp_hm_yguards.py` had four consecutive
  `opt_ = ...` lines, so only the last ever took effect.
* **`import sys`** — the originals call `sys.exit()` on every error path
  without importing it, turning a clear message into a `NameError`.
* **`correct_MYG_squash` is fixed.** The copy in `interpolatex_dmp_hm.py`
  assigned `dsN1` then read `ds`, so it raised `NameError` whenever the MYG
  fix-up was actually needed.
* **Conservation check.** `mode='check_result'` compares volume-integrated
  Ne / Pe / Pi between source and refined dataset. Interpolation is not
  conservative, but a few-percent change is expected and a large one means
  something is wrong (usually a mismatched region or a bad ψ mapping).

Caveats
-------
* **`interp_x`/`interp_y` do not conserve** — expect small changes in total
  particles and energy, and a transient when the refined run starts. Cubic
  splines overshoot at steep gradients, so positive-definite variables are
  clamped with `abs()` (the originals did this too), which is a blunt fix.
  `coarsen_x` has neither problem: it conserves exactly and is bounded.
* `to_restart` needs `nxpe`/`nype` matching the new run's decomposition.
"""

import os
import sys
import fnmatch

import numpy as np
import netCDF4 as nc
from scipy.interpolate import CubicSpline

# ============================================================================
# CONFIG
# ============================================================================
src_dir = "/pitagora_work/FUPA2_SOLBOU10/hermes3/cd_MU/new_ny28/fresh_ES/data/113/"                       # folder holding the OLD squashed data
dst_dir = "./"    # folder holding the NEW-grid template

src_grid = "/pitagora_work/FUPA2_SOLBOU10/hermes3/cd_MU/new_ny28/fresh_ES/exp45470_x132_y28_src8e21_5eV.nc"      # None -> search upward from src_dir for a *.grd.nc / *.nc
dst_grid = "../exp45470_nx132_ny40_src8e21_5eV.nc"

src_data  = "boutdata.nc"          # read-only. GUARDLESS for interp_*/check when the
dst_data  = "boutdata.nc"          # grid has y_boundary_guards=0; guard squashes go in
                                   # src_guard_data/dst_guard_data (add_yguards), not here.
                                   # coarsen_x accepts either.

#src_dir = "../15/"                       # folder holding the OLD squashed data
#dst_dir = "../../../new_ny44/data/1/"    # folder holding the NEW-grid template
#
#src_grid = None      # None -> search upward from src_dir for a *.grd.nc / *.nc
#dst_grid = None
#
dst_write = "boutdata_write.nc"    # a COPY of dst_data; this is what gets written

# y-guards for grids built with y_boundary_guards=0 (mode='add_yguards').
# interp_* run on guardless squashes (the grid and dump must agree for xBOUT),
# but BOUT++ restarts carry MYG guard rows (8x8x27, not 8x4x27), so the
# interpolated interior is moved into a guard-retaining copy of the new run:
dst_guard_data  = "boutdata_guards.nc"        # NEW run, yguards='include_upper' (shape source, read-only)
dst_guard_write = "boutdata_guards_write.nc"  # a COPY of dst_guard_data; add_yguards fills it,
                                              # to_restart then reads it
src_guard_data  = "boutdata_guards.nc"        # OLD run, yguards='include_upper': guard values to
                                              # copy across. None -> zero-gradient from the interior
run_myg = 2              # MYG of the run being restarted (BOUT++ default 2). A
                         # guardless squash records MYG=0, so it cannot be read
                         # back from boutdata.nc.

mode = "interp_y"
#  'check'         : report topology + region index map, touch nothing
#  'interp_y'      : poloidal interpolation, region by region, field-aligned
#  'interp_x'      : radial interpolation in normalised psi (REFINING)
#  'coarsen_x'     : radial COARSENING by conservative averaging in psi
#  'add_yguards'   : interpolated interior + y-guard rows -> dst_guard_write
#                    ('sheath' is an alias)
#  'interp_z'      : TOROIDAL refinement (MZ), spectral, straight on the
#                    restart files — no grid, no template, no squash needed
#  'to_restart'    : write BOUT.restart.* from the filled template
#  'copy_restart'  : copy variables between two existing restart sets
#  'check_result'  : volume-integrated Ne/Pe/Pi, source vs refined

# ── mode='interp_z' ──────────────────────────────────────────────────────────
# A z refinement touches nothing else: no new grid (the Hypnotoad grid has no
# z), no template dump, no interp_x/interp_y. It reads the OLD restart files
# and writes new ones with the z axis resized, keeping the same x/y
# decomposition, so NXPE/NYPE and every other variable carry over unchanged.
mz_new    = 81                     # new MZ (e.g. 27 -> 81 at zperiod=5)
z_src_dir = "./old_restart/"       # folder of the existing BOUT.restart.*.nc
z_dst_dir = "./new_restart_mz81/"  # created; must differ from z_src_dir

variables = ["Ne", "Ni", "Ve", "Vi", "NVe", "NVi", "Te", "Ti",
             "Pe", "Pi", "Vort", "phi"]
# A NEUTRALS run names its species differently and the defaults above silently
# skip them (the loop ignores anything absent), so a coarsened restart would
# start with no neutrals at all. For the MAST-U nn_cd deck add:
#   ["Ne","NVe","Pe","Te","Ve", "Nd+","NVd+","Pd+","Td+","Vd+",
#    "Nd","NVd","Pd","Td","Vd", "Vort","phi"]
positive_vars = ("Ne", "Ni", "Te", "Ti", "Pe", "Pi")   # clamped with abs()

allow_file_fixup = False # let the script WRITE a corrected MYG into a squashed
                         # dump. Off by default: it modifies a file you think of
                         # as read-only, and if the guard inference is wrong it
                         # corrupts the metadata of good data. Left off, the
                         # script reports what it would change and carries the
                         # right value in memory instead.
dry_run = False           # interp_*: do everything EXCEPT the write, and report
                         # per-region shapes/ranges. Always run this first —
                         # the write modifies dst_write in place.
uniform_theta = True     # False -> use parallel arc length int dy*sqrt(g_22)
coarsen_method = "conservative"   # 'conservative' (default) | 'spline'
                         # 'spline' reuses the interp_x mapping, for comparison
                         # only: it aliases and overshoots. See the docstring.
tind = -1                # time index of the source to interpolate from

nxpe, nype = 16, 9      # decomposition for 'to_restart'
restart_vars = ["Ne", "NVe", "NVi", "Pe", "Pi", "Vort", "phi"]
copy_from, copy_to = "./1/", "./to_restart/"           # for 'copy_restart'
# ============================================================================


def find_file(pattern, path):
    """First file matching `pattern` at or below `path` (the originals' helper)."""
    for root, _dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None


def dump_has_yguards(datafile, grid_ny):
    """Does the squashed dump actually carry y-boundary guard cells?

    Ask the FILE, not the grid. `squashoutput(yguards="include_upper")` writes
    a dump with `ny + 4*MYG` poloidal cells even when the grid declares
    `y_boundary_guards = 0`, and it can stamp an `MYG` that disagrees with
    both. Trusting the grid's flag there makes xBOUT mis-slice every poloidal
    region — silently, because the shapes still broadcast. Comparing the
    dump's poloidal size against the grid's `ny` is unambiguous.
    """
    with nc.Dataset(datafile) as d:
        dim = "y" if "y" in d.dimensions else "theta"
        n = len(d.dimensions[dim])
    return n > grid_ny, n


def fix_squash_myg(datafile, MYG, allow_write=False):
    """Report — and optionally correct — a wrong `MYG` in a squashed dump.

    `boutdata.squashoutput` can stamp an `MYG` that disagrees with the grid,
    which makes xBOUT mis-slice the poloidal regions. The originals rewrote the
    file unconditionally; that is a write to something the user reasonably
    treats as read-only, and if the guard inference is wrong it corrupts good
    metadata. So this only reports unless `allow_write=True`.

    (Fixed version of the originals' `correct_MYG_squash` — the copy in
    `interpolatex_dmp_hm.py` opened the file as `dsN1` but read `ds`, so it
    raised `NameError` exactly when it was needed.)
    """
    try:
        with nc.Dataset(datafile, "r") as d:
            if "MYG" not in d.variables:
                return
            current = int(d["MYG"][:])
    except OSError as exc:
        print(f"    could not read MYG from {datafile}: {exc}")
        return
    if current == MYG:
        return
    if not allow_write:
        print(f"    NOTE {os.path.basename(datafile)} has MYG={current}, grid "
              f"implies {MYG}. Using {MYG} in memory; set "
              f"allow_file_fixup=True to write it into the file.")
        return
    try:
        with nc.Dataset(datafile, "r+") as d:
            d["MYG"][:] = MYG
        print(f"    corrected MYG {current} -> {MYG} in {datafile}")
    except OSError as exc:
        print(f"    could not fix MYG in {datafile}: {exc}")


def _is_vlen_str(v):
    """True for a netCDF VLEN/string variable (BOUT++ `run_id` etc.)."""
    try:
        if v.dtype is str:
            return True
    except Exception:
        pass
    try:
        return getattr(v.dtype, "kind", "") in ("U", "S", "O")
    except Exception:
        return False


def interpft_z(arr, nz_new, axis=-1):
    """Spectral interpolation along the periodic toroidal axis. EXACT.

    Zero-padding the rfft spectrum *is* band-limited interpolation for a
    periodic, uniformly spaced axis: every mode the source resolves is
    reproduced exactly and nothing is invented.  So unlike the radial
    direction, z needs no conservative-averaging path -- mode 0 is untouched,
    so the z-average (hence every volume integral) is preserved to round-off
    in BOTH directions, refining and coarsening.  Coarsening is then the exact
    low-pass rather than the aliasing mess that sampling a spline downwards is
    in x.

    Verified for `27 -> 81` (the MZ refinement this exists for): since
    `81 = 3*27` and both grids start at z=0, every 3rd fine point coincides
    with a source point, and the interpolant reproduces the source there to
    3.6e-15; the z-mean holds to 1.3e-15, the 27->81->27 round trip to 5.3e-15
    and the rms ratio is 1.000000.

    THREE BUGS in STORM's `interpft_dbZ` (`Storm/drsep0_scripts/copy_restart.py`,
    where the call site is also commented out, so it was never exercised) --
    all three bite on 27 -> 81:
      * numpy's `irfft` divides by its OUTPUT length, so the spectrum must be
        scaled by `nz_new/nz_old`.  The hard-coded `*2.0` is right only for
        doubling; 27 -> 81 needs 3.0, so every field came out at ~2/3.
      * `irfft` must be given `n=nz_new`.  Its default is `2*(m-1)`, which for
        an ODD target returns `nz_new - 1` points -- 80 instead of 81.
      * if `nz_old` is EVEN its last rfft bin is the Nyquist mode, counted once
        there but twice as an interior mode of the longer transform, so it must
        be halved when padding (and doubled when truncating onto a new even
        Nyquist).  A no-op for odd -> odd, wrong in general without it.
    """
    a = np.moveaxis(np.asarray(arr, dtype=float), axis, -1)
    nz_old = a.shape[-1]
    if nz_old == nz_new:
        return np.moveaxis(a.copy(), -1, axis)
    A = np.fft.rfft(a, axis=-1)
    m_old = A.shape[-1]
    m_new = nz_new // 2 + 1
    if nz_old % 2 == 0:                      # source Nyquist splits into +/-k
        A[..., -1] = A[..., -1] * 0.5
    if m_new >= m_old:                       # refine: zero-pad
        pad = list(A.shape[:-1]) + [m_new - m_old]
        A = np.concatenate([A, np.zeros(pad, dtype=A.dtype)], axis=-1)
    else:                                    # coarsen: exact low-pass
        A = A[..., :m_new].copy()
        if nz_new % 2 == 0:                  # recombine into the new Nyquist
            A[..., -1] = A[..., -1] * 2.0
    out = np.fft.irfft(A, n=nz_new, axis=-1) * (float(nz_new) / float(nz_old))
    return np.moveaxis(out, -1, axis)


def psi_cell_edges(gridfile):
    """Radial cell edges in psi, plus centres and the volume weight J*dx.

    `dx` is exactly the psi width of a cell, so the edges follow from a
    cumulative sum anchored half a cell below the first centre. That is
    checked here rather than assumed: the rebuilt centres must reproduce
    `psixy`, which they do to ~3e-12 relative on the SXD and CD grids. If a
    grid ever stores `dx` in some other normalisation this aborts instead of
    silently averaging over the wrong intervals.

    Returns (edges (nx+1, ny), centres (nx, ny), J (nx, ny)). The volume
    weight of a cell is `J * dx`, but `overlap_weights` needs `J` alone: the
    overlap length it computes already IS the (possibly partial) width, so
    passing `J*dx` would square the width and bias the average towards the
    wider cells. Caught by the integral check, which read -99.97 %.
    """
    with nc.Dataset(gridfile) as f:
        psi = np.asarray(f["psixy"][:], dtype=float)
        dxg = np.asarray(f["dx"][:], dtype=float)
        J = np.abs(np.asarray(f["J"][:], dtype=float))
    nx, ny = psi.shape
    edges = np.empty((nx + 1, ny))
    edges[0, :] = psi[0, :] - 0.5 * dxg[0, :]
    edges[1:, :] = edges[0, :] + np.cumsum(dxg, axis=0)
    centres = 0.5 * (edges[:-1, :] + edges[1:, :])
    span = np.ptp(psi, axis=0)
    err = np.max(np.abs(centres - psi) / np.where(span == 0, 1.0, span))
    if err > 1e-6:
        sys.exit(f"{gridfile}: cell edges from dx do not reproduce psixy "
                 f"(max rel error {err:.2e}). dx is not the psi cell width "
                 f"for this grid; conservative coarsening cannot be trusted.")
    return edges, centres, J


def strip_fill_values(savepath, prefix="BOUT.restart"):
    """Remove `_FillValue` from every variable in the written restart files.

    xarray attaches `_FillValue = NaN` to float variables by default, so
    `to_restart` writes restarts carrying it on ~148 variables. A restart
    BOUT++ wrote itself carries it on NONE. On restart BOUT++ reopens the file
    to write, and netcdf-cxx4 then refuses to define a fill value on a variable
    that already holds data:

        Error while writing value 'BOUT_VERSION' :
        NetCDF: Attempt to define fill value when data already exists.
        ... netcdf-cxx4-src/cxx4/ncVar.cpp line:322

    The attribute only ever described unwritten regions, and `to_restart`
    writes every element, so removing it changes no data — verified by
    comparing every variable before and after.
    """
    import glob
    n_files = n_attrs = 0
    for f in sorted(glob.glob(os.path.join(savepath, prefix + ".*.nc"))):
        with nc.Dataset(f, "a") as d:
            hit = False
            for v in d.variables.values():
                if "_FillValue" in v.ncattrs():
                    v.delncattr("_FillValue")
                    n_attrs += 1
                    hit = True
        n_files += hit
    return n_files, n_attrs


def grid_row_map(ny_grid, ny_dump, MYG, ny_inner):
    """Map each poloidal row of a guard-retaining dump to its grid row.

    `squashoutput(yguards='include_upper')` writes `ny + 4*MYG` poloidal rows
    for a double null (`ny + 2*MYG` for a single null) while a
    `y_boundary_guards=0` grid has only `ny`. `coarsen_x` needs the radial psi
    profile of each column, and a y-guard row sits on the SAME flux surfaces as
    the domain row beside it, so the guards simply reuse their neighbour's
    column — no interpolation, and no assumption about what the guards hold.

    The offsets are the ones used throughout this file: region A starts at
    +MYG, region B at +3*MYG. Returns None if the two sizes cannot be
    reconciled, so the caller aborts instead of mis-pairing rows.
    """
    if ny_dump == ny_grid:
        return np.arange(ny_grid)
    j = np.arange(ny_dump)
    if ny_inner and ny_dump == ny_grid + 4 * MYG:          # double null
        return np.where(j < ny_inner + 2 * MYG,
                        np.clip(j - MYG, 0, ny_inner - 1),
                        np.clip(j - 3 * MYG, ny_inner, ny_grid - 1))
    if ny_dump == ny_grid + 2 * MYG:                       # single null
        return np.clip(j - MYG, 0, ny_grid - 1)
    return None


def domain_row_for_grid_row(ny_grid, ny_dump, MYG, ny_inner):
    """Inverse of `grid_row_map`: the dump row holding each grid row's DATA.

    Guard rows duplicate their neighbour, so inverting the map by search would
    sometimes land on a guard. The forward formula is exact and unambiguous:
    region A sits at +MYG, region B at +3*MYG.
    """
    if ny_dump == ny_grid:
        return np.arange(ny_grid)
    g = np.arange(ny_grid)
    if ny_inner and ny_dump == ny_grid + 4 * MYG:          # double null
        return np.where(g < ny_inner, g + MYG, g + 3 * MYG)
    if ny_dump == ny_grid + 2 * MYG:                       # single null
        return g + MYG
    return None


def dump_poloidal_info(datafile):
    """(nx, poloidal size, MYG, ny_inner) as the DUMP itself reports them."""
    with nc.Dataset(datafile) as d:
        dim = "y" if "y" in d.dimensions else "theta"
        n_y = len(d.dimensions[dim])
        xdim = "x" if "x" in d.dimensions else "psi"
        n_x = len(d.dimensions[xdim])
        g = lambda k: (int(np.atleast_1d(d[k][:]).ravel()[0])
                       if k in d.variables else None)
        return n_x, n_y, (g("MYG") or 2), g("ny_inner")


def overlap_weights(src_edges, dst_edges, src_J):
    """Weight matrix W[k, i] for averaging source cells into dest cell k.

    `W[k, i] = overlap_in_psi(k, i) * J_i`, so summing it over i gives the
    source volume flowing into destination cell k.

    The overlap is measured in psi, so the two grids need not be nested and
    may pack cells quite differently — which they do: dpsi varies 15x across
    the SXD radial mesh, so an every-other-cell average by INDEX would weight
    the near-separatrix cells as heavily as the far-SOL ones.
    """
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    return np.clip(hi - lo, 0.0, None) * src_J[None, :]


def grid_topology(gridfile):
    """Topology numbers from a grid file, plus whether it carries y-guards."""
    g = nc.Dataset(gridfile)
    q = lambda k: int(g[k][:].item()) if k in g.variables else None
    t = dict(nx=len(g.dimensions["x"]), ny=len(g.dimensions["y"]),
             ixseps1=q("ixseps1"), ixseps2=q("ixseps2"), ny_inner=q("ny_inner"),
             jyseps1_1=q("jyseps1_1"), jyseps2_1=q("jyseps2_1"),
             jyseps1_2=q("jyseps1_2"), jyseps2_2=q("jyseps2_2"),
             MYG=q("y_boundary_guards") or 0)
    g.close()
    t["yguards"] = t["MYG"] == 2
    return t


def region_slices(ds, sub):
    """Index ranges of a `from_region` sub-array inside the full array.

    Replaces the hand-written per-region index table in the originals. Each
    region carries its own `x`/`theta` coordinate values, so matching them back
    into the full coordinate arrays recovers the slice for any topology.
    Raises if the result is not contiguous, which would mean the region
    selection and the full array disagree.
    """
    out = {}
    for dim in ("x", "theta"):
        full = np.asarray(ds[dim].values)
        vals = np.asarray(sub[dim].values)
        idx = np.searchsorted(full, vals)
        if idx.size > 1 and not np.all(np.diff(idx) == 1):
            raise RuntimeError(f"region is not contiguous in {dim}")
        out[dim] = slice(int(idx[0]), int(idx[-1]) + 1)
    return out["x"], out["theta"]


def write_band(region, xs, ixseps1):
    """Trim a region's x range to the band it actually owns.

    `from_region` returns MXG x-guard cells on the side facing the separatrix,
    so neighbouring regions OVERLAP: on the example case the core/PFR regions
    span x 0:68 and the SOL regions x 64:132, both 68 wide, while the
    separatrix is at 66. Writing both in full would let whichever is processed
    last overwrite two cells belonging to the other — silently corrupting two
    radial cells either side of the separatrix.

    Each destination cell must therefore be written exactly once: core/PFR own
    `x < ixseps1`, SOL owns `x >= ixseps1`. The originals encoded this as a
    hand-written table with a magic `xO_beg['SOL'] = 2`; deriving it from
    `ixseps1` reproduces that table exactly and survives a change of grid.

    Returns `(lo, hi, sub_lo, sub_hi)` — the destination range and the matching
    slice of the region sub-array.
    """
    if region.endswith("SOL"):
        lo, hi = max(xs.start, ixseps1), xs.stop
    else:                       # 'core' and 'PFR' regions are inside ixseps1
        lo, hi = xs.start, min(xs.stop, ixseps1)
    sub_lo = lo - xs.start
    return lo, hi, sub_lo, sub_lo + (hi - lo)


def n_left_guards(sub_guarded, sub_plain):
    """How many guard cells `with_guards=2` actually prepended to a region.

    **Not always 2.** A region that abuts a target gets guards only on the
    side facing the domain, so the count is 0 there; on a grid written with
    `y_boundary_guards=0` this is the common case, and it differs region by
    region. On the CDN example: 2 for `inner_core`/`outer_core` and the
    X-point-adjacent regions, but **0** for `lower_inner_*` and
    `upper_outer_*`, which sit against a target.

    Getting this wrong shifts the whole poloidal coordinate by half a region
    and silently ruins the interpolation in exactly those target-adjacent
    regions. Counting the cells that fall before the unguarded region's first
    cell is exact and covers 0, 1 or 2 uniformly.
    """
    tg = np.asarray(sub_guarded["theta"].values)
    t0 = float(np.asarray(sub_plain["theta"].values)[0])
    return int((tg < t0 * 0.999).sum())


def theta_coord(sub_guarded, sub_plain, n_no_guards, ds=None, region=None,
                uniform=True):
    """Poloidal coordinate for the spline, normalised so old and new align.

    `uniform=True` uses the cell index divided by the region's cell count — the
    right choice for a uniform refinement. `uniform=False` uses cumulative
    parallel arc length `∫dy·√g_22`, needed when the two grids pack poloidal
    cells differently.

    The origin is set at the first *interior* cell, so that the destination
    coordinate `arange(n_new)/n_new` lands inside the source range and the
    spline interpolates rather than extrapolates.
    """
    n_g = sub_guarded.sizes["theta"]
    n_l = n_left_guards(sub_guarded, sub_plain)
    if uniform:
        return (np.arange(n_g) - n_l) / float(n_no_guards)
    dy = (ds["dy"].bout.from_region(region, with_guards=2).values
          * np.sqrt(ds["g_22"].bout.from_region(region, with_guards=2).values))
    th = np.zeros(dy.shape)
    for i in range(dy.shape[0]):
        th[i, :] = np.insert(np.cumsum((dy[i, :-1] + dy[i, 1:]) / 2), 0, 0)
        if n_l:
            th[i, :] -= th[i, n_l]
    return th


# ---------------------------------------------------------------------------
# mode='interp_z' resolves NO grid and runs NO preflight: a toroidal
# refinement has no counterpart in the Hypnotoad grid (z exists only as MZ in
# BOUT.inp), so the grid lookup below — which sys.exit()s when it cannot find
# a pair — would block a mode that never reads one.
_needs_grid = mode != "interp_z"
if _needs_grid:
    src_grid = src_grid or find_file("*.nc", os.path.join(src_dir, ".."))
    dst_grid = dst_grid or find_file("*.nc", os.path.join(dst_dir, "..", ".."))
    if src_grid is None or dst_grid is None:
        sys.exit(f"could not find grids (src={src_grid}, dst={dst_grid}) — set them in CONFIG")

def preflight(t_src, t_dst, src_file, dst_file, write_file, mode):
    """Say what this refinement needs before doing any work.

    Reactive errors ("file missing") arrive after you have already waited for a
    multi-GB load. This states the plan, the prerequisites and the next step
    up front.
    """
    notes, blockers = [], []
    dx = t_src["nx"] != t_dst["nx"]
    dy = t_src["ny"] != t_dst["ny"]

    coarser = t_dst["nx"] < t_src["nx"]
    xmode = "coarsen_x" if coarser else "interp_x"

    print("\n" + "-" * 72)
    print("PREFLIGHT")
    print(f"  radial     : nx {t_src['nx']} -> {t_dst['nx']}"
          f"{f'  ({xmode} needed)' if dx else '  (unchanged)'}")
    print(f"  poloidal   : ny {t_src['ny']} -> {t_dst['ny']}"
          f"{'  (interp_y needed)' if dy else '  (unchanged)'}")

    if dx and dy:
        blockers.append(
            f"BOTH nx and ny change. `{xmode}` requires the two grids to share\n"
            "     ny (it maps psi column by column), so this cannot be done in\n"
            "     one hop. Build an INTERMEDIATE grid with the new nx and the OLD\n"
            f"     ny ({t_dst['nx']} x {t_src['ny']}), run {xmode} onto it, then\n"
            "     interp_y from there onto the final grid.")
    if dx and coarser:
        notes.append(
            f"nx is DECREASING — use mode='coarsen_x', not 'interp_x'.\n"
            "     Sampling a spline downwards aliases the sub-grid scales back\n"
            "     into the resolved band and can overshoot the data range;\n"
            "     coarsen_x averages over cell overlaps in psi instead.\n"
            f"     Halve in steps ({t_src['nx']} -> ... -> {t_dst['nx']}) if the\n"
            "     ratio is large; each hop is one run of this script.")
    if not dx and not dy:
        notes.append("grids have the same shape — nothing to interpolate.")

    if t_src["ixseps1"] != t_dst["ixseps1"] and not dx:
        notes.append(f"ixseps1 differs ({t_src['ixseps1']} vs {t_dst['ixseps1']}) "
                     f"with nx unchanged — check the grids are the same case.")

    # y-guards: what is present, and what the sheath step will need
    for tag, f, t in (("source", src_file, t_src), ("target", dst_file, t_dst)):
        if os.path.exists(f):
            has, n = dump_has_yguards(f, t["ny"])
            print(f"  {tag:10s} : {os.path.basename(f)}  {n} poloidal cells, "
                  f"y-guards {'present' if has else 'ABSENT'}")
            # A dump belonging to this grid has either ny cells (no guards) or
            # ny + 4*MYG (guards, 4 boundaries in a double null). Anything else
            # means the dump and the grid are not a pair — the commonest cause
            # of a silently wrong refinement, so stop rather than guess.
            if n not in (t["ny"], t["ny"] + 8, t["ny"] + 4):
                blockers.append(
                    f"{tag} dump has {n} poloidal cells, but this grid has "
                    f"ny={t['ny']}\n     (so {t['ny']} without guards or "
                    f"{t['ny'] + 8} with). The dump and grid are not a pair —\n"
                    f"     check {os.path.basename(f)} against "
                    f"{os.path.basename(src_grid if tag == 'source' else dst_grid)}.")
        else:
            blockers.append(f"{tag} dump missing: {f}\n     squash it first "
                            f"(boutdata.squashoutput, or squash_hermes.py).")

    if t_dst["MYG"] == 0 and run_myg > 0:
        notes.append(
            f"grid has y_boundary_guards=0 but the run uses MYG={run_myg}, so the\n"
            "     restart needs y-guard rows the guardless interpolation lacks:\n"
            "       1. interp_* on guardless squashes (src_data = dst_data =\n"
            "          'boutdata.nc', dst_write = 'boutdata_write.nc')\n"
            "       2. mode='add_yguards': interior + guard rows -> "
            f"{dst_guard_write}\n"
            f"          needs <dst>/{dst_guard_data}, a guard-RETAINING squash of\n"
            "          the NEW run. squash_hermes.py will NOT make it: it reads\n"
            "          y_boundary_guards from the grid, finds 0, and passes\n"
            "          yguards=False. In the folder holding this run's\n"
            "          BOUT.dmp.*.nc (often <dst>/1/), run instead:\n"
            "            python -c \"from boutdata.squashoutput import "
            "squashoutput; \\\n"
            "              squashoutput(datadir='./', outputname="
            f"'{dst_guard_data}',\\\n"
            "                           yguards='include_upper')\"\n"
            f"          then move it beside {dst_data} and\n"
            f"            cp {dst_guard_data} {dst_guard_write}\n"
            "          (squashoutput keeps delete=False, so the dmp files stay.)\n"
            "       3. mode='to_restart'")

    # the write template must exist and must not be the file being read
    if mode in ("interp_x", "interp_y", "to_restart", "check_result"):
        if not os.path.exists(write_file):
            blockers.append(
                f"write template missing: {write_file}\n"
                f"     cp {dst_file} {write_file}\n"
                f"     (the template is written in place; keeping it separate\n"
                f"      means a half-finished run cannot corrupt your source.)")
        elif os.path.exists(dst_file) and os.path.samefile(dst_file, write_file):
            blockers.append(
                f"{write_file} IS {dst_file} (same file). Make a real copy — "
                f"writing\n     into the file being read corrupts it mid-run.")

    for n in notes:
        print(f"  NOTE  {n}")
    for b in blockers:
        print(f"  STOP  {b}")
    print("-" * 72)
    if blockers and mode != "check":
        sys.exit("preflight found blockers (above). Fix them, or run "
                 "mode='check' to inspect without doing work.")


if _needs_grid:
    t_src, t_dst = grid_topology(src_grid), grid_topology(dst_grid)
    print(f"source grid : {src_grid}\n   {t_src}")
    print(f"target grid : {dst_grid}\n   {t_dst}")

    src_file = os.path.join(src_dir, src_data)
    dst_file = os.path.join(dst_dir, dst_data)
    write_file = os.path.join(dst_dir, dst_write)

    preflight(t_src, t_dst, src_file, dst_file, write_file, mode)
else:
    print(f"mode='{mode}': no grid and no preflight — a toroidal refinement "
          "needs neither.")

if mode in ("interp_y", "interp_x", "check", "check_result"):
    import xhermes

    for f in (src_file, dst_file):
        if not os.path.exists(f):
            sys.exit(f"missing {f} — squash it first")

    # Reconcile the grid's guard flag against what the dumps actually contain.
    # A mismatch is the classic failure with grids written with
    # y_boundary_guards=0: the dump has guards, the grid says it does not, and
    # every poloidal region silently comes out shifted.
    for tag, f, t in (("src", src_file, t_src), ("dst", dst_file, t_dst)):
        has, n = dump_has_yguards(f, t["ny"])
        if has != t["yguards"]:
            print(f"  {tag}: dump has {n} poloidal cells vs grid ny={t['ny']} "
                  f"-> guards {'PRESENT' if has else 'ABSENT'}, but the grid "
                  f"says y_boundary_guards={t['MYG']}.")
            print(f"      trusting the dump: keep_yboundaries={has}")
            t["yguards"] = has
            t["MYG"] = 2 if has else 0
    fix_squash_myg(src_file, t_src["MYG"], allow_file_fixup)
    fix_squash_myg(dst_file, t_dst["MYG"], allow_file_fixup)

    # xBOUT merges the grid into the dump under geometry="toroidal", so a
    # guardless grid and a guard-retaining dump collide on the y dimension.
    # The bare xarray error names two variables and no remedy
    # ("conflicting sizes for dimension 'y': length 28 on 'psixy' and length
    # 36 on ... 'Bxy'"), so say what it means and who does not need it.
    for tag, f, g, t in (("source", src_file, src_grid, t_src),
                         ("target", dst_file, dst_grid, t_dst)):
        with nc.Dataset(g) as _g:
            ny_grid = len(_g.dimensions["y"])
        _, ny_dump, _, _ = dump_poloidal_info(f)
        if ny_grid != ny_dump:
            sys.exit(
                f"{tag}: {os.path.basename(f)} has {ny_dump} poloidal rows (y-guards) "
                f"but grid {os.path.basename(g)} has ny={ny_grid} (none).\n"
                f"  FIX: interpolate the GUARDLESS squashes, the guard files come later:\n"
                f"         src_data = dst_data = 'boutdata.nc'\n"
                f"         dst_write = 'boutdata_write.nc'   (fresh cp of dst boutdata.nc)\n"
                f"       then mode='add_yguards' (uses src_guard_data/dst_guard_data),\n"
                f"       then mode='to_restart'.\n"
                f"  Why: geometry=\"toroidal\" merges grid and dump, so their y sizes\n"
                f"  must agree. (coarsen_x reads with netCDF4 and has no such limit.)")

    ds1 = xhermes.open_hermesdataset(
        src_file, geometry="toroidal", gridfilepath=src_grid,
        keep_xboundaries=True, keep_yboundaries=t_src["yguards"],
        info=False, unnormalise=False)
    ds2 = xhermes.open_hermesdataset(
        dst_file, geometry="toroidal", gridfilepath=dst_grid,
        keep_xboundaries=True, keep_yboundaries=t_dst["yguards"],
        info=False, unnormalise=False)
    print(f"\nloaded  src {dict(ds1.sizes)}\n        dst {dict(ds2.sizes)}")
    # The "plain" region must carry NO y halo. from_region(region) with no
    # with_guards adds MYG halo rows from the CONNECTED region, and MYG comes
    # from the file — which a guardless squash may still stamp as 2 (CD
    # fresh_ES/113 does; the SXD squash wrote 0, which is why SXD was fine).
    # The core's halo is the far end of the core ring, so writing it put core
    # values onto the PFR rows beside the X-point: -19 % Ne inventory, inner
    # core halved, lower-inner/upper-outer PFR filled with core plasma. The
    # SOL survived only because its halo is the same field line. x guards are
    # unchanged (MXG), as before.
    y_plain = {ds2.metadata["bout_ydim"]: 0}


if mode == "check":
    print(f"\nregions ({len(ds1.regions)}):")
    print("  {:20s}{:>16s}{:>16s}{:>7s}{:>18s}".format(
        "region", "src (x, theta)", "dst (x, theta)", "gL/gR", "dst write band"))
    for region in ds1.regions.keys():
        s1 = ds1["Ne"].bout.from_region(region, with_guards=y_plain)
        s2 = ds2["Ne"].bout.from_region(region, with_guards=y_plain)
        xs, ys = region_slices(ds2, s2)
        lo, hi, a, b = write_band(region, xs, t_dst["ixseps1"])
        g1 = ds1["Ne"].bout.from_region(region, with_guards=2)
        nl = n_left_guards(g1, s1)
        print("  {:20s}{:>16s}{:>16s}{:>7s}{:>18s}".format(
            region, f"{s1.sizes['x']}, {s1.sizes['theta']}",
            f"{s2.sizes['x']}, {s2.sizes['theta']}",
            f"{nl}/{g1.sizes['theta'] - s1.sizes['theta'] - nl}",
            f"x{lo}:{hi} y{ys.start}:{ys.stop}"))
    missing = [v for v in variables if v not in ds1.variables]
    print(f"\nvariables present: {[v for v in variables if v in ds1.variables]}")
    if missing:
        print(f"variables absent (skipped): {missing}")
    print(f"\nwould write into: {write_file}"
          f"  {'(exists)' if os.path.exists(write_file) else '(MISSING)'}")
    print("\nNEXT: set mode='interp_y' (dry_run=True first), then dry_run=False,"
          "\n      then mode='add_yguards' if the grid has no y-guards,"
          "\n      then mode='check_result', then mode='to_restart'.")


elif mode == "interp_y":
    # Prove the destination tiling BEFORE writing: every cell exactly once.
    # The same check after the loop only warned, by which time a bad region map
    # had already overwritten the template (CD fresh_ES -> ny40: -19 % Ne).
    _ref = next(v for v in variables if v in ds2.variables)
    _hits = np.zeros((ds2.sizes["x"], ds2.sizes["theta"]), dtype=int)
    for region in ds1.regions.keys():
        _s2 = ds2[_ref].isel(t=tind).bout.from_region(region, with_guards=y_plain)
        _xs, _ys = region_slices(ds2, _s2)
        _lo, _hi, _a, _b = write_band(region, _xs, t_dst["ixseps1"])
        _hits[_lo:_hi, _ys] += 1
    if (_hits != 1).any():
        sys.exit(f"interp_y: the region map does not tile the destination — "
                 f"{int((_hits == 0).sum())} cells never written, "
                 f"{int((_hits > 1).sum())} written more than once.\n"
                 f"     Nothing was written. Run mode='check' for the region map.")
    print(f"\ntiling OK: all {_hits.size} destination cells written exactly once")
    if dry_run:
        out = None
        print("\nDRY RUN — nothing will be written")
    else:
        if not os.path.exists(write_file):
            sys.exit(f"{write_file} missing — copy {dst_data} to it first, so "
                     f"the template is never the file being read")
        fix_squash_myg(write_file, t_dst["MYG"], allow_file_fixup)
        out = nc.Dataset(write_file, "r+")
        print(f"\nwriting into {write_file}")
    covered = {}

    for var_name in variables:
        if var_name not in ds1.variables:
            continue
        print(f"  {var_name}")
        for region in ds1.regions.keys():
            sub1 = ds1[var_name].isel(t=tind).bout.from_region(region, with_guards=2)
            sub1_plain = ds1[var_name].isel(t=tind).bout.from_region(region, with_guards=y_plain)
            n_no_g = sub1_plain.sizes["theta"]
            var = np.asarray(sub1.bout.to_field_aligned().values)

            th1 = theta_coord(sub1, sub1_plain, n_no_g, ds1, region,
                              uniform_theta)
            sub2 = ds2[var_name].isel(t=tind).bout.from_region(region, with_guards=y_plain)
            n_new = sub2.sizes["theta"]
            if uniform_theta:
                th2 = np.arange(n_new) / float(n_new)
            else:
                dy2 = (ds2["dy"].bout.from_region(region, with_guards=y_plain).values
                       * np.sqrt(ds2["g_22"].bout.from_region(region, with_guards=y_plain).values))
                th2 = np.zeros(dy2.shape)
                for i in range(dy2.shape[0]):
                    th2[i, :] = np.insert(
                        np.cumsum((dy2[i, :-1] + dy2[i, 1:]) / 2), 0, 0)

            nx = var.shape[0]
            new = np.empty((nx, n_new, var.shape[2]))
            for ix in range(nx):
                if uniform_theta:
                    cs = CubicSpline(th1, var[ix], axis=0, extrapolate=True)
                    new[ix] = cs(th2)
                else:
                    cs = CubicSpline(th1[ix], var[ix], axis=0, extrapolate=True)
                    new[ix] = cs(th2[ix])
            if var_name in positive_vars:
                new = np.abs(new)

            # back out of field-aligned coordinates before writing
            aligned = sub2.bout.to_field_aligned()
            aligned[:, :, :] = new
            result = aligned.bout.from_field_aligned().values

            xs, ys = region_slices(ds2, sub2)
            lo, hi, a, b = write_band(region, xs, t_dst["ixseps1"])
            block = result[a:b]
            if not np.all(np.isfinite(block)):
                raise RuntimeError(f"{var_name}/{region}: non-finite after "
                                   f"interpolation")
            if out is not None:
                out[var_name][tind, lo:hi, ys, :] = block
            # Compare like with like: the SOURCE region without guards, trimmed
            # to the same radial band, against the block actually written.
            # Using the guarded sub-array here would fold the guard cells and a
            # two-cell-wider x range into the "change".
            src_plain = ds1[var_name].isel(t=tind).bout.from_region(region, with_guards=y_plain)
            sxs, _sys = region_slices(ds1, src_plain)
            slo, shi, sa, sb = write_band(region, sxs, t_src["ixseps1"])
            covered.setdefault(var_name, []).append(
                (region, (lo, hi), (ys.start, ys.stop), block.shape,
                 float(np.asarray(src_plain.values)[sa:sb].mean()),
                 float(block.mean())))

    if out is not None:
        out.close()

    # Every destination cell must be written exactly once. Check it rather than
    # trusting the region map — a gap leaves stale template values in the
    # refined run, an overlap silently discards one region's result.
    ref = next(iter(covered))
    nx_t, ny_t = ds2.sizes["x"], ds2.sizes["theta"]
    hits = np.zeros((nx_t, ny_t), dtype=int)
    for _r, (lo, hi), (y0, y1), _sh, _a, _b in covered[ref]:
        hits[lo:hi, y0:y1] += 1
    print(f"\ncoverage of the {nx_t}x{ny_t} destination plane:")
    print(f"  written once {int((hits == 1).sum())},  never {int((hits == 0).sum())},"
          f"  more than once {int((hits > 1).sum())}")
    if (hits != 1).any():
        print("  WARNING not a clean tiling — check the region map with mode='check'")

    print("\nper-region means (source -> interpolated), first variable."
          "\n  Low-density PFR regions change most: they sit orders of magnitude"
          "\n  below the core and have the steepest near-target gradients, where"
          "\n  a cubic spline is least reliable. Large changes there are expected;"
          "\n  large changes in the core are not.")
    print("  {:20s}{:>16s}{:>14s}{:>14s}{:>9s}".format(
        "region", "shape", "src mean", "new mean", "change"))
    for r, _lo, _y, sh, m1, m2 in covered[ref]:
        chg = 100 * (m2 / m1 - 1) if m1 else float("nan")
        print(f"  {r:20s}{str(sh):>16s}{m1:14.4e}{m2:14.4e}{chg:8.1f}%")
    print(f"\ninterp_y {'dry run' if out is None else 'done'}"
          f"  ({len(covered)} variables x {len(covered[ref])} regions)")
    print("NEXT: " + ("set dry_run=False to actually write."
                      if out is None else
                      "mode='check_result', then mode='add_yguards' if the "
                      "grid has no y-guards, then "
                      "mode='to_restart'."))


elif mode == "interp_x":
    if not os.path.exists(write_file):
        sys.exit(f"{write_file} missing — copy {dst_data} to it first")

    def psi_raw(gridfile):
        """RAW psixy [Wb] — deliberately NOT normalised.

        Each grid normalises by its OWN psi_axis/psi_bdry, so psi_n is not a
        common coordinate between two grids unless those happen to agree --
        and in general they do not, for the SAME equilibrium: the exp45470 132
        and 68 grids give psi_n spans 0.8528..1.1277 vs 0.8553..1.1256 (~1 % of
        the radial domain: ~1 cell on 132, several near the separatrix where the
        packing is 15x finer) while their RAW spans are identical. The SXD
        132/244 pair differs only at 2e-7. Being pair-dependent is exactly why
        the raw coordinate is the one to use. `coarsen_x` already worked in raw
        psi; `interp_x` did not until 2026-10-13.
        """
        with nc.Dataset(gridfile) as f:
            return (np.asarray(f["psixy"][:]),
                    float(f["psi_axis"][:]), float(f["psi_bdry"][:]))

    psi1, _a0, _a1 = psi_raw(src_grid)
    psi2, _b0, _b1 = psi_raw(dst_grid)
    if (_a0, _a1) != (_b0, _b1):
        print(f"  NOTE: the grids normalise psi differently (axis "
              f"{_a0:.8f}/{_b0:.8f}, bdry {_a1:.8f}/{_b1:.8f}); interpolating "
              "in RAW psi, which is unaffected by that.")
    if psi1.shape[1] != psi2.shape[1]:
        sys.exit(f"different ny ({psi1.shape[1]} vs {psi2.shape[1]}) — "
                 f"interp_x only refines the radial direction")
    # Raw and normalised psi differ by a positive affine factor, so they share
    # the monotonic direction, but do not assume increasing: CubicSpline needs
    # ascending x.
    _flip = psi1[0, 0] > psi1[-1, 0]
    if _flip:
        print("  psi decreases with x — reversing for the spline")

    src = nc.Dataset(src_file, "r")
    out = None if dry_run else nc.Dataset(write_file, "r+")
    # Target dimensions come from the write template. In a dry run `out` is None,
    # so open the template read-only for the shape (reusing `out` when writing).
    shp = out if out is not None else nc.Dataset(write_file, "r")
    print("\nDRY RUN — nothing will be written" if dry_run
          else f"\nwriting into {write_file}")
    for var_name in variables:
        if var_name not in src.variables or var_name not in shp.variables:
            continue
        print(f"  {var_name}")
        var = np.asarray(src[var_name][tind, :, :, :])
        _, nx_new, ny_new, nz = shp[var_name].shape
        new = np.empty((nx_new, ny_new, nz))
        n_clip = 0
        for jy in range(ny_new):
            p1, col = psi1[:, jy], var[:, jy, :]
            if _flip:
                p1, col = p1[::-1], col[::-1]
            # CLAMP rather than extrapolate. A cubic continued past its data
            # diverges, and `np.abs()` below then turns a large negative into a
            # large POSITIVE — fabricating a big value instead of a small one.
            # Clipping the target into the source range is exactly the constant
            # extrapolation the MATLAB original does
            # (STORM/manual/additional files/interpolatex_restart.m, which
            # finds psi2 > psi1[end] / psi2 < psi1[0] and pins the edge value).
            tgt = psi2[:, jy]
            cl = np.clip(tgt, p1[0], p1[-1])
            n_clip += int(np.count_nonzero(cl != tgt))
            cs = CubicSpline(p1, col, axis=0, extrapolate=False)
            new[:, jy, :] = cs(cl)
        if n_clip:
            print(f"    {n_clip} of {nx_new * ny_new} target cells lay OUTSIDE "
                  "the source psi range and were pinned to the edge value "
                  "(constant extrapolation)")
        if var_name in positive_vars:
            new = np.abs(new)
        if not np.all(np.isfinite(new)):
            raise RuntimeError(f"{var_name}: non-finite after interpolation")
        print(f"      src mean {var.mean():.4e} -> new mean {new.mean():.4e}"
              f"  ({100*(new.mean()/var.mean()-1):+.1f}%)")
        if out is not None:
            out[var_name][tind, :, :, :] = new
    src.close()
    if out is not None:
        out.close()
    else:
        shp.close()
    print(f"interp_x {'dry run' if dry_run else 'done'}")
    print("NEXT: " + ("set dry_run=False to actually write."
                      if dry_run else "mode='check_result', then mode='to_restart'."))


elif mode == "coarsen_x":
    # Radial COARSENING: 256 -> 128 -> 64, one hop per run. The opposite
    # direction from interp_x, and deliberately not the same algorithm — see
    # the module docstring for why sampling a spline downwards aliases.
    if not os.path.exists(write_file):
        sys.exit(f"{write_file} missing — copy {dst_data} to it first")
    if coarsen_method not in ("conservative", "spline"):
        sys.exit(f"coarsen_method={coarsen_method!r} — "
                 f"expected 'conservative' or 'spline'")

    se, sc, sJ = psi_cell_edges(src_grid)
    de, dc, _ = psi_cell_edges(dst_grid)
    sw = sJ * (se[1:, :] - se[:-1, :])          # source cell volume weight
    if se.shape[1] != de.shape[1]:
        sys.exit(f"different ny ({se.shape[1]} vs {de.shape[1]}) — coarsen_x "
                 f"maps psi column by column, so ny must match. Coarsen nx "
                 f"first, then run interp_y onto the final grid.")
    nx_src, nx_dst, ny = sc.shape[0], dc.shape[0], sc.shape[1]

    # The weights come from the GRID but the data is read from the DUMP, so
    # their poloidal layouts must agree. They do not when the dump was squashed
    # with `yguards='include_upper'` against a `y_boundary_guards=0` grid — the
    # commonest pairing here — and the loop below would then quietly pair
    # destination row jy with the WRONG source row, printing a plausible dry-run
    # table for it. Rather than demanding a re-squash, expand the grid's psi
    # columns onto the dump's rows: a guard row lies on the same flux surfaces
    # as its neighbour, so it reuses that column.
    for tag, f, n_x in (("source", src_file, nx_src),
                        ("target", write_file, nx_dst)):
        if not os.path.exists(f):
            continue
        n_xd, n_yd, _myg, _nyi = dump_poloidal_info(f)
        if n_xd != n_x:
            sys.exit(f"{tag} dump {os.path.basename(f)} has nx={n_xd} but its "
                     f"grid has nx={n_x} — not a matching pair.")

    # Weights stay in GRID-row space and the two dumps are reached through
    # maps, so the source and target need not be squashed the same way. That
    # matters: the usual pairing here is a plainly-squashed old run (ny rows)
    # filling a template squashed with yguards='include_upper' (ny+4*MYG).
    _, ny_src_dump, myg_s, nyi_s = dump_poloidal_info(src_file)
    dom_src = domain_row_for_grid_row(ny, ny_src_dump, myg_s, nyi_s)
    if dom_src is None:
        sys.exit(f"source dump has {ny_src_dump} poloidal rows, grid has {ny} "
                 f"(MYG={myg_s}, ny_inner={nyi_s}) — cannot reconcile.")
    _, ny_dst_dump, myg_d, nyi_d = dump_poloidal_info(write_file)
    row_dst = grid_row_map(ny, ny_dst_dump, myg_d, nyi_d)
    if row_dst is None:
        sys.exit(f"target dump has {ny_dst_dump} poloidal rows, grid has "
                 f"{ny} (MYG={myg_d}, ny_inner={nyi_d}) — cannot reconcile.")
    if ny_src_dump != ny or ny_dst_dump != ny:
        print(f"  poloidal rows: grid {ny}, source dump {ny_src_dump}, "
              f"target dump {ny_dst_dump}")
    dom_dst = domain_row_for_grid_row(ny, ny_dst_dump, myg_d, nyi_d)
    # When both dumps share a layout, pair rows DIRECTLY. Routing through the
    # grid rows instead would refill the target's guards from the neighbouring
    # domain row and throw away the guard values the source actually holds —
    # measured on the nn_cd pair as Ve max 2.3152 -> 3.6e-4, i.e. the sheath
    # velocities deleted. Note the integral check does not see this (it sums
    # over grid rows only); the mean/max columns are what expose it.
    src_row_for_dst = (np.arange(ny_dst_dump) if ny_src_dump == ny_dst_dump
                       else dom_src[row_dst])
    if ny_dst_dump > ny:
        print(f"  target has {ny_dst_dump - ny} y-guard rows; each is filled "
              f"from its adjacent domain row (zero-gradient, as add_yguards does without an old-run squash)")

    if nx_dst > nx_src:
        print(f"  NOTE: nx {nx_src} -> {nx_dst} is a REFINEMENT; averaging "
              f"would just smear it. Use mode='interp_x' instead.")
    print(f"\n  nx {nx_src} -> {nx_dst}  ({nx_src / nx_dst:.2f}x coarser), "
          f"ny {ny}, method '{coarsen_method}'")

    # Per-column weight matrices, built once (in GRID rows) and reused.
    W = np.empty((ny, nx_dst, nx_src))
    for jy in range(ny):
        W[jy] = overlap_weights(se[:, jy], de[:, jy], sJ[:, jy])
    tot = W.sum(axis=2)                                  # (ny, nx_dst)
    uncovered = tot <= 0.0
    if uncovered.any():
        # A destination cell outside the source psi range — normally only the
        # outermost guard column when the new grid extends slightly further.
        # Fall back to the nearest source cell rather than dividing by zero.
        n_bad = int(uncovered.sum())
        print(f"  {n_bad} of {ny * nx_dst} destination cells lie outside the "
              f"source psi range; filling those from the nearest source cell")
        for jy, kx in zip(*np.nonzero(uncovered)):
            W[jy, kx, np.argmin(np.abs(sc[:, jy] - dc[kx, jy]))] = 1.0
        tot = W.sum(axis=2)
    # `tot[k]` is the total SOURCE volume flowing into destination cell k, so
    # `sum_k f_k tot_k` is the source integral exactly when the destination
    # covers the source range. Keep it to check conservation below.
    src_volume_in = tot.T.copy()                         # (nx_dst, ny)
    W /= tot[:, :, None]

    # How much of the SOURCE the destination psi range actually covers. The
    # per-variable `integral` column below compares against the volume that was
    # mapped IN, so it reads 0.000 % even if the destination grid stops short —
    # this is the number that catches that, and a value below ~100 % means the
    # two grids do not span the same plasma. Compare RAW psi, not psi_n: each
    # grid normalises by its own psi_axis/psi_bdry, so the exp45470 132 and 68
    # grids look like they differ (psi_n 0.8528..1.1277 vs 0.8553..1.1256) while
    # their raw spans are identical and coverage is exactly 100 %.
    covered = tot.T.sum() / sw.sum()
    print(f"  destination covers {100 * covered:.3f} % of the source volume"
          + ("" if abs(covered - 1) < 2e-3 else
             "   <-- the two grids span different psi ranges"))

    src = nc.Dataset(src_file, "r")
    out = None if dry_run else nc.Dataset(write_file, "r+")
    print("\nDRY RUN — nothing will be written" if dry_run
          else f"\nwriting into {write_file}")
    print(f"\n  {'variable':10s}{'src mean':>13s}{'new mean':>13s}"
          f"{'mean':>9s}{'integral':>10s}{'src max':>13s}{'new max':>13s}")
    missing_src, missing_dst = [], []
    todo = []
    with nc.Dataset(write_file) as _t:
        for var_name in variables:
            if var_name not in src.variables:
                missing_src.append(var_name)
            elif var_name not in _t.variables:
                missing_dst.append(var_name)
            else:
                todo.append(var_name)
    # Check BOTH files up front. Skipping only on the source let the loop reach
    # the write and die with a bare `IndexError: Ni not found in /` partway
    # through — after some variables were already written, so the template was
    # left half-filled and looked usable.
    if missing_src:
        print(f"  not in the source dump, skipped: {', '.join(missing_src)}")
    if missing_dst:
        print(f"  not in the TARGET template, skipped: {', '.join(missing_dst)}"
              f"\n     (the two runs evolve different species — check that is "
              f"what you intend)")
    if not todo:
        sys.exit("none of `variables` is present in both files — for a "
                 "neutrals run use the Nd/Nd+ names, not Ni/Ti/Pi.")

    for var_name in todo:
        var = np.asarray(src[var_name][tind, :, :, :])   # (x, y_src, z)
        nz = var.shape[2]
        new = np.empty((nx_dst, ny_dst_dump, nz))
        # Destination row k <- grid row g <- source DOMAIN row dom_src[g].
        for k in range(ny_dst_dump):
            g = row_dst[k]
            col = var[:, src_row_for_dst[k], :]
            if coarsen_method == "conservative":
                new[:, k, :] = W[g] @ col
            else:
                new[:, k, :] = CubicSpline(sc[:, g], col, axis=0,
                                           extrapolate=True)(dc[:, g])
        if coarsen_method == "spline":
            if var_name in positive_vars:
                # Only the spline path can undershoot; a weighted average is
                # bounded by its inputs, so it never needs clamping.
                new = np.abs(new)
        if not np.all(np.isfinite(new)):
            raise RuntimeError(f"{var_name}: non-finite after coarsening")

        # Volume integral along each column, source vs coarsened. This is the
        # number to watch: the mean can move simply because the cell
        # distribution changed, the integral should not.
        # Compare over GRID rows — the physical domain — so duplicated guard
        # rows are not double-counted on either side.
        i_src = float((var[:, dom_src, :] * sw[:, :, None]).sum())
        i_new = float((new[:, dom_dst, :] * src_volume_in[:, :, None]).sum())
        d_mean = 100 * (new.mean() / var.mean() - 1) if var.mean() else np.nan
        d_int = 100 * (i_new / i_src - 1) if i_src else np.nan
        print(f"  {var_name:10s}{var.mean():13.4e}{new.mean():13.4e}"
              f"{d_mean:8.2f}%{d_int:9.3f}%{var.max():13.4e}{new.max():13.4e}")
        if out is not None:
            out[var_name][tind, :, :, :] = new
    src.close()
    if out is not None:
        out.close()
    print(f"\ncoarsen_x {'dry run' if dry_run else 'done'}")
    if dry_run:
        print("NEXT: set dry_run=False to actually write.")
    else:
        # `check_result` needs xBOUT and so cannot take a guardless grid beside
        # a guard-retaining dump; the integral column above already reports the
        # same quantity, exactly, so do not send people to a mode that will
        # refuse to run for them.
        print("NEXT: mode='to_restart'.")
        print("      The `integral` column above IS the conservation check "
              "(check_result adds nothing here,\n"
              "      and cannot run when the grid and dump disagree about "
              "y-guards).")


elif mode in ("add_yguards", "sheath"):
    # A grid with y_boundary_guards=0 forces interp_* onto guardless squashes
    # (xBOUT merges grid and dump, so their y sizes must agree), but the run
    # itself has MYG guard rows and BOUT++ refuses a restart without them:
    #   "Size mismatch for option Ne: Tensor (8, 4, 27) cannot be converted
    #    to Field3D (8, 8, 27)".
    # So the interpolated interior goes into a guard-retaining copy of the NEW
    # run, and the guard rows come from the OLD run's guard-retaining squash
    # (the sheath values, which the originals copied across too) — or, when no
    # such squash exists or its x/z shape differs (after interp_x/coarsen_x),
    # a zero-gradient copy of the adjacent interpolated row. Hermes-3 resets
    # these rows from the sheath condition on the first RHS call, so they only
    # have to be finite and sensible, not exact.
    #
    # The old 'sheath' mode copied the guards but never moved the interior
    # in, so the restart either lacked guards or held the template's state.
    interior_file = write_file
    tmpl_file = os.path.join(dst_dir, dst_guard_data)
    out_file = os.path.join(dst_dir, dst_guard_write)
    src_g = os.path.join(src_dir, src_guard_data) if src_guard_data else None
    for f, hint in ((interior_file, "run interp_y/interp_x/coarsen_x first"),
                    (out_file, f"cp {tmpl_file} {out_file}  (squash the new run "
                               f"with yguards='include_upper' first)")):
        if not os.path.exists(f):
            sys.exit(f"missing {f} — {hint}")
    if os.path.exists(tmpl_file) and os.path.samefile(tmpl_file, out_file):
        sys.exit(f"{out_file} IS {tmpl_file}. Make a real copy.")

    def guard_layout(datafile, t):
        """Domain-row positions and guard blocks of a dump of grid `t`.

        Returns (myg, dom, blocks): `dom[g]` is the dump row holding domain row
        g; `blocks` lists (guard rows, adjacent domain row) for each y boundary
        in file order — lower, [upper-A, lower-B,] upper. myg is 0 for a
        guardless dump. Region A sits at +MYG, region B at +3*MYG.
        """
        _, n, _, _ = dump_poloidal_info(datafile)
        ny, nyi = t["ny"], t["ny_inner"]
        dn = t["jyseps2_1"] != t["jyseps1_2"]
        nb = 4 if dn else 2
        if n == ny:
            return 0, np.arange(ny), []
        if (n - ny) % nb:
            sys.exit(f"{datafile}: {n} poloidal rows is not ny={ny} + "
                     f"{nb}*MYG — dump and grid are not a pair")
        m = (n - ny) // nb
        g = np.arange(ny)
        if dn:
            dom = np.where(g < nyi, g + m, g + 3 * m)
            blocks = [(np.arange(0, m), m),
                      (np.arange(nyi + m, nyi + 2 * m), nyi + m - 1),
                      (np.arange(nyi + 2 * m, nyi + 3 * m), nyi + 3 * m),
                      (np.arange(ny + 3 * m, ny + 4 * m), ny + 3 * m - 1)]
        else:
            dom = g + m
            blocks = [(np.arange(0, m), m),
                      (np.arange(ny + m, ny + 2 * m), ny + m - 1)]
        return m, dom, blocks

    myg_in, dom_in, _ = guard_layout(interior_file, t_dst)
    myg_out, dom_out, blk_out = guard_layout(out_file, t_dst)
    if myg_out == 0:
        sys.exit(f"{out_file} has no y-guard rows — squash the new run with "
                 f"yguards='include_upper'")
    if myg_out != run_myg:
        print(f"  NOTE {os.path.basename(out_file)} has MYG={myg_out}, "
              f"run_myg={run_myg}; using the file's {myg_out}")
    blk_src = None
    if src_g and os.path.exists(src_g):
        myg_s, _, blk_s = guard_layout(src_g, t_src)
        if myg_s == myg_out and len(blk_s) == len(blk_out):
            blk_src = blk_s
        else:
            print(f"  NOTE {os.path.basename(src_g)} has MYG={myg_s}, "
                  f"{len(blk_s)} y-boundaries — not usable, zero-gradient fill")
    elif src_g:
        print(f"  NOTE {src_g} not found — zero-gradient guard fill")

    a = nc.Dataset(interior_file, "r")
    b = nc.Dataset(out_file, "r" if dry_run else "r+")
    s = nc.Dataset(src_g, "r") if blk_src is not None else None
    print(f"\n{'DRY RUN — ' if dry_run else ''}add_yguards -> {out_file}"
          f"\n  interior : {os.path.basename(interior_file)}"
          f" ({len(dom_in)} rows, MYG={myg_in})"
          f"\n  guards   : {myg_out} rows x {len(blk_out)} boundaries from "
          f"{os.path.basename(src_g) + ' (old run)' if s else 'zero-gradient'}")
    absent = [v for v in variables if v not in a.variables or v not in b.variables]
    if absent:
        print(f"  not in both files (skipped): {absent}")
    print("  {:8s}{:>12s}{:>12s}{:>12s}  {}".format(
        "var", "interior", "guards", "max|guard|", "guard source"))
    n_done = 0
    # Only the evolved fields: diagnostics are recomputed on restart and some
    # (efe_*_ylow, ...) legitimately hold NaN in their guard rows.
    for v in variables:
        if v in absent:
            continue
        bv = b[v]
        av = a[v]
        if av.ndim != bv.ndim or av.shape[1] != bv.shape[1] or \
                av.shape[3:] != bv.shape[3:]:
            print(f"  {v:8s} skipped: shape {av.shape[1:]} vs {bv.shape[1:]}")
            continue
        new = np.array(np.ma.getdata(bv[tind]))         # (x, y_guard[, z])
        new[:, dom_out] = np.asarray(av[tind])[:, dom_in]
        src_tag = "zero-gradient"
        sv = None
        if s is not None and v in s.variables and \
                s[v].shape[1] == bv.shape[1] and s[v].shape[3:] == bv.shape[3:]:
            sv = np.asarray(s[v][tind])
            src_tag = "old run"
        for k, (rows, adj) in enumerate(blk_out):
            if sv is not None:
                new[:, rows] = sv[:, blk_src[k][0]]
            else:
                new[:, rows] = new[:, [adj] * len(rows)]
        gmask = np.zeros(new.shape[1], bool)
        for rows, _ in blk_out:
            gmask[rows] = True
        if not np.all(np.isfinite(new)):
            sys.exit(f"{v}: non-finite values after the merge — not writing")
        print(f"  {v:8s}{new[:, ~gmask].mean():12.4e}{new[:, gmask].mean():12.4e}"
              f"{np.abs(new[:, gmask]).max():12.4e}  {src_tag}")
        if not dry_run:
            bv[tind] = new
        n_done += 1
    a.close(); b.close()
    if s is not None:
        s.close()
    print(f"add_yguards {'dry run' if dry_run else 'done'} ({n_done} variables)")
    print("NEXT: " + ("set dry_run=False to write." if dry_run else
                      f"mode='to_restart' (it reads {dst_guard_write})."))


elif mode == "to_restart":
    import xhermes
    import xarray as xr
    xr.set_options(file_cache_maxsize=224)

    # The restart needs run_myg y-guard rows per processor. A guardless squash
    # records MYG=0, and xBOUT only pads guards when MYG>0, so splitting
    # boutdata_write.nc gave 8x4x27 restarts that BOUT++ rejects ("Size
    # mismatch ... Tensor (8, 4, 27) cannot be converted to Field3D (8, 8,
    # 27)"). So read the add_yguards output whenever the run has guards.
    restart_src = write_file
    _, _ny_dump, _, _ = dump_poloidal_info(write_file) \
        if os.path.exists(write_file) else (None, 0, None, None)
    if _ny_dump <= t_dst["ny"] and run_myg > 0:
        guard_file = os.path.join(dst_dir, dst_guard_write)
        if not os.path.exists(guard_file):
            sys.exit(f"{write_file} has no y-guard rows but the run has "
                     f"MYG={run_myg}: the restart would be short by "
                     f"{2 * run_myg} rows per processor.\n"
                     f"     Run mode='add_yguards' first (writes {guard_file});\n"
                     f"     it needs {os.path.join(dst_dir, dst_guard_data)} — see the\n"
                     f"     PREFLIGHT note above for the squashoutput command.")
        restart_src = guard_file
        _, _ny_dump, _, _ = dump_poloidal_info(restart_src)
        print(f"\n{os.path.basename(write_file)} has no y-guards -> reading "
              f"{os.path.basename(restart_src)} (from add_yguards)")
    if not os.path.exists(restart_src):
        sys.exit(f"{restart_src} missing")
    if _ny_dump > t_dst["ny"]:
        # the file's own MYG must describe its rows, or xBOUT mis-splits them
        dn = t_dst["jyseps2_1"] != t_dst["jyseps1_2"]
        myg_rows = (_ny_dump - t_dst["ny"]) // (4 if dn else 2)
        fix_squash_myg(restart_src, myg_rows, allow_file_fixup)
    # No grid file: `to_restart` only splits the arrays across processors and
    # needs no geometry, while passing a guardless grid alongside a
    # guard-retaining dump makes xBOUT reject the pair outright — the same trap
    # the other scripts here avoid by loading without geometry. All the
    # metadata it needs (nx/ny/MXG/MYG/jyseps*) is in the dump.
    ds = xhermes.open_hermesdataset(
        restart_src, keep_xboundaries=True,
        keep_yboundaries=_ny_dump > t_dst["ny"],
        info=False, unnormalise=False)
    print(f"\nwriting BOUT.restart.* into {dst_dir}  (nxpe={nxpe}, nype={nype})")
    ds.bout.to_restart(savepath=dst_dir, tind=tind, nxpe=nxpe, nype=nype,
                       overwrite=True)
    n_files, n_attrs = strip_fill_values(dst_dir)
    print(f"  stripped _FillValue from {n_attrs} variables across "
          f"{n_files} restart files")
    # BOUT++ reads each field at (MXSUB+2*MXG, MYSUB+2*MYG, nz) and aborts
    # on anything else, so check the first file says what the run expects
    with nc.Dataset(os.path.join(dst_dir, "BOUT.restart.0.nc")) as r:
        shp = r["Ne"].shape if "Ne" in r.variables else None
        want_y = int(r["MYSUB"][:]) + 2 * run_myg
    print(f"  BOUT.restart.0.nc: Ne {shp}, run expects y = MYSUB + 2*MYG = {want_y}")
    if shp is not None and shp[1] != want_y:
        print(f"  WARNING y size {shp[1]} != {want_y}: BOUT++ will refuse "
              f"this restart (check run_myg and the y-guard rows of "
              f"{os.path.basename(restart_src)})")
    print("to_restart done")
    print("NEXT: copy BOUT.restart.* into the new run directory and start it.\n"
          "      Expect a transient: interp_x/interp_y do not conserve, and\n"
          "      coarsen_x does conserve but the sub-grid scales are gone.")


elif mode == "interp_z":
    # ── TOROIDAL refinement, straight on the restart files ───────────────────
    # z needs none of the machinery the other modes exist for: it has NO guard
    # cells, no regions and no branch cuts, it is uniformly spaced and exactly
    # periodic, and the Hypnotoad grid has no z at all (MZ is purely a BOUT.inp
    # setting).  So there is no new grid, no template dump and no squash
    # round-trip: read the old per-processor restarts, resize the z axis, write
    # new ones.  The x/y decomposition is untouched, so NXPE/NYPE carry over
    # and the new run can use the same layout.  That also sidesteps xBOUT's
    # `to_restart` and its `_FillValue` trap.
    if not os.path.isdir(z_src_dir):
        sys.exit(f"z_src_dir not found: {z_src_dir}")
    files = sorted(fnmatch.filter(os.listdir(z_src_dir), "BOUT.restart.*.nc"),
                   key=lambda s: int(s.split(".")[2]))
    if not files:
        sys.exit(f"no BOUT.restart.*.nc in {z_src_dir}")

    with nc.Dataset(os.path.join(z_src_dir, files[0])) as f0:
        # A real BOUT++/Hermes-3 restart carries TWO PARALLEL DIMENSION SETS of
        # identical sizes -- `('x','y','z')` and `('dim_0','dim_1','dim_2')` --
        # and the 3-D variables are split between them (7 vs 19 on the SXD
        # ny36 case). Taking the z name from the FIRST 3-D variable resizes one
        # set and leaves the other at the old length, so assigning the
        # interpolated array fails with
        #   ValueError: shape mismatch ... (8, 8, 81) and (8, 8, 162)
        # Collect EVERY last-axis dimension name instead.
        nz_old, zdims, odd = None, set(), []
        for vn, v in f0.variables.items():
            if v.ndim != 3:
                continue
            if nz_old is None:
                nz_old = v.shape[-1]
            if v.shape[-1] == nz_old:
                zdims.add(v.dimensions[-1])
            else:
                odd.append((vn, v.shape[-1]))
        if nz_old is None:
            sys.exit("no 3-D (Field3D) variables in the restart — nothing to do")
        if odd:
            sys.exit("3-D variables disagree on the toroidal length: "
                     f"{nz_old} vs {odd} — refusing to guess which is z")
        meta = {k: (int(f0[k][:].ravel()[0]) if k in f0.variables else None)
                for k in ("nz", "MZ", "MZSUB", "MZG", "NZPE", "zperiod")}
    print(f"\n{len(files)} restart files in {z_src_dir}")
    print(f"  z dimension(s) {sorted(zdims)}:  MZ {nz_old} -> {mz_new}"
          f"   ({mz_new / nz_old:g}x)")
    print("  stored metadata: "
          + ", ".join(f"{k}={v}" for k, v in meta.items() if v is not None))
    # The MZ-vs-nz convention CHANGED in BOUT++ history: `nz` used to be stored
    # as `MZ` with MZ = nz + 1 (see xbout/load.py:414).  So preserve whatever
    # relation THIS file uses rather than assuming MZ == nz.
    offsets = {k: (meta[k] - nz_old) for k in ("nz", "MZ", "MZSUB")
               if meta[k] is not None}
    new_meta = {k: mz_new + off for k, off in offsets.items()}
    print("  will rewrite: "
          + ", ".join(f"{k} {meta[k]} -> {v}" for k, v in new_meta.items()))
    if any(off != 0 for off in offsets.values()):
        print("  NOTE: a nonzero MZ-vs-nz offset was found and is preserved: "
              + ", ".join(f"{k}{off:+d}" for k, off in offsets.items()
                          if off != 0))
    if mz_new < nz_old:
        print("  COARSENING: spectral truncation, which is the exact low-pass "
              "for a\n    uniform periodic axis — unlike the radial direction, "
              "no conservative\n    averaging is needed (mode 0 is untouched, "
              "so integrals are preserved).")

    if dry_run:
        print("\n[dry run] nothing written. Set dry_run = False to write.")
    else:
        os.makedirs(z_dst_dir, exist_ok=True)
        if os.path.abspath(z_dst_dir) == os.path.abspath(z_src_dir):
            sys.exit("z_dst_dir must differ from z_src_dir (never write over "
                     "the restarts you are reading)")

    worst_mean, worst_mean_var = 0.0, ""
    neg_report = []
    nan_report = {}
    for fi, fname in enumerate(files):
        src = os.path.join(z_src_dir, fname)
        with nc.Dataset(src) as si:
            if dry_run:
                so = None
            else:
                so = nc.Dataset(os.path.join(z_dst_dir, fname), "w",
                                format=si.data_model)
                for dname, dim in si.dimensions.items():
                    n = mz_new if dname in zdims else (
                        None if dim.isunlimited() else len(dim))
                    so.createDimension(dname, n)
                so.setncatts({a: si.getncattr(a) for a in si.ncattrs()})

            for vname, vin in si.variables.items():
                arr = vin[:]
                is_field3d = (vin.ndim == 3 and vin.shape[-1] == nz_old)
                if is_field3d:
                    old = np.asarray(arr, dtype=float)
                    new = interpft_z(old, mz_new, axis=-1)
                    # mode 0 is untouched by construction, so the z-average --
                    # and hence every volume integral -- must be preserved to
                    # round-off. This is the free conservation check.
                    d = float(np.abs(new.mean(axis=-1) - old.mean(axis=-1)).max())
                    scale = float(np.abs(old).max()) or 1.0
                    if d / scale > worst_mean:
                        worst_mean, worst_mean_var = d / scale, vname
                    if vname in positive_vars and new.min() < 0.0 <= old.min():
                        neg_report.append((fname, vname, float(new.min())))
                    # An FFT of a z row containing ONE non-finite value returns
                    # an all-non-finite row, so a partial NaN is smeared across
                    # the whole toroidal row. Harmless for diagnostics that
                    # BOUT++ does not read back as state (and these restarts do
                    # carry NaN in their y-guard rows), fatal for an evolved
                    # field. Compare against the count expected if the affected
                    # rows had ALREADY been wholly non-finite.
                    _no = int(np.count_nonzero(~np.isfinite(old)))
                    if _no:
                        _nn = int(np.count_nonzero(~np.isfinite(new)))
                        if _nn > _no * (mz_new / nz_old) + 0.5:
                            nan_report.setdefault(vname, (_no, _nn))
                    arr = new
                elif vname in new_meta:
                    arr = np.asarray(arr).astype(vin.dtype) * 0 + new_meta[vname]
                if so is not None:
                    vout = so.createVariable(vname, vin.dtype, vin.dimensions)
                    vout.setncatts({a: vin.getncattr(a) for a in vin.ncattrs()})
                    if _is_vlen_str(vin):
                        # netCDF4 REFUSES `v[:] = ...` on a VLEN string
                        # variable: "data can only be assigned to VLEN
                        # variables using integer indices". Real BOUT++
                        # restarts carry `run_id` and `run_restart_from` this
                        # way, so every real file hits this — a fixture without
                        # string variables passes and hides it.
                        _v = np.asarray(arr, dtype=object).ravel()
                        for _i, _s in enumerate(_v):
                            vout[_i] = "" if _s is None else str(_s)
                    else:
                        vout[:] = arr
            if so is not None:
                so.close()
        if fi == 0 or (fi + 1) % 16 == 0 or fi == len(files) - 1:
            print(f"  [{fi + 1:4d}/{len(files)}] {fname}", flush=True)

    print(f"\n  z-mean preserved to {worst_mean:.2e} (relative, worst: "
          f"{worst_mean_var or 'n/a'}) — must be ~1e-15")
    if nan_report:
        print(f"  NOTE: an FFT spreads a single non-finite value across its "
              f"whole z row, and {len(nan_report)} variable(s) gained "
              "non-finite cells beyond\n        the pure 1:1 row scaling:")
        for _v, (_o, _n) in sorted(nan_report.items()):
            _tag = "   <-- EVOLVED, THIS MATTERS" if _v in variables else ""
            print(f"    {_v:22s} {_o} -> {_n} non-finite "
                  f"(1:1 would be {int(round(_o * mz_new / nz_old))}){_tag}")
        print("    Diagnostics BOUT++ does not read back as state are harmless "
              "(these\n    restarts carry NaN in their y-guard rows by "
              "design). For anything in\n    `variables`, fix the source "
              "restart first — do not restart from this.")

    if neg_report:
        # STORM evolves logn/logp so positivity is automatic there; Hermes-3
        # evolves N and P directly, so band-limited interpolation can undershoot
        # between the coarse samples. abs()-clamping would break the exactness,
        # so this REPORTS instead of silently fixing.
        print(f"  WARNING: {len(neg_report)} field(s) went negative where the "
              "source was not:")
        for fn, vn, mn in neg_report[:8]:
            print(f"    {fn}  {vn}  min = {mn:.3e}")
        print("    Gibbs undershoot between coarse samples. Raise density_floor,"
              "\n    or smooth the source in z first — do NOT abs()-clamp, that "
              "destroys\n    the spectral exactness that makes this mode "
              "conservative.")
    if not dry_run:
        print(f"\ninterp_z done -> {z_dst_dir}")
        print(f"NEXT: set MZ = {mz_new} in the new run's BOUT.inp (zperiod "
              f"unchanged at {meta.get('zperiod')}), copy these restarts in,\n"
              "      and start it. No grid change is needed — the grid file has "
              "no z.")


elif mode == "copy_restart":
    with nc.Dataset(os.path.join(copy_from, "BOUT.restart.0.nc")) as f:
        n_proc = int(f["NXPE"][:]) * int(f["NYPE"][:])
    print(f"\ncopying {restart_vars} across {n_proc} restart files\n"
          f"  {copy_from} -> {copy_to}")
    for i in range(n_proc):
        a = nc.Dataset(os.path.join(copy_from, f"BOUT.restart.{i}.nc"))
        b = nc.Dataset(os.path.join(copy_to, f"BOUT.restart.{i}.nc"), "r+")
        if a["Ne"].shape != b["Ne"].shape:
            a.close(); b.close()
            sys.exit(f"restart {i}: shapes differ {a['Ne'].shape} vs "
                     f"{b['Ne'].shape} — different decomposition?")
        for v in restart_vars:
            b[v][:] = a[v][:]
        a.close(); b.close()
    print("copy_restart done")


elif mode == "check_result":
    # Interpolation is not conservative, but a few percent is expected and a
    # large change means a mis-mapped region or a bad psi mapping.
    if not os.path.exists(write_file):
        sys.exit(f"{write_file} missing")
    fix_squash_myg(write_file, t_dst["MYG"], allow_file_fixup)
    import xhermes
    ds3 = xhermes.open_hermesdataset(
        write_file, geometry="toroidal", gridfilepath=dst_grid,
        keep_xboundaries=True, keep_yboundaries=t_dst["yguards"],
        info=False, unnormalise=False)

    def total(ds, v):
        if v not in ds.variables:
            return np.nan
        dV = ds["J"] * ds["dx"] * ds["dy"] * ds["dz"]
        return float((ds[v].isel(t=tind) * dV).sum().values)

    print("\nvolume-integrated totals (normalised units, full torus factor omitted)")
    print("  {:6s}{:>16s}{:>16s}{:>10s}".format("var", "source", "refined", "change"))
    for v in ("Ne", "Pe", "Pi"):
        a, b = total(ds1, v), total(ds3, v)
        if np.isfinite(a) and np.isfinite(b) and a != 0:
            print(f"  {v:6s}{a:16.6e}{b:16.6e}{100*(b/a-1):9.2f}%")
    print("\n  Interpolation does not conserve; a few percent is normal, and a"
          "\n  large change usually means a mis-mapped region or bad psi mapping.")

else:
    sys.exit(f"unknown mode {mode!r}")
