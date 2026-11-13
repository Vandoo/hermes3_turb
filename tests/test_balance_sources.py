"""Guard masking in volume_integral, and the 3/2 on the energy source.

Two properties, both checked analytically and both load-bearing for any
energy or particle budget:

* a source volume integral runs over interior cells only. Guard cells carry a
  full volume element, and whether they also carry a nonzero source is
  grid-dependent (zero in the x-guards on the SXD ny36 grid, +5.3 % on
  `from_archer_6kW`), so a correct integral cannot rely on them vanishing.
* the energy source is `3/2*P<sp>_src`, not `P<sp>_src`. The storage term and
  the 5/2 enthalpy flux carry their own factors, so a budget built on the raw
  pressure source closes 1.5x high.
"""
import unittest
from pathlib import Path
import numpy as np
import xarray as xr

import hermes3_turb as h3t
from hermes3_turb.balance import (
    guard_rows, interior_mask, volume_integral, particle_source, energy_source,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def balance_dataset(nx=10, ny=8, nz=4, mxg=2, ybg=0, ny_inner=None,
                    jyseps=(1, 3, 5, 7), unit_volume=True):
    """Synthetic dump with a known volume and a unit source everywhere.

    `ny` is the INTERIOR poloidal size; `ybg` guard rows are added per guard
    band, four bands in a double null (ny+4*ybg) and two in a single null
    (ny+2*ybg), exactly as xBOUT returns a dump built from a grid with
    `y_boundary_guards=ybg`. A single null is signalled, as in a real grid, by
    `jyseps2_1 == jyseps1_2`.
    """
    ny_inner = ny // 2 if ny_inner is None else ny_inner
    j11, j21, j12, j22 = jyseps
    n_bands = 4 if (ny_inner > 0 and j21 != j12) else 2
    ny_arr = ny + n_bands * ybg
    meta = {
        "MXG": mxg, "MYG": 2, "nx": nx, "ny": ny, "ny_inner": ny_inner,
        "ixseps1": nx // 2, "ixseps2": nx // 2,
        "jyseps1_1": j11, "jyseps2_1": j21, "jyseps1_2": j12, "jyseps2_2": j22,
        "Nnorm": 1e19, "Tnorm": 50.0, "rho_s0": 1.0,
    }
    ds = xr.Dataset(attrs={"metadata": meta})
    # J*dx*dy = 1 per cell, so ∫dV over the full torus = 2π * (#cells summed).
    for name in ("J", "dx", "dy"):
        ds[name] = xr.DataArray(np.ones((nx, ny_arr)), dims=("x", "theta"))
    if not unit_volume:
        # Fat guard columns, as a separatrix-packed dx really has.
        ds["dx"].values[:mxg, :] = 5.0
        ds["dx"].values[nx - mxg:, :] = 5.0
    ds["dz"] = xr.DataArray(2 * np.pi / nz)
    shape = (nx, ny_arr, nz)
    ds["Se_src"] = xr.DataArray(np.ones(shape), dims=("x", "theta", "zeta"))
    ds["Pe_src"] = xr.DataArray(np.full(shape, 2.0), dims=("x", "theta", "zeta"))
    return ds


class TestGuardRows(unittest.TestCase):
    def test_no_guards_gives_empty_list(self):
        self.assertEqual(guard_rows(balance_dataset(ybg=0)), [])

    def test_double_null_layout_is_derived_from_the_dump_not_myg(self):
        # Infer the guard count from the dump when ybg differs from MYG.
        ds = balance_dataset(ny=8, ybg=1, ny_inner=4)
        # bands: [0], [ny_inner+1 .. ny_inner+2] = [5,6], [ny+3 .. ny+3] = [11]
        self.assertEqual(guard_rows(ds), [0, 5, 6, 11])
        self.assertEqual(ds.sizes["theta"], 12)

    def test_double_null_with_myg_sized_guards(self):
        ds = balance_dataset(ny=8, ybg=2, ny_inner=4)
        self.assertEqual(guard_rows(ds), [0, 1, 6, 7, 8, 9, 14, 15])

    def test_single_null_detected_when_jyseps2_1_equals_jyseps1_2(self):
        ds = balance_dataset(ny=8, ybg=2, ny_inner=0, jyseps=(1, 4, 4, 7))
        self.assertEqual(guard_rows(ds), [0, 1, 10, 11])

    def test_uninferrable_layout_raises_instead_of_keeping_guards(self):
        ds = balance_dataset(ny=8, ybg=0)
        ds.attrs["metadata"] = dict(ds.metadata, ny=7)   # 1 extra row: odd
        with self.assertRaises(ValueError):
            guard_rows(ds)


class TestInteriorMask(unittest.TestCase):
    def test_x_guards_are_dropped_on_both_sides(self):
        mask = interior_mask(balance_dataset(nx=10, mxg=2, ybg=0))
        self.assertTrue(mask[2:8, :].all())
        self.assertFalse(mask[:2, :].any())
        self.assertFalse(mask[8:, :].any())

    def test_y_guard_rows_are_dropped_too(self):
        ds = balance_dataset(ny=8, ybg=2, ny_inner=4)
        mask = interior_mask(ds)
        for j in guard_rows(ds):
            self.assertFalse(mask[:, j].any(), f"row {j} kept")
        self.assertEqual(mask.sum(), (10 - 4) * (16 - 8))

    def test_strip_yguards_false_keeps_the_rows(self):
        ds = balance_dataset(ny=8, ybg=2, ny_inner=4)
        self.assertEqual(interior_mask(ds, strip_yguards=False).sum(),
                         (10 - 4) * 16)


class TestVolumeIntegral(unittest.TestCase):
    def test_unit_source_over_known_volume(self):
        # 6 interior x-columns x 8 rows, unit source, unit volume element.
        ds = balance_dataset(nx=10, ny=8, mxg=2, ybg=0)
        self.assertAlmostEqual(float(volume_integral(ds, ds["Se_src"])),
                               6 * 8 * 2 * np.pi)

    def test_interior_only_excludes_the_guards(self):
        ds = balance_dataset(nx=10, ny=8, mxg=2, ybg=0)
        full = float(volume_integral(ds, ds["Se_src"], interior_only=False))
        interior = float(volume_integral(ds, ds["Se_src"]))
        self.assertAlmostEqual(full, 10 * 8 * 2 * np.pi)
        self.assertAlmostEqual(interior, 6 * 8 * 2 * np.pi)
        self.assertGreater(full, interior)

    def test_fat_guard_columns_dominate_the_guard_contribution(self):
        # dx is separatrix-packed in these grids, so the guard contribution is
        # much larger than 4/nx.
        ds = balance_dataset(nx=10, ny=8, mxg=2, ybg=0, unit_volume=False)
        full = float(volume_integral(ds, ds["Se_src"], interior_only=False))
        interior = float(volume_integral(ds, ds["Se_src"]))
        self.assertAlmostEqual(interior, 6 * 8 * 2 * np.pi)
        self.assertAlmostEqual(full, (6 + 4 * 5) * 8 * 2 * np.pi)

    def test_y_guard_rows_are_excluded_as_well(self):
        ds = balance_dataset(nx=10, ny=8, ybg=2, ny_inner=4)
        self.assertAlmostEqual(float(volume_integral(ds, ds["Se_src"])),
                               6 * 8 * 2 * np.pi)

    def test_a_source_that_is_zero_in_the_guards_is_unaffected(self):
        # On a grid whose sources vanish in the x-guards the two answers
        # agree exactly, so agreement here proves nothing about the masking —
        # which is why the cases above use a source that does not vanish.
        ds = balance_dataset(nx=10, ny=8, mxg=2, ybg=0)
        ds["Se_src"].values[:2, :, :] = 0.0
        ds["Se_src"].values[8:, :, :] = 0.0
        self.assertAlmostEqual(
            float(volume_integral(ds, ds["Se_src"], interior_only=False)),
            float(volume_integral(ds, ds["Se_src"])))


class TestSourceHelpers(unittest.TestCase):
    def test_energy_source_carries_the_three_halves(self):
        ds = balance_dataset()
        np.testing.assert_allclose(energy_source(ds, "e").values,
                                   1.5 * ds["Pe_src"].values)

    def test_particle_source_carries_no_factor(self):
        ds = balance_dataset()
        np.testing.assert_allclose(particle_source(ds, "e").values,
                                   ds["Se_src"].values)

    def test_missing_species_returns_none_rather_than_raising(self):
        ds = balance_dataset()
        self.assertIsNone(energy_source(ds, "i"))
        self.assertIsNone(particle_source(ds, "i"))

    def test_the_three_halves_rescales_a_closure_by_two_thirds(self):
        # A budget whose flux+storage is 1.572x the raw pressure-source
        # integral closes at 104.8 % against the energy source. These are the
        # measured numbers on sxd_MU/new_ny36/fresh_ES/140.
        ds = balance_dataset()
        S_raw = float(volume_integral(ds, ds["Pe_src"]))
        S_eng = float(volume_integral(ds, energy_source(ds, "e")))
        flux_plus_storage = 1.572 * S_raw
        self.assertAlmostEqual(flux_plus_storage / S_raw * 100, 157.2, places=4)
        self.assertAlmostEqual(flux_plus_storage / S_eng * 100, 104.8, places=4)

    def test_both_helpers_are_exported_from_the_package(self):
        for name in ("guard_rows", "interior_mask",
                     "particle_source", "energy_source"):
            self.assertIn(name, h3t.__all__)
            self.assertTrue(hasattr(h3t, name))


class TestScriptsUseTheHelpers(unittest.TestCase):
    """The three call sites that need the source helpers."""

    def test_balance_diagnose_uses_energy_source(self):
        src = (SCRIPTS / "balances/v3_balance_diagnose.py").read_text()
        self.assertIn("h3t.energy_source(ds, species)", src)
        self.assertNotIn("volume_integral(ds, ds[src_P_name])", src)

    def test_plasma_balance_uses_energy_source_in_both_places(self):
        src = (SCRIPTS / "balances/v3_plasma_balance.py").read_text()
        self.assertEqual(src.count("h3t.energy_source(ds, species)"), 2)
        self.assertNotIn("source_density=ds[src_P_name]", src)
        self.assertNotIn("source_profile(src_P_name)", src)

    def test_balance_diagnose_reports_the_lcfs_area_without_the_legs(self):
        src = (SCRIPTS / "balances/v3_balance_diagnose.py").read_text()
        self.assertIn("sol_band_regions", src)
        self.assertNotIn('_A = elem.isel(x=ixsep1).sum(pdim)', src)

    def test_plasma_balance_cache_suffix_is_at_least_b9(self):
        """Accept compatible cache formats at or above the required version.
        """
        import re
        src = (SCRIPTS / "balances/v3_plasma_balance.py").read_text()
        tags = re.findall(r'"b(\d+)"\)', src)
        self.assertTrue(tags, "no b<N> cache tag found")
        self.assertGreaterEqual(max(int(t) for t in tags), 9)

    def test_no_v3_loader_keys_its_guard_detection_on_myg(self):
        """Guard detection lives in `h3t.guard_rows` only. A per-script
        `npol == ny + 4*MYG` test strips nothing when the grid's
        y_boundary_guards differs from MYG, and does so silently."""
        # v3_target_heatflux.py is deliberately excluded: it KEEPS the
        # y-guards (the plate cells are the measurement) and uses
        # `_expected_guard_rows` to validate the layout, not to strip it.
        stripping = ["v3_radial_flux", "v3_radial_conservation",
                     "v3_plasma_balance", "v3_poloidal_flux",
                     "v3_balance_diagnose"]
        self.assertEqual(
            sorted(p.stem for p in SCRIPTS.rglob("v3_*.py")),
            sorted(stripping + ["v3_target_heatflux"]),
            "a new v3_* script appeared: give it h3t.guard_rows too")
        for name in stripping:
            src = (next(SCRIPTS.rglob(f"{name}.py"))).read_text()
            self.assertNotIn("_MYG0", src, f"{name} still keys on MYG")
            self.assertIn("h3t.guard_rows(ds, ny_grid=_ny_g)", src, name)


if __name__ == "__main__":
    unittest.main()
