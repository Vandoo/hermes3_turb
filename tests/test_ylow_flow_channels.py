"""`ef*_tot_ylow` is not the pressure equation's parallel energy flow.

`evolve_pressure` builds the published total as

    flow_ylow = flow_ylow_advection
              + flow_ylow_viscous_heating      <- if numerical_viscous_heating
                                                  OR diagnose
              + every component's registered `energy_flow_ylow`

and the viscous-heating line sits OUTSIDE the guard that applies `Sp_nvh` to
`ddt(P)` (`evolve_pressure.cxx:317-328`). `numerical_viscous_heating` defaults
to false (`:164-166`), so on the usual `diagnose = true` run the term is
computed, published inside the total, and never applied to the equation.

Separately, `ef*_cond_ylow` IS in `ddt(P)`, but `braginskii_conduction`
registers the same physics as `energy_source` too, so it is also inside `SP`.
"""
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

import hermes3_turb as h3t
from hermes3_turb.balance import parallel_energy_flow_ylow

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# The measured per-face powers [kW], from_archer_6kW/422-425, 32 frames.
FACES = {              # face:        adv     visc    cond     tot
    "inner_lower": (-0.8, 0.1, 2.6, 1.9),
    "inner_upper": (8.5, 3.7, 0.3, 12.5),
    "outer_upper": (124.4, 31.7, 57.8, 213.9),
    "outer_lower": (122.3, 4.0, 65.7, 192.0),
}


def ylow_dataset(species=("e", "i"), adv=10.0, cond=5.0, visc=2.0,
                 with_visc=True, with_cond=True, with_tot=True):
    """Dump carrying the Y-face flow diagnostics, one constant per channel."""
    ds = xr.Dataset(attrs={"metadata": {"MXG": 2, "MYG": 2}})
    shape, dims = (3, 4), ("x", "theta")
    for sp in species:
        ds[f"ef{sp}_adv_ylow"] = xr.DataArray(np.full(shape, adv), dims=dims)
        if with_cond:
            ds[f"ef{sp}_cond_ylow"] = xr.DataArray(np.full(shape, cond), dims=dims)
        if with_visc:
            ds[f"ef{sp}_visc_heat_ylow"] = xr.DataArray(np.full(shape, visc), dims=dims)
        if with_tot:
            total = adv + (cond if with_cond else 0.0) + (visc if with_visc else 0.0)
            ds[f"ef{sp}_tot_ylow"] = xr.DataArray(np.full(shape, total), dims=dims)
    return ds


class TestTheMeasuredSplitIsExact(unittest.TestCase):
    """tot = adv + visc + cond on every face, so nothing else is published."""

    def test_each_face_adds_up(self):
        for face, (adv, visc, cond, tot) in FACES.items():
            self.assertAlmostEqual(adv + visc + cond, tot, places=1, msg=face)

    def test_totals(self):
        adv, visc, cond, tot = (sum(v[i] for v in FACES.values()) for i in range(4))
        self.assertAlmostEqual(adv, 254.4, places=1)
        self.assertAlmostEqual(visc, 39.5, places=1)
        self.assertAlmostEqual(cond, 126.4, places=1)
        self.assertAlmostEqual(tot, 420.3, places=1)
        self.assertAlmostEqual(adv + visc + cond, tot, places=1)

    def test_the_budget_term_is_adv_plus_cond_not_the_published_total(self):
        adv, visc, cond, tot = (sum(v[i] for v in FACES.values()) for i in range(4))
        # a pressure budget whose volume side has 3/2 int SP dV wants adv alone
        self.assertAlmostEqual(adv, tot - visc - cond, places=1)
        # the full parallel ENERGY flow into the legs wants adv + cond
        self.assertAlmostEqual(adv + cond, tot - visc, places=1)


class TestHelperSelectsTheRightChannels(unittest.TestCase):
    def test_default_is_advection_plus_conduction(self):
        ds = ylow_dataset()
        flow = parallel_energy_flow_ylow(ds, "e")
        np.testing.assert_allclose(flow.values, 15.0)

    def test_viscous_heating_is_excluded_by_default(self):
        ds = ylow_dataset()
        flow = parallel_energy_flow_ylow(ds, "e")
        published = ds["efe_tot_ylow"]
        self.assertAlmostEqual(float((published - flow).mean()), 2.0)

    def test_viscous_heating_can_be_asked_for(self):
        ds = ylow_dataset()
        flow = parallel_energy_flow_ylow(ds, "e", include_viscous_heating=True)
        np.testing.assert_allclose(flow.values, ds["efe_tot_ylow"].values)

    def test_conduction_can_be_dropped_for_an_SP_budget(self):
        ds = ylow_dataset()
        flow = parallel_energy_flow_ylow(ds, "e", include_conduction=False)
        np.testing.assert_allclose(flow.values, 10.0)

    def test_missing_conduction_is_not_an_error(self):
        ds = ylow_dataset(with_cond=False)
        np.testing.assert_allclose(
            parallel_energy_flow_ylow(ds, "e").values, 10.0)

    def test_missing_species_returns_none_rather_than_raising(self):
        ds = ylow_dataset(species=("e",))
        self.assertIsNone(parallel_energy_flow_ylow(ds, "d+"))

    def test_it_never_reads_the_published_total(self):
        """A dump whose tot is nonsense must not change the answer."""
        ds = ylow_dataset()
        ds["efe_tot_ylow"] = ds["efe_tot_ylow"] * 0.0 + 999.0
        np.testing.assert_allclose(
            parallel_energy_flow_ylow(ds, "e").values, 15.0)

    def test_species_are_independent(self):
        ds = ylow_dataset()
        ds["efi_adv_ylow"] = ds["efi_adv_ylow"] * 0.0 + 40.0
        self.assertAlmostEqual(
            float(parallel_energy_flow_ylow(ds, "e").mean()), 15.0)
        self.assertAlmostEqual(
            float(parallel_energy_flow_ylow(ds, "i").mean()), 45.0)

    def test_it_is_exported_from_the_package(self):
        self.assertIn("parallel_energy_flow_ylow", h3t.__all__)
        self.assertTrue(hasattr(h3t, "parallel_energy_flow_ylow"))


class TestScriptsNoLongerTrustThePublishedTotal(unittest.TestCase):
    def test_target_poloidal_check_uses_the_helper(self):
        src = (SCRIPTS / "targets/target_poloidal_check.py").read_text()
        self.assertIn("h3t.parallel_energy_flow_ylow(", src)
        self.assertIn("include_viscous_heating=False", src)

        self.assertNotIn('efe = dsa.get("efe_tot_ylow", None)', src)

    def test_validate_poloidal_efe_compares_against_adv_plus_cond(self):
        src = (SCRIPTS / "targets/validate_poloidal_efe.py").read_text()
        self.assertIn("tot_nat = adv_nat + np.nan_to_num(cond_nat)", src)
        self.assertIn("visc_name", src)
        self.assertNotIn(
            "adv_nat, cond_nat, tot_nat = native(adv_name), native(cond_name), "
            "native(tot_name)", src)

    def test_both_scripts_explain_why(self):
        for name in ("target_poloidal_check.py", "validate_poloidal_efe.py"):
            src = (next(SCRIPTS.rglob(name))).read_text()
            self.assertIn("numerical_viscous_heating", src, name)


if __name__ == "__main__":
    unittest.main()
