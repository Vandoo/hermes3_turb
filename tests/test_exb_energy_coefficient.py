"""Check the energy-per-pressure coefficient for each transport channel.

Perpendicular ExB uses 3/2; parallel advection, diamagnetic drift, and
magnetic flutter use 5/2. Anomalous diffusion uses 3/2. Assertions cover
coefficient application, decomposition, and cache-format consistency.
"""
import inspect
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

import hermes3_turb as h3t
from hermes3_turb import fluxes

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def exb_dataset(nx=6, ny=4, nz=8):
    """Dataset whose ExB radial velocity is a known constant."""
    meta = {"MXG": 1, "MYG": 1, "ny": ny, "ny_inner": ny // 2,
            "Nnorm": 1e19, "Tnorm": 50.0, "ixseps1": nx // 2}
    ds = xr.Dataset(attrs={"metadata": meta})
    shape = (nx, ny, nz)
    z = 2 * np.pi * np.arange(nz) / nz
    # phi = sin(zeta) -> dphi/dzeta = cos(zeta), a clean non-zero v_ExB
    ds["phi"] = xr.DataArray(
        np.broadcast_to(np.sin(z), shape).copy(), dims=("x", "theta", "zeta"))
    ds["Pe"] = xr.DataArray(np.full(shape, 3.0), dims=("x", "theta", "zeta"))
    for name, val in (("g11", 1.0), ("g_22", 1.0), ("g_23", 0.0),
                      ("g_33", 1.0), ("Bxy", 1.0), ("dx", 1.0), ("dy", 1.0)):
        ds[name] = xr.DataArray(np.full((nx, ny), val), dims=("x", "theta"))
    ds["dz"] = xr.DataArray(2 * np.pi / nz)
    return ds


class TestCoefficientValue(unittest.TestCase):
    def test_exb_energy_coefficient_is_three_halves(self):
        self.assertEqual(fluxes.EXB_ENERGY_COEFF, 1.5)

    def test_it_is_exported(self):
        self.assertIn("EXB_ENERGY_COEFF", h3t.__all__)
        self.assertIs(h3t.EXB_ENERGY_COEFF, fluxes.EXB_ENERGY_COEFF)

    def test_radial_heat_flux_defaults_to_it(self):
        sig = inspect.signature(fluxes.radial_heat_flux)
        self.assertEqual(sig.parameters["coefficient"].default, 1.5)

    def test_decompose_radial_heat_flux_defaults_to_it(self):
        sig = inspect.signature(fluxes.decompose_radial_heat_flux)
        self.assertEqual(sig.parameters["coefficient"].default, 1.5)


class TestCoefficientIsApplied(unittest.TestCase):
    """Not just declared — actually multiplied in.

    `ExB_radial_velocity` needs xBOUT's `.bout` accessor, which is not
    available in every environment, so it is replaced by a known field. That
    is the point of these tests anyway: they check the COEFFICIENT, not the
    velocity formula.
    """

    def setUp(self):
        self.ds = exb_dataset()
        self.v = xr.DataArray(
            np.linspace(1.0, 2.0, 6 * 4 * 8).reshape(6, 4, 8),
            dims=("x", "theta", "zeta"))
        self._real = fluxes.ExB_radial_velocity
        fluxes.ExB_radial_velocity = lambda ds, phi, **kw: self.v

    def tearDown(self):
        fluxes.ExB_radial_velocity = self._real

    def test_radial_heat_flux_is_coefficient_times_P_times_v(self):
        P = self.ds["Pe"]
        q = fluxes.radial_heat_flux(self.ds, P, self.ds["phi"])
        np.testing.assert_allclose(q.values, 1.5 * P.values * self.v.values,
                                   rtol=1e-14)

    def test_it_scales_linearly_with_the_coefficient(self):
        P = self.ds["Pe"]
        q15 = fluxes.radial_heat_flux(self.ds, P, self.ds["phi"],
                                      coefficient=1.5)
        q25 = fluxes.radial_heat_flux(self.ds, P, self.ds["phi"],
                                      coefficient=2.5)
        self.assertGreater(float(np.abs(q15).max()), 0.0)
        np.testing.assert_allclose(q25.values, (2.5 / 1.5) * q15.values,
                                   rtol=1e-14)

    def test_the_old_value_is_five_thirds_of_the_new_one(self):
        """Compare the 5/2 enthalpy form with the 3/2 ExB energy form.
        """
        P = self.ds["Pe"]
        old = fluxes.radial_heat_flux(self.ds, P, self.ds["phi"],
                                      coefficient=2.5)
        new = fluxes.radial_heat_flux(self.ds, P, self.ds["phi"])
        self.assertAlmostEqual(
            float(np.abs(old).max() / np.abs(new).max()), 5.0 / 3.0, places=12)

    def test_decompose_splits_without_changing_the_total(self):
        P = self.ds["Pe"]
        q_mean, q_turb, q_tot = fluxes.decompose_radial_heat_flux(
            self.ds, P, self.ds["phi"], zdim="zeta")
        np.testing.assert_allclose((q_mean + q_turb).values, q_tot.values,
                                   rtol=1e-14)
        direct = 1.5 * (P * self.v).mean("zeta")
        np.testing.assert_allclose(q_tot.values, direct.values, rtol=1e-14)


class TestOtherChannelsUnchanged(unittest.TestCase):
    """Only the ExB channel moves. These three must stay at their own values."""

    def test_diamagnetic_radial_flux_keeps_five_halves(self):
        src = inspect.getsource(fluxes.radial_magnetic_drift_heat_flux)
        self.assertIn("2.5", src)
        self.assertNotIn("EXB_ENERGY_COEFF", src)

    def test_poloidal_parallel_flow_keeps_five_halves(self):
        src = inspect.getsource(fluxes.total_poloidal_heat_flux)
        self.assertIn("q_par = 2.5", src)
        self.assertIn("EXB_ENERGY_COEFF * P * ExB_poloidal_velocity", src)

    def test_flutter_keeps_five_halves(self):
        src = inspect.getsource(fluxes.radial_flutter_fluxes)
        self.assertIn("2.5", src)
        self.assertNotIn("EXB_ENERGY_COEFF", src)


class TestScriptsAndCaches(unittest.TestCase):
    def test_no_script_multiplies_an_ExB_flux_by_2_5_inline(self):
        for path in sorted(SCRIPTS.rglob("v3_*.py")):
            src = path.read_text()
            for bad in ("2.5 * P * v_ExB_pol", "2.5 * P * ExB_",
                        "2.5 * (Pe + Pi) * v_r"):
                self.assertNotIn(bad, src, f"{path.name}: {bad}")

    def test_caches_were_bumped_so_old_numbers_cannot_be_reloaded(self):
        import re
        for name, pat, floor in (
                ("v3_radial_flux.py", r'"si(\d+)"\)', 9),
                ("v3_radial_conservation.py", r'"si(\d+)",', 9),
                ("v3_poloidal_flux.py", r'"t(\d+)"\)', 3),
                ("v3_plasma_balance.py", r'"b(\d+)"\)', 10)):
            src = (next(SCRIPTS.rglob(name))).read_text()
            tags = [int(t) for t in re.findall(pat, src)]
            self.assertTrue(tags, f"{name}: no cache tag matched {pat}")
            self.assertGreaterEqual(max(tags), floor, name)

    def test_conservation_script_tests_the_identity_not_F_in_vs_F_out(self):
        src = (SCRIPTS / "balances/v3_radial_conservation.py").read_text()
        self.assertIn("CONSERVATION IDENTITY", src)
        self.assertIn("F_in_q_SP", src)
        self.assertIn("F_in_q_VgP", src)

        self.assertNotIn("these are equal:    F_in(x)  =  F_out(x)", src)

    def test_target_script_drops_y_guards_from_the_input_power(self):
        src = (SCRIPTS / "targets/v3_target_heatflux.py").read_text()
        self.assertIn("h3t.guard_rows(ds_avg)", src)


if __name__ == "__main__":
    unittest.main()
