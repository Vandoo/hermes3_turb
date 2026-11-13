"""Physical-time labels, cache units and legacy timestamp recovery."""
import ast
from pathlib import Path
import tempfile
import unittest

import netCDF4 as nc
import numpy as np
import xarray as xr

from hermes3_turb.profile_time import profile_times, time_label


class TestProfileTime(unittest.TestCase):
    def test_cached_seconds_and_irregular_sampling(self):
        t = profile_times({'t_s': [.010, .012, .030]}, 'unused.npz', expected_frames=3)
        self.assertEqual(time_label(t), 'Δt=20.0 ms')
        self.assertEqual(time_label(t, mode='range'), '10.0–30.0 ms')

    def test_single_snapshot(self):
        self.assertEqual(time_label(np.array([.012])), 't=12.0 ms, single snapshot')

    def test_missing_metadata_never_guesses_time(self):
        with self.assertWarnsRegex(UserWarning, 'averaging time unavailable'):
            t = profile_times({}, 'legacy.npz')
        self.assertIsNone(t)
        self.assertEqual(time_label(t), 'time unavailable')

    def test_mismatch_and_invalid_time(self):
        for cache in ({'t_s': [.01, .02]}, {'t_s': [.01, np.nan, .03]},
                      {'t_s': [.01, .02, .02]}):
            with self.subTest(cache=cache), self.assertWarns(UserWarning):
                self.assertIsNone(profile_times(cache, 'cache.npz', expected_frames=3))

    def squash(self, path, times):
        path.parent.mkdir(parents=True, exist_ok=True)
        with nc.Dataset(path, 'w') as ds:
            ds.createDimension('t', len(times))
            ds.createVariable('t_array', 'f8', ('t',))[:] = times
            ds.createVariable('Omega_ci', 'f8', ())[:] = 10000.

    def test_old_omp_recovers_seconds_and_deduplicates_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            self.squash(data/'1/boutdata.nc', [100, 200])
            self.squash(data/'2/boutdata.nc', [200, 400])
            path = data/'cache/v3_radial_flux_dirs1-2_yd_si8.npz'
            with self.assertWarnsRegex(UserWarning, 'recovered legacy cache times'):
                t = profile_times({'Ne_omp': [1.]}, path, expected_frames=3)
            np.testing.assert_allclose(t, [.01, .02, .04])
            self.assertEqual(time_label(t), 'Δt=30.0 ms')

    def test_old_target_prefers_guard_retaining_squash(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            self.squash(data/'1/boutdata.nc', [100, 200])
            self.squash(data/'1/boutdata_guards.nc', [300, 500])
            path = data/'cache/v3_target_heatflux_dirs1_i_omp_fxv8.npz'
            with self.assertWarnsRegex(UserWarning, 'recovered legacy cache times'):
                t = profile_times({}, path, expected_frames=2)
            np.testing.assert_allclose(t, [.03, .05])

    def test_empty_cached_times_fall_back_to_squash(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            self.squash(data/'1/boutdata_guards.nc', [300, 500])
            path = data/'cache/v3_target_heatflux_dirs1_i_omp_fxv8.npz'
            with self.assertWarnsRegex(UserWarning, 'recovered legacy cache times'):
                t = profile_times({'t_s': np.zeros(0)}, path, expected_frames=2)
            np.testing.assert_allclose(t, [.03, .05])

    def test_radial_cache_producer_converts_normalised_time_to_seconds(self):
        script = Path(__file__).resolve().parents[1]/'scripts/fluxes/v3_radial_flux.py'
        tree = ast.parse(script.read_text())
        keyword = next(n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == 't_s')
        ds = xr.Dataset(coords={'t': [100., 400.]}, attrs={'metadata': {'Omega_ci': 10000.}})
        t = eval(compile(ast.Expression(body=keyword.value), str(script), 'eval'),
                 {'np': np, 'ds': ds})
        np.testing.assert_allclose(t, [.01, .04])


if __name__ == '__main__':
    unittest.main()
