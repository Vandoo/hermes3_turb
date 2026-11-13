"""Release-layout and loader regressions with synthetic data, never real dumps."""
import ast
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
import warnings

import netCDF4 as nc
import numpy as np
import xarray as xr
import hermes3_turb as h3t

ROOT = Path(__file__).resolve().parents[1]


def function_from_script(relative, name, env):
    path = ROOT / "scripts" / relative
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[name]


class TestReleaseLoaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        (self.data / "BOUT.settings").write_text("current settings")
        self.parts = {}
        for folder, velocity in ((1, None), (2, 20.0)):
            path = self.data / str(folder)
            path.mkdir()
            (path / "boutdata.nc").touch()
            (path / "BOUT.inp").write_text("archived settings")
            fields = {name: (("t", "x", "y", "z"), np.full((1, 2, 2, 3), value))
                      for name, value in (("Pe", 2.0), ("Pi", 3.0), ("Ve", 5.0))}
            ds = xr.Dataset(fields, coords={"t": [float(folder)]},
                            attrs={"metadata": {"ny": 2, "ixseps1": 1},
                                   "options": {"folder": folder}})
            if velocity is not None:
                ds["Vi"] = xr.full_like(ds.Pe, velocity)
            ds = ds.assign_coords(dx=("t", [float(folder)]))
            self.parts[folder] = ds
        self.captured = []
        self.recovered = []
        self.loaded = []
        self.cache = {}

        def loader(path, **kwargs):
            self.loaded.append((path, kwargs))
            return self.parts[int(Path(path).parent.name)].copy(deep=True)

        def recover(ds, *args):
            self.recovered.append(ds)
            self.assertEqual(ds.options["folder"], 1)
            return ds.assign(Vi=xr.full_like(ds.Pe, 11.0))

        def targets(ds):
            self.captured.append(ds)
            return {}

        helper = SimpleNamespace(
            cache_path=lambda *a, **kw: self.data / "unused.npz",
            existing_dirs=lambda *a: [1, 2],
            detect_poloidal_dim=h3t.detect_poloidal_dim,
            dedup_restart_times=h3t.dedup_restart_times,
            save_cache=lambda path, **data: self.cache.update(data),
            load_cache=lambda path: self.cache)
        env = dict(os=os, np=np, xr=xr, warnings=warnings, h3t=helper,
                   ion_species="i", AA_ion=2.0, Z_ion=1.0,
                   fit_coordinate="omp", use_cache=False, targets_to_plot=None,
                   open_hermesdataset=loader, recover_target_ion_velocity=recover,
                   get_target_boundaries=lambda ds: ({}, True),
                   get_parallel_velocity=lambda ds, sp: ds[f"V{sp}"],
                   get_midplane_y_indices=lambda ds: {}, get_target_y_indices=targets)
        self.load_case = function_from_script("targets/target_profiles.py", "load_case", env)

    def test_archived_options_and_mixed_velocity_before_concat(self):
        self.load_case("example", str(self.data), range(1, 3), None)
        self.assertEqual(len(self.recovered), 1)
        for path, kwargs in self.loaded:
            self.assertEqual(kwargs["inputfilepath"], str(Path(path).parent / "BOUT.inp"))
        np.testing.assert_allclose(self.captured[0].Vi, 15.5)
        self.assertIn("dx", self.captured[0].coords)
        self.assertEqual(float(self.captured[0].dx), 1.0)
        np.testing.assert_array_equal(self.cache["t_s"], [1.0, 2.0])

    def test_missing_electron_velocity_refuses_momentum_guard_fallback(self):
        self.parts[1] = self.parts[1].drop_vars("Ve")
        with self.assertRaisesRegex(SystemExit, "Save Ve|save Ve"):
            self.load_case("example", str(self.data), range(1, 3), None)
        self.assertFalse(self.cache)

    def test_lcfs_monitor_uses_dump_guard_count_when_myg_differs(self):
        grid = self.data / "grid.nc"
        with nc.Dataset(grid, "w") as handle:
            handle.createVariable("ny", "i4")[...] = 8
        ds = xr.Dataset(coords={"x": [0], "y": np.arange(12)}, attrs={"metadata": {
            "ny": 8, "ny_inner": 4, "MYG": 2, "jyseps2_1": 2, "jyseps1_2": 5}})
        drop = function_from_script("run_diagnostics/lcfs_transport_monitor.py",
                                    "drop_y_guards", {"nc": nc, "h3t": h3t})
        result = drop(ds, str(grid), "y")
        np.testing.assert_array_equal(result.y, [1, 2, 3, 4, 7, 8, 9, 10])

    def test_single_folder_prefers_squash_and_its_archived_settings(self):
        script = ROOT / "scripts/targets/target_heatflux.py"
        tree = ast.parse(script.read_text())
        start = next(i for i, node in enumerate(tree.body)
                     if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "os.path.isdir(casepath)")
        stop = next(i for i in range(start, len(tree.body))
                    if isinstance(tree.body[i], ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_settings"
                            for t in tree.body[i].targets))
        env = {"os": os, "casepath": str(self.data / "1")}
        exec(compile(ast.Module(body=tree.body[start:stop+1], type_ignores=[]),
                     str(script), "exec"), env)
        self.assertEqual(env["datapath"], str(self.data / "1/boutdata.nc"))
        self.assertEqual(env["_settings"], str(self.data / "1/BOUT.inp"))


if __name__ == "__main__":
    unittest.main()
