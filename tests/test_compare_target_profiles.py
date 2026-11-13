"""Cache-only target comparisons using synthetic heat flux, never simulation dumps."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

import hermes3_turb as h3t

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/targets/compare_target_profiles.py"
spec = importlib.util.spec_from_file_location("compare_target_profiles", SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class TestTargetComparison(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(plt.close, "all")

    def cache(self, name="case", coord="OMP-mapped (ψ)", frames=3, targets=None):
        targets = targets or ["outer_lower"]
        s = np.arange(-4., 5.)
        q = np.array([90., 80., 1., 2., 4., 2., 1., 80., 90.])
        flat = dict(target_names=targets, target_scalars={
            t: dict(coord_label=coord, P_total_kW=123.) for t in targets})
        flat['t_s'] = .010 + np.arange(frames or 3) * .002
        for target in targets:
            prefix = f"tgt_{target}__"
            flat[prefix + "s_mm"] = s
            flat[prefix + "q_MWm2"] = q * 7  # deliberately NOT the per-frame mean
            if frames:
                flat[prefix + "q_t_MWm2"] = np.stack([q * (i + 1) for i in range(frames)])
        path = Path(self.tmp.name) / f"{name}.npz"
        h3t.save_cache(path, **flat)
        return path

    def test_snapshot_mean_sample_sd_and_guard_removal(self):
        p = comparison.read_profiles(self.cache())["outer_lower"]
        np.testing.assert_allclose(p["s"], [-2, -1, 0, 1, 2])
        np.testing.assert_allclose(p["q"], np.array([1, 2, 4, 2, 1]) * 2)
        np.testing.assert_allclose(p["sd"], [1, 2, 4, 2, 1])
        self.assertEqual(p["nt"], 3)

    def test_zero_guards_and_single_frame(self):
        p = comparison.read_profiles(self.cache(frames=1), guards=0)["outer_lower"]
        self.assertEqual(p["s"].size, 9)
        self.assertIsNone(p["sd"])

    def test_legacy_mean_warns_and_has_no_band(self):
        with self.assertWarnsRegex(UserWarning, "no per-frame"):
            p = comparison.read_profiles(self.cache(frames=0))["outer_lower"]
        self.assertEqual(p["nt"], 0)
        self.assertIsNone(p["sd"])
        self.assertEqual(p["q"].max(), 28)

    def test_coordinate_mismatch_rejected(self):
        a = self.cache("a")
        b = self.cache("b", coord="target (⊥ flux surf)")
        with self.assertRaisesRegex(ValueError, "incompatible coordinates"):
            comparison.compare([("a", a), ("b", b)])

    def test_nonfinite_frame_rejected(self):
        path = self.cache()
        data = h3t.load_cache(path)
        data["tgt_outer_lower__q_t_MWm2"][0, 4] = np.nan
        h3t.save_cache(path, **data)
        with self.assertRaisesRegex(ValueError, "nonfinite per-frame"):
            comparison.read_profiles(path)

    def test_window_normalization_scales_band(self):
        fig = comparison.compare([("case", self.cache())], normalization="peak")
        ax = fig.axes[0]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [.25, .5, 1, .5, .25])
        self.assertEqual(ax.lines[0].get_label(), 'case (Δt=4.0 ms)')
        # At s=0 the normalized sample sigma is 4/8 = 0.5.
        vertices = ax.collections[0].get_paths()[0].vertices
        np.testing.assert_allclose(sorted(set(vertices[vertices[:, 0] == 0, 1])), [.5, 1.5])

    def test_cli_two_cases_missing_target_and_png(self):
        a = self.cache("a", targets=["outer_lower", "inner_lower"])
        b = self.cache("b")
        png = Path(self.tmp.name) / "comparison.png"
        result = subprocess.run([
            sys.executable, str(SCRIPT), "--case", "coarse", str(a), "--case", "fine", str(b),
            "--no-show", "--save", str(png)], capture_output=True, text=True,
            # the package need not be pip-installed for the subprocess to import it
            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                filter(None, [str(SCRIPT.parents[2]), os.environ.get("PYTHONPATH")]))})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("inner_lower absent", result.stderr)
        self.assertIn("coarse", result.stdout)
        self.assertGreater(png.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
