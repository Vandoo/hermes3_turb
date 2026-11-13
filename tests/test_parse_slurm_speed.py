"""Job-local start-time reconstruction for throughput accounting."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_diagnostics/parse_slurm_speed.py"
spec = importlib.util.spec_from_file_location("parse_slurm_speed", SCRIPT)
speed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed)


class TestParseSlurmSpeed(unittest.TestCase):
    def parse(self, options, times=(1200, 1400), runtime="1 m 40 s"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slurm-123.out"
            path.write_text(options + "\n" + "\n".join(
                f"{time:.3e}  99  1.00e+01" for time in times
            ) + f"\nRun time : {runtime}\n")
            return speed.parse_file(str(path), omega_ci=1000)

    def test_omitted_initial_output_restores_interval_and_physical_speed(self):
        r = self.parse("Option dump_on_restart = false (data/BOUT.inp)\n"
                       "Solver running for 10 outputs with output timestep of 2e2")
        self.assertEqual(r["sim_start"], 1000)
        self.assertEqual(r["delta_sim"], 400)
        self.assertEqual(r["us_per_day"], 345600000)
        self.assertTrue(r["omitted_initial_interval"])
        self.assertEqual(r["job_id"], 123)

    def test_initial_output_included_does_not_add_interval(self):
        r = self.parse("Option dump_on_restart = true (default)\n"
                       "Option timestep = 200 (data/BOUT.inp)")
        self.assertEqual(r["delta_sim"], 200)
        self.assertFalse(r["omitted_initial_interval"])

    def test_legacy_log_keeps_conservative_start_without_guessing(self):
        r = self.parse("Option timestep = 200 (data/BOUT.inp)")
        self.assertEqual(r["sim_start"], 1200)
        self.assertIn("unknown", r["start_source"])

    def test_single_output_after_restart_is_valid(self):
        r = self.parse("Option dump_on_restart = false (default)\n"
                       "Option solver:output_step = 200 (default)", times=(1200,))
        self.assertEqual(r["delta_sim"], 200)

    def test_runtime_interval_overrides_legacy_option(self):
        r = self.parse("Option timestep = 100 (default)\n"
                       "Option dump_on_restart = 0 (default)\n"
                       "Solver running for 10 outputs with output timestep of 2.0e2")
        self.assertEqual(r["delta_sim"], 400)

    def test_missing_interval_does_not_silently_undercount(self):
        with self.assertRaisesRegex(ValueError, "output timestep is missing"):
            self.parse("Option dump_on_restart = false (default)")

    def test_nonpositive_interval_is_rejected(self):
        for interval in ("0", "-200"):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "invalid"):
                self.parse("Option dump_on_restart = false (default)\n"
                           f"Option timestep = {interval} (default)")

    def test_single_initial_snapshot_and_incomplete_jobs_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-positive"):
            self.parse("Option dump_on_restart = true (default)", times=(1200,))
        with self.assertRaisesRegex(ValueError, "No 'Run time'"):
            self.parse("", runtime="unfinished")


if __name__ == "__main__":
    unittest.main()
