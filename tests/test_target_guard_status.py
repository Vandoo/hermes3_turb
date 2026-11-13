"""Exercise the script's guard gate without importing xhermes or reading dumps."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
import warnings

import hermes3_turb as h3t


class TestTargetGuardStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'scripts/targets/v3_target_heatflux.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_report_guard_status')
        env = {'warnings': warnings}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        cls.report = staticmethod(env['_report_guard_status'])

    def test_guardless_data_stop_with_recovery_instructions(self):
        with self.assertRaisesRegex(SystemExit, 'use_cache=False'):
            self.report(False, 52, 52)

    def test_legacy_cache_warns_instead_of_claiming_validity(self):
        with self.assertWarnsRegex(UserWarning, 'UNVERIFIED'):
            self.report(None, cached=True)

    def test_new_cache_records_guard_check_without_claiming_current_file_check(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'target.npz'
            h3t.save_cache(path, has_yguards=True, ny_loaded=60, ny_domain=52)
            cache = h3t.load_cache(path)
            output = io.StringIO()
            with redirect_stdout(output):
                self.report(cache['has_yguards'], cache['ny_loaded'], cache['ny_domain'],
                            cached=True)
            self.assertIn('recorded when cache was computed', output.getvalue())
            self.assertIn('loaded y=60, domain ny=52', output.getvalue())


if __name__ == '__main__':
    unittest.main()
