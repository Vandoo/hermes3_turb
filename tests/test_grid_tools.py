"""Check source budgets, topology masks, and copied-grid coefficient writes."""
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import netCDF4 as nc
import numpy as np

from hermes3_turb.plasma_source_reader import PlasmaSourceReader


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts/grid_tools'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebuild = load_script('rebuild_src')
inject = load_script('inject_anomalous')


def make_grid(path, yguards=0, nz=None, single_null=False):
    nx, ny = 12, 16
    loaded = ny+(2 if single_null else 4)*yguards
    with nc.Dataset(path, 'w') as grid:
        grid.createDimension('x', nx)
        grid.createDimension('y', loaded)
        if nz is not None:
            grid.createDimension('z', nz)
        metadata = dict(nx=nx, ny=ny, ny_inner=8, y_boundary_guards=yguards,
                        ixseps1=5, ixseps2=5, jyseps1_1=1, jyseps2_1=5,
                        jyseps1_2=5 if single_null else 9, jyseps2_2=13)
        for name, value in metadata.items():
            grid.createVariable(name, 'i4')[:] = value
        x, y = np.arange(nx)[:, None], np.arange(loaded)[None, :]
        fields = dict(J=1+x*.1+y*.01, dx=np.broadcast_to(.2+x*.01, (nx, loaded)),
                      dy=np.full((nx, loaded), .1), Bpxy=np.ones((nx, loaded)))
        fields['hthe'] = fields['J'].copy()
        fields['unrelated'] = 3+np.broadcast_to(x, (nx, loaded))
        for name, values in fields.items():
            grid.createVariable(name, 'f8', ('x', 'y'))[:] = values


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GridToolTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.source = self.directory/'grid.nc'
        self.output = self.directory/'out.nc'
        make_grid(self.source)
        self.before = digest(self.source)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def independent_budgets(self, path):
        with nc.Dataset(path) as grid:
            # Independent geometry route; no use of the builder's integral helper.
            volume = np.asarray(grid['hthe'][:])/np.asarray(grid['Bpxy'][:])
            volume *= np.asarray(grid['dx'][:])*np.asarray(grid['dy'][:])
            particles = np.asarray(grid['Ne_src'][:])
            rows = list(range(2, 6))+list(range(10, 14))
            core = (particles[2:-2, rows]*volume[2:-2, rows]).sum()*2*np.pi
            total = (particles[2:-2]*volume[2:-2]).sum()*2*np.pi
            power = sum((np.asarray(grid[name][:])[2:-2]*volume[2:-2]).sum()*3*np.pi
                        for name in ('Pe_src', 'Pi_src'))
            np.testing.assert_allclose(grid['Pe_src'][:], grid['Pi_src'][:])
            np.testing.assert_allclose(grid['unrelated'][:],
                                       np.broadcast_to(3+np.arange(12)[:, None], (12, 16)))
        return float(core), float(total-core), float(power)

    def test_corrected_core_mask_has_no_particle_leak_on_leg_row(self):
        reader = PlasmaSourceReader(str(self.source))
        reader.apply_analytical_sources(S_n0=1., S_E0=1., apply_background_sources=False)
        np.testing.assert_array_equal(reader.S[:, 9], 0.)
        noncore = [0, 1, 6, 7, 8, 9, 14, 15]
        np.testing.assert_array_equal(reader.S[:, noncore], 0.)
        np.testing.assert_array_equal(reader.S[:2], 0.)
        np.testing.assert_array_equal(reader.S[-2:], 0.)
        self.assertGreater(float(reader.S[2:-2, 10:14].sum()), 0.)

    def test_core_only_source_matches_particle_and_energy_budgets(self):
        rebuild.rebuild_sources(self.source, self.output, mode='sxd', flux=1e20, heating=3000.)
        core, target, power = self.independent_budgets(self.output)
        self.assertAlmostEqual(core/1e20, 1., places=12)
        self.assertLess(abs(target)/core, 1e-12)
        self.assertAlmostEqual(power, 6000., places=8)
        self.assertEqual(digest(self.source), self.before)

    def test_target_source_has_requested_fraction_and_matched_energy(self):
        summary = rebuild.rebuild_sources(self.source, self.output, mode='cd', flux=1e20,
                                          heating=3000., target_frac=.02, T_inj=5.)
        core, target, power = self.independent_budgets(self.output)
        self.assertAlmostEqual(core/1e20, 1., places=12)
        self.assertAlmostEqual(target/(core+target), .02, places=12)
        self.assertAlmostEqual(power, 6000., places=8)
        self.assertAlmostEqual(summary['S_ET']/(summary['S_nT']*rebuild.E_CHARGE), 5.)
        self.assertEqual(digest(self.source), self.before)

    def test_match_preserves_existing_core_input_and_total_power(self):
        rebuild.rebuild_sources(self.source, self.output, mode='cd', flux=2e20,
                                 heating=4500., target_frac=.03)
        matched = self.directory/'matched.nc'
        rebuild.rebuild_sources(self.source, matched, match=self.output, mode='sxd')
        core, target, power = self.independent_budgets(matched)
        self.assertAlmostEqual(core/2e20, 1., places=12)
        self.assertLess(abs(target)/core, 1e-12)
        self.assertAlmostEqual(power, 9000., places=8)

    def test_sources_can_update_existing_source_variables(self):
        rebuild.rebuild_sources(self.source, self.output, flux=1e20, heating=3000.)
        second = self.directory/'second.nc'
        rebuild.rebuild_sources(self.output, second, mode='cd', flux=3e20,
                                 heating=2500., target_frac=.01)
        core, target, power = self.independent_budgets(second)
        self.assertAlmostEqual(core/3e20, 1., places=12)
        self.assertAlmostEqual(target/(core+target), .01, places=12)
        self.assertAlmostEqual(power, 5000., places=8)

    def test_source_invalid_parameters_and_impossible_budget_do_not_write(self):
        for kwargs in (dict(flux=0), dict(target_frac=1), dict(peak=0), dict(heating=-1),
                       dict(mode='cd', heating=0, target_frac=.1, T_inj=5)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rebuild.rebuild_sources(self.source, self.output, **kwargs)
            self.assertFalse(self.output.exists())
        self.assertEqual(digest(self.source), self.before)

    def test_source_rejects_guarded_grid_instead_of_misplacing_sources(self):
        guarded = self.directory/'guarded.nc'
        make_grid(guarded, yguards=2)
        with self.assertRaisesRegex(ValueError, 'without y guards'):
            rebuild.rebuild_sources(guarded, self.output)
        self.assertFalse(self.output.exists())

    def test_anomalous_profiles_include_both_upper_private_flux_regions(self):
        inject.inject_anomalous(self.source, self.output, species=['e', 'i'])
        with nc.Dataset(self.output) as grid:
            for sp in ('e', 'i'):
                self.assertEqual(grid['D_'+sp].dimensions, ('x', 'y'))
                self.assertEqual(grid['nu_'+sp].dimensions, ('x', 'y', 'z'))
                np.testing.assert_array_equal(grid['D_'+sp][:], 4.)
                np.testing.assert_array_equal(grid['chi_'+sp][2:5, 2:6], 10.)
                np.testing.assert_array_equal(grid['chi_'+sp][5:, :], 4.)
                np.testing.assert_array_equal(grid['nu_'+sp][2:5, 6:10, 0], 0.)
                np.testing.assert_array_equal(grid['nu_'+sp][2:5, 2:6, 0], 2.)
                self.assertEqual(grid['D_'+sp].units, 'm^2 s^-1')
            np.testing.assert_allclose(grid['dv'][:],
                2*np.pi*grid['J'][:]*grid['dx'][:]*grid['dy'][:])
        self.assertEqual(digest(self.source), self.before)

    def test_anomalous_fills_every_existing_z_plane_and_updates_variables(self):
        source = self.directory/'zgrid.nc'
        make_grid(source, nz=5)
        inject.inject_anomalous(source, self.output)
        second = self.directory/'second.nc'
        inject.inject_anomalous(self.output, second, D=8., chi=7., chi_core=12., nu=3.)
        with nc.Dataset(second) as grid:
            for sp in ('e', 'd+'):
                np.testing.assert_array_equal(grid['D_'+sp][:], 8.)
                self.assertEqual(grid['nu_'+sp].shape, (12, 16, 5))
                self.assertFalse(np.ma.getmaskarray(grid['nu_'+sp][:]).any())
                for z in range(5):
                    np.testing.assert_array_equal(grid['nu_'+sp][..., z], grid['nu_'+sp][..., 0])

    def test_anomalous_maps_double_null_y_guards(self):
        guarded = self.directory/'guarded.nc'
        make_grid(guarded, yguards=2)
        inject.inject_anomalous(guarded, self.output)
        with nc.Dataset(self.output) as grid:
            row = np.asarray(grid['chi_e'][3, :])
            np.testing.assert_array_equal(np.flatnonzero(row == 10.),
                                           list(range(4, 8))+list(range(16, 20)))

    def test_anomalous_supports_single_null_grid(self):
        source = self.directory/'single.nc'
        make_grid(source, single_null=True)
        inject.inject_anomalous(source, self.output)
        with nc.Dataset(self.output) as grid:
            np.testing.assert_array_equal(np.flatnonzero(grid['chi_e'][3, :] == 10.),
                                           np.arange(2, 14))

    def test_bad_coefficients_and_dimensions_leave_input_and_output_untouched(self):
        for kwargs in (dict(D=-1), dict(chi=np.nan), dict(species=['e', 'e'])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                inject.inject_anomalous(self.source, self.output, **kwargs)
            self.assertFalse(self.output.exists())
        with nc.Dataset(self.source, 'a') as grid:
            grid.createVariable('D_e', 'f8', ('x',))
        before = digest(self.source)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            inject.inject_anomalous(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(digest(self.source), before)

    def test_both_tools_require_distinct_output_and_explicit_overwrite(self):
        for writer in (rebuild.rebuild_sources, inject.inject_anomalous):
            with self.subTest(writer=writer.__name__):
                with self.assertRaisesRegex(ValueError, 'different'):
                    writer(self.source, self.source)
                self.output.write_bytes(b'existing')
                with self.assertRaises(FileExistsError):
                    writer(self.source, self.output)
                self.assertEqual(self.output.read_bytes(), b'existing')
                writer(self.source, self.output, overwrite=True)
                self.assertEqual(digest(self.source), self.before)
                self.output.unlink()

    def test_integer_grid_fields_are_rejected_instead_of_truncating_profiles(self):
        for writer, name in ((rebuild.rebuild_sources, 'Pi_src'),
                             (inject.inject_anomalous, 'D_e')):
            source = self.directory/(name+'.nc')
            make_grid(source)
            with nc.Dataset(source, 'a') as grid:
                grid.createVariable(name, 'i4', ('x', 'y'))[:] = 0
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'floating-point'):
                writer(source, self.output)
            self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
