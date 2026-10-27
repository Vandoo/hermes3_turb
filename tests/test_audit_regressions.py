"""Synthetic regressions for the 2026-10-27 review; no simulation files needed."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import xarray as xr

import hermes3_turb as h3t


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('species,aa', [('i', 2.), ('e', h3t.SPECIES_AA['e'])])
@pytest.mark.parametrize('si', [False, True])
def test_velocity_from_momentum_in_normalised_and_si_units(species, aa, si):
    density, velocity = 1e19 if si else 3., 1000. if si else 2.
    mp, nscale, vscale = 1.67e-27, 1e19, 2e4
    mass = aa * mp if si else aa
    units_type = 'SI' if si else 'hermes'
    ds = xr.Dataset({
        'N'+species: xr.DataArray(density, attrs={
            'units_type': units_type, 'conversion': nscale}),
        'NV'+species: xr.DataArray(mass*density*velocity, attrs={
            'units_type': units_type, 'conversion': mp*nscale*vscale}),
    }, attrs={'metadata': {'Nnorm': nscale, 'Cs0': vscale}})
    assert float(h3t.get_parallel_velocity(ds, species)) == pytest.approx(velocity)


def test_velocity_si_fallback_without_conversion_metadata():
    mp = 1.67262192369e-27
    ds = xr.Dataset({
        'NHe': xr.DataArray(1e19, attrs={'units_type': 'SI'}),
        'NVHe': xr.DataArray(4*mp*1e19*1200, attrs={'units_type': 'SI'}),
    })
    assert float(h3t.get_parallel_velocity(ds, 'He', AA=4)) == pytest.approx(1200)


def test_velocity_preserves_unannotated_normalised_inputs_and_direct_velocity():
    ds = xr.Dataset({'Ni': xr.DataArray(3.), 'NVi': xr.DataArray(12.)})
    assert float(h3t.get_parallel_velocity(ds, 'i')) == 2.
    ds['Vi'] = xr.DataArray(17.)
    assert h3t.get_parallel_velocity(ds, 'i').identical(ds.Vi)


def test_velocity_rejects_mixed_units():
    ds = xr.Dataset({'Ni': xr.DataArray(3.),
                     'NVi': xr.DataArray(12., attrs={'units_type': 'SI'})})
    with pytest.raises(ValueError, match='matching units_type'):
        h3t.get_parallel_velocity(ds, 'i')


def test_velocity_rejects_invalid_conversion_scales():
    ds = xr.Dataset({
        'Ni': xr.DataArray(1e19, attrs={'units_type': 'SI', 'conversion': 1e19}),
        'NVi': xr.DataArray(1., attrs={'units_type': 'SI', 'conversion': 1.}),
    }, attrs={'metadata': {'Cs0': 0.}})
    with pytest.raises(ValueError, match='conversion factors'):
        h3t.get_parallel_velocity(ds, 'i')


def test_cache_preserves_existing_names_when_folders_become_available(tmp_path):
    for number in (1, 3):
        folder = tmp_path/str(number)
        folder.mkdir()
        (folder/'boutdata.nc').touch()
    before = h3t.existing_dirs(str(tmp_path), [1, 2, 3])
    old_cache = h3t.cache_path('flux', before, cache_dir=tmp_path/'cache')
    h3t.save_cache(old_cache, value=42)
    (tmp_path/'2').mkdir()
    (tmp_path/'2/boutdata.nc').touch()
    after = h3t.existing_dirs(str(tmp_path), [1, 2, 3])
    new_cache = h3t.cache_path('flux', after, cache_dir=tmp_path/'cache')
    # Retain the historical name for cross-machine and hand-written paths.
    # After filling a gap, the user must set use_cache=False to recompute.
    assert old_cache == new_cache
    assert new_cache.name == 'flux_dirs1-3.npz'
    assert h3t.load_cache(new_cache)['value'] == 42


def test_cache_names_are_portable_and_accept_numpy_and_nan_extras(tmp_path):
    remote = h3t.cache_path('v3_radial_flux', range(146, 148), 'yd', 'si8',
                           cache_dir=tmp_path/'pitagora')
    local = h3t.cache_path('v3_radial_flux', range(146, 148), 'yd', 'si8',
                          cache_dir=tmp_path/'sshfs')
    assert remote.name == local.name == 'v3_radial_flux_dirs146-147_yd_si8.npz'
    numeric = h3t.cache_path('test', [1], np.int64(2), np.float64(np.nan),
                            cache_dir=tmp_path)
    assert numeric.name == 'test_dirs1_2_nan.npz'


@pytest.fixture
def target_dataset():
    # Identical geometry at all poloidal positions. Pressure and velocity
    # are correlated in time, so <P V> = 5 whereas <P><V> = 4.
    shape = (2, 6, 4, 8)
    p = np.broadcast_to(np.array([1., 3.])[:, None, None, None], shape).copy()
    ds = xr.Dataset({name: (('t', 'x', 'theta', 'zeta'), p.copy())
                     for name in ('Pe', 'Pi', 'Ve', 'Vi')},
                    attrs={'metadata': {'MXG': 1, 'MYG': 0, 'ny': 4}})
    for name in ('dx', 'g_11', 'g_22', 'g_33', 'g11', 'Bxy', 'Bpxy'):
        ds[name] = (('x', 'theta'), np.ones((6, 4)))
    for name in ('g_13', 'g_23'):
        ds[name] = (('x', 'theta'), np.zeros((6, 4)))
    return ds


@pytest.mark.parametrize('mode,expected_flux', [('per_snapshot', 35.), ('avg_fields', 28.)])
def test_target_script_integrates_the_selected_profile(target_dataset, mode, expected_flux):
    ds = target_dataset

    def q_for(d):
        return h3t.target_heat_flux_density(d, d.Pe, d.Pi, d.Ve, d.Vi,
                                          bohm_cap_v=False)[1]

    average = ds.mean('t', keep_attrs=True)
    q_avg = q_for(average)
    q_snap = (q_for(ds.isel(t=0)) + q_for(ds.isel(t=1))) / 2
    # Execute the script's actual selection/integration statements without
    # running its top-level data loading or requiring the unavailable xhermes.
    tree = ast.parse((ROOT/'scripts/targets/v3_target_heatflux.py').read_text())
    compute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_compute')
    names = {'q_MWm2', 'q_selected', 'P_total_kW'}
    nodes = [n for n in ast.walk(compute) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    env = dict(q_snap_MWm2=q_snap.values*1e-6, q_avg_MWm2=q_avg.values*1e-6,
               target_mean=mode, ds_avg=average, q_avg=q_avg, y_idx=3,
               total_target_power=h3t.total_target_power)
    for node in sorted(nodes, key=lambda n: n.lineno):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<target script>', 'exec'), env)
    np.testing.assert_allclose(env['q_MWm2'], expected_flux*1e-6)
    # Four physical x cells, unit area per radian, full torus.
    assert env['P_total_kW'] == pytest.approx(4*expected_flux*2*np.pi/1000)


@pytest.mark.parametrize('repeated_geometry', [False, True])
def test_target_power_repeated_geometry_and_guard_exclusion(target_dataset, repeated_geometry):
    ds = target_dataset.mean('t', keep_attrs=True)
    if repeated_geometry:
        for name in ('dx', 'g_11', 'g_33', 'g_13'):
            ds[name] = ds[name].expand_dims(zeta=np.arange(8))
    q = xr.DataArray([1000., 10., 10., 10., 10., 1000.], dims='x')
    assert h3t.total_target_power(ds, q, 3) == pytest.approx(40*2*np.pi)
    assert h3t.total_target_power(ds, q, 3, exclude_xguards=False) == pytest.approx(2040*2*np.pi)
    q_times = xr.concat([q, 3*q], dim='t')
    assert h3t.total_target_power(ds, q_times, 3) == pytest.approx(80*2*np.pi)


def test_conduction_uses_parallel_length_metric():
    temperature = xr.DataArray(np.ones((2, 3, 8)), dims=('x', 'theta', 'zeta'))
    ds = xr.Dataset({'g_22': (('x', 'theta'), np.full((2, 3), 4.))},
                    coords={'zeta': np.arange(8)})
    # Isolate xBOUT's coordinate derivative: dT/dtheta=1 eV/rad.
    # With sqrt(g_22)=2 m/rad and kappa=2 W/(m eV), q=-1 W/m².
    with patch.object(xr.DataArray, 'bout', new_callable=PropertyMock, create=True) as accessor:
        accessor.return_value = SimpleNamespace(ddy=lambda: xr.ones_like(temperature))
        result = h3t.parallel_conductive_heat_flux(ds, temperature, 2.)
    np.testing.assert_allclose(result, -1.)


@pytest.fixture
def storage_functions():
    tree = ast.parse((ROOT/'scripts/balances/v3_balance_diagnose.py').read_text())
    names = {'inventory_rate', 'closure_percent', 'closure_label', 'print_row'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    env = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<storage functions>', 'exec'), env)
    return env


@pytest.mark.parametrize('times,inventory', [([1.], [2.]), ([], []),
    ([1., 1.], [2., 3.]), ([2., 1.], [2., 3.]),
    ([0., np.nan], [2., 3.]), ([0., 1.], [2., np.nan])])
def test_invalid_storage_interval_is_unavailable(storage_functions, times, inventory, capsys):
    env = storage_functions
    rate = env['inventory_rate'](times, inventory)
    assert np.isnan(rate)
    assert np.isnan(env['closure_percent'](10., 4., rate))
    env['print_row']('particle', 10., 4., rate)
    assert 'closure=unavailable' in capsys.readouterr().out


def test_valid_storage_rate_and_zero_source(storage_functions):
    env = storage_functions
    assert env['inventory_rate']([0., 1., 2.], [2., 5., 8.]) == pytest.approx(3.)
    assert env['closure_percent'](10., 7., 3.) == pytest.approx(100.)
    assert env['closure_label'](100.) == '100.0%'
    assert env['closure_label'](env['closure_percent'](0., 7., 3.)) == 'unavailable'


def test_closure_plot_title_handles_single_frame_and_missing_source(storage_functions):
    env = dict(storage_functions, S_part=0., S_eng=0., F_part=7., F_eng=8.,
               dN_dt=float('nan'), dU_dt=float('nan'), species='e')
    titles = []
    env['fig'] = SimpleNamespace(suptitle=titles.append)
    tree = ast.parse((ROOT/'scripts/balances/v3_balance_diagnose.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == 'suptitle')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<closure plot>', 'exec'), env)
    assert titles[0].count('unavailable') == 2
