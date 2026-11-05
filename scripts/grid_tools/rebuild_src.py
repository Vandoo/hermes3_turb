#!/usr/bin/env python3
r"""Build calibrated core and near-target plasma sources in a new grid.

Example (all powers are per species):
  python rebuild_src.py --in grid.nc --out grid_src.nc --mode cd \
      --flux 1e22 --heating 300e3 --target-frac 0.02 --T-inj 5

The sxd mode sets the near-target contribution to zero. The cd mode sets its
fraction of the total particle input. Names describe source configurations;
both work with guardless double-null CD or SXD grids. Writes Ne_src [m^-3/s]
and Pe_src=Pi_src [Pa/s]. Total heating is 3/2 times the pressure-source integral.
Leave species source options unset in BOUT.inp, and set source_only_in_core=false
when retaining the near-target source. Requires an installed hermes3_turb.
"""
import argparse
import os
from pathlib import Path
import tempfile

import netCDF4 as nc
import numpy as np

from hermes3_turb.plasma_source_reader import PlasmaSourceReader

E_CHARGE = 1.602e-19


def rowsets(path):
    """Read the guardless double-null topology used by this source builder."""
    with nc.Dataset(path) as grid:
        scalar = lambda name: int(np.asarray(grid[name][:]).item())
        nx, ny, ny_inner = (scalar(k) for k in ('nx', 'ny', 'ny_inner'))
        jy = tuple(scalar(k) for k in ('jyseps1_1', 'jyseps2_1',
                                      'jyseps1_2', 'jyseps2_2'))
        guards = scalar('y_boundary_guards') if 'y_boundary_guards' in grid.variables else 0
        if guards or grid['J'].shape != (nx, ny):
            raise ValueError('Source calibration currently requires a grid without y guards')
        if nx <= 4 or not (0 <= jy[0] < jy[1] < ny_inner <= jy[2] < jy[3] < ny):
            raise ValueError('Source calibration requires a valid double-null grid topology')
        for name in ('Ne_src', 'Pe_src', 'Pi_src'):
            if name in grid.variables and grid[name].dimensions != ('x', 'y'):
                raise ValueError(f'{name} must have dimensions (x, y)')
            if name in grid.variables and grid[name].dtype.kind != 'f':
                raise ValueError(f'{name} must use a floating-point datatype')
    core = list(range(jy[0]+1, jy[1]+1)) + list(range(jy[2]+1, jy[3]+1))
    noncore = [y for y in range(ny) if y not in set(core)]
    return nx, ny, ny_inner, jy, core, noncore


def rebuild_sources(grid_in, grid_out, mode='sxd', flux=1e22, heating=300e3,
                    target_frac=0.02, T_inj=5., peak=5., match=None, overwrite=False):
    """Calibrate particle/pressure sources, then copy and write the output grid."""
    source, output = Path(grid_in).expanduser().resolve(), Path(grid_out).expanduser().resolve()
    if source == output:
        raise ValueError('Input and output grids must be different files')
    if output.exists() and not overwrite:
        raise FileExistsError(f'{output} exists; use --overwrite to replace it')
    if mode not in ('sxd', 'cd'):
        raise ValueError('mode must be sxd or cd')
    if not np.isfinite([flux, heating, target_frac, T_inj, peak]).all():
        raise ValueError('Source parameters must be finite')
    if flux <= 0 or heating < 0 or T_inj < 0 or peak <= 0 or not 0 <= target_frac < 1:
        raise ValueError('Need flux>0, heating>=0, T-inj>=0, peak>0 and 0<=target-frac<1')
    nx, ny, ny_inner, jy, core_rows, noncore_rows = rowsets(source)
    interior = slice(2, nx-2)

    def split(field, volume):
        return tuple(float((field[interior, rows]*volume[interior, rows]).sum())*2*np.pi
                     for rows in (core_rows, noncore_rows))

    def profiles(S_n0, S_nT, S_E0=0., S_ET=0.):
        reader = PlasmaSourceReader(str(source), MXG=2, MYG=0,
                                    density_var='Ne_src', energy_var='Pe_src')
        reader.apply_analytical_sources(
            S_n0=S_n0, S_nT=S_nT, S_E0=S_E0, S_ET=S_ET,
            y_peaking=peak, x_Sn=0.05, x_SE=0.05, w_Sn=0.05, w_SE=0.05,
            S_E_baseline=0., apply_core_sources=True)
        return reader

    core = profiles(1., 0.)
    target = profiles(0., 1.)
    c_c, c_l = split(core.S, core.elem_vol)
    t_c, t_l = split(target.S, target.elem_vol)
    if not np.isfinite([c_c, c_l, t_c, t_l]).all() or c_c <= 0 or t_l <= 0:
        raise ValueError('Source calibration needs finite positive core/target integrals')
    if abs(c_l) > 1e-9*c_c:
        raise ValueError('Core source leaked onto non-core poloidal rows')

    core_input, power_input = flux, 2*heating
    if match is not None:
        match = Path(match).expanduser().resolve()
        if rowsets(match)[:4] != (nx, ny, ny_inner, jy):
            raise ValueError('Matched grid must have the same dimensions and topology')
        with nc.Dataset(match) as old:
            volume = np.asarray(old['J'][:])*np.asarray(old['dx'][:])*np.asarray(old['dy'][:])
            core_input, _ = split(np.asarray(old['Ne_src'][:]), volume)
            power_input = sum(float((np.asarray(old[v][:])[interior]*volume[interior]).sum())
                              *2*np.pi*1.5 for v in ('Pe_src', 'Pi_src'))
        if not np.isfinite([core_input, power_input]).all() or core_input <= 0 or power_input < 0:
            raise ValueError('Matched grid has invalid particle/heating integrals')

    S_nT = 0. if mode == 'sxd' else target_frac*core_input/(t_l*(1-target_frac))
    S_n0 = (core_input-S_nT*t_c)/c_c
    S_ET = S_nT*T_inj*E_CHARGE
    energy_target = profiles(0., 0., S_ET=S_ET)
    energy_core = profiles(0., 0., S_E0=1.)

    def power(reader):
        # Pa/s -> internal-energy source: 3/2, times two identical species.
        return float((reader.S_E[interior]*reader.elem_vol[interior]).sum())*2*np.pi*1.5*2

    e_c, e_t = power(energy_core), power(energy_target)
    if not np.isfinite([e_c, e_t]).all() or e_c <= 0:
        raise ValueError('Pressure-source calibration has an invalid integral')
    S_E0 = (power_input-e_t)/e_c
    if S_n0 < 0 or S_E0 < 0:
        raise ValueError('Requested target contribution exceeds particle/heating budget')
    reader = profiles(S_n0, S_nT, S_E0=S_E0, S_ET=S_ET)
    core_rate, target_rate = split(reader.S, reader.elem_vol)
    summary = dict(core_particles_s=core_rate, target_particles_s=target_rate,
                   target_fraction=target_rate/(core_rate+target_rate),
                   total_power_W=power(reader), S_n0=S_n0, S_nT=S_nT, S_E0=S_E0, S_ET=S_ET)
    expected_frac = target_frac if mode == 'cd' else 0.
    if not (np.isclose(core_rate, core_input, rtol=1e-10)
            and np.isclose(summary['target_fraction'], expected_frac, rtol=1e-10, atol=1e-12)
            and np.isclose(summary['total_power_W'], power_input, rtol=1e-10, atol=1e-8)):
        raise ValueError('Calibrated source integrals do not match requested budgets')

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix='.nc', dir=output.parent)
    os.close(fd)
    try:
        reader.save_sources_to_grid(temporary)
        with nc.Dataset(temporary, 'a') as grid:
            if 'Pi_src' not in grid.variables:
                grid.createVariable('Pi_src', 'f8', ('x', 'y'))
            grid['Pi_src'][:] = reader.S_E
            for name, units, description in (
                    ('Ne_src', 'm^-3 s^-1', 'Electron particle source'),
                    ('Pe_src', 'Pa s^-1', 'Electron pressure source'),
                    ('Pi_src', 'Pa s^-1', 'Ion pressure source')):
                grid[name].setncatts(dict(units=units, long_name=description, bout_type='Field2D'))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'Wrote {output}: core={core_rate:.6e}/s, target={target_rate:.6e}/s, '
          f'fraction={summary["target_fraction"]:.3%}, e+i power={power(reader)/1e3:.6g} kW')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in', dest='grid_in', required=True)
    parser.add_argument('--out', dest='grid_out', required=True)
    parser.add_argument('--mode', choices=('sxd', 'cd'), required=True)
    parser.add_argument('--flux', type=float, default=1e22, help='core-row particle input [s^-1]')
    parser.add_argument('--heating', type=float, default=300e3, help='total heating per species [W]')
    parser.add_argument('--match', help='match an existing grid’s core-row input and total heating')
    parser.add_argument('--target-frac', dest='target_frac', type=float, default=.02)
    parser.add_argument('--T-inj', dest='T_inj', type=float, default=5., help='target injection temperature [eV]')
    parser.add_argument('--peak', type=float, default=5.)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    try:
        rebuild_sources(**vars(args))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
