#!/usr/bin/env python3
r"""Write physical anomalous diffusivities to a copy of a BOUT++ grid.

  python inject_anomalous.py --in grid.nc --out grid_diff.nc --species e d+ \
      --D 4 --chi 4 --chi-core 10 --nu 2 --nu-pfr 0

D and chi are Field2D; nu is Field3D (constant in z). All coefficients are in
m^2/s. Core chi applies inside the inner separatrix on the closed poloidal
bands. PFR nu applies inside that separatrix outside those bands. Other cells
use chi/nu. Also writes dv=2*pi*J*dx*dy, the full-torus cell volume [m^3].
Hermes-3 species anomalous_D/chi/nu options override the corresponding grid
fields; leave those options unset to use these profiles.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import tempfile

import netCDF4 as nc
import numpy as np


def inject_anomalous(grid_in, grid_out, species=('e', 'd+'), D=4., chi=4.,
                     chi_core=10., nu=2., nu_pfr=0., overwrite=False):
    """Write the configured coefficients, retaining all unrelated grid fields."""
    source, output = Path(grid_in).expanduser().resolve(), Path(grid_out).expanduser().resolve()
    if source == output:
        raise ValueError('Input and output grids must be different files')
    if output.exists() and not overwrite:
        raise FileExistsError(f'{output} exists; use --overwrite to replace it')
    species = list(species)
    if (not species or len(set(species)) != len(species)
            or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_+\-]*', s) for s in species)):
        raise ValueError('Specify distinct species labels, e.g. e i or e d+')
    coefficients = np.asarray([D, chi, chi_core, nu, nu_pfr], dtype=float)
    if not np.isfinite(coefficients).all() or (coefficients < 0).any():
        raise ValueError('Diffusivities must be finite and nonnegative')
    with nc.Dataset(source) as grid:
        scalar = lambda name: int(np.asarray(grid[name][:]).item())
        if grid['J'].dimensions != ('x', 'y'):
            raise ValueError('J must have dimensions (x, y)')
        nx, ny_loaded = grid['J'].shape
        guards = scalar('y_boundary_guards') if 'y_boundary_guards' in grid.variables else 0
        j11, j21, j12, j22 = (scalar(k) for k in
                              ('jyseps1_1', 'jyseps2_1', 'jyseps1_2', 'jyseps2_2'))
        dn = j21 != j12
        ny = scalar('ny') if 'ny' in grid.variables else ny_loaded-(4 if dn else 2)*guards
        ny_inner = scalar('ny_inner') if dn else ny
        if guards < 0 or ny_loaded != ny+(4 if dn else 2)*guards:
            raise ValueError('Grid y size is inconsistent with y_boundary_guards and ny')
        if not (-1 <= j11 < j21 <= j12 < j22 < ny):
            raise ValueError('Invalid poloidal separatrix indices')
        if dn and not (j21 < ny_inner <= j12):
            raise ValueError('ny_inner is inconsistent with double-null topology')
        ixsep = min(scalar('ixseps1'), scalar('ixseps2'))
        if not 0 < ixsep < nx:
            raise ValueError('Invalid radial separatrix index')
        rows = np.arange(ny_loaded)
        y_raw = rows-guards
        if dn:
            y_raw = np.where(rows < ny_inner+2*guards, rows-guards, rows-3*guards)
        core_y = ((y_raw > j11) & (y_raw <= j21)) | ((y_raw > j12) & (y_raw <= j22))
        core = (np.arange(nx)[:, None] < ixsep) & core_y[None, :]
        pfr = (np.arange(nx)[:, None] < ixsep) & ~core_y[None, :]
        profiles = dict(D=np.full((nx, ny_loaded), D),
                        chi=np.where(core, chi_core, chi),
                        nu=np.where(pfr, nu_pfr, nu))
        dv = 2*np.pi*np.asarray(grid['J'][:])*np.asarray(grid['dx'][:])*np.asarray(grid['dy'][:])
        if dv.shape != (nx, ny_loaded) or not np.isfinite(dv).all():
            raise ValueError('Invalid grid cell-volume array')
        names = {f'{kind}_{sp}': ('x', 'y', 'z') if kind == 'nu' else ('x', 'y')
                 for sp in species for kind in profiles}
        names['dv'] = ('x', 'y')
        for name, dims in names.items():
            if name in grid.variables and grid[name].dimensions != dims:
                raise ValueError(f'{name} must have dimensions {dims}')
            if name in grid.variables and grid[name].dtype.kind != 'f':
                raise ValueError(f'{name} must use a floating-point datatype')
        if 'z' in grid.dimensions and len(grid.dimensions['z']) == 0:
            raise ValueError('Existing z dimension must be nonempty')

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix='.nc', dir=output.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        with nc.Dataset(temporary, 'a') as grid:
            if 'z' not in grid.dimensions:
                grid.createDimension('z', 1)
            nz = len(grid.dimensions['z'])
            for sp in species:
                for kind, values in profiles.items():
                    name = f'{kind}_{sp}'
                    if name not in grid.variables:
                        grid.createVariable(name, 'f8', names[name])
                    data = np.broadcast_to(values[:, :, None], (nx, ny_loaded, nz)) if kind == 'nu' else values
                    grid[name][:] = data
                    grid[name].setncatts(dict(units='m^2 s^-1',
                        bout_type='Field3D' if kind == 'nu' else 'Field2D'))
            if 'dv' not in grid.variables:
                grid.createVariable('dv', 'f8', ('x', 'y'))
            grid['dv'][:] = dv
            grid['dv'].setncatts(dict(units='m^3', bout_type='Field2D'))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'Wrote {output}: species={", ".join(species)}; '
          f'D={D:g}, chi={chi:g}/{chi_core:g} (core), nu={nu:g}/{nu_pfr:g} (PFR) m^2/s')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in', dest='grid_in', required=True)
    parser.add_argument('--out', dest='grid_out', required=True)
    parser.add_argument('--species', nargs='+', default=['e', 'd+'])
    parser.add_argument('--D', type=float, default=4.)
    parser.add_argument('--chi', type=float, default=4.)
    parser.add_argument('--chi-core', dest='chi_core', type=float, default=10.)
    parser.add_argument('--nu', type=float, default=2.)
    parser.add_argument('--nu-pfr', dest='nu_pfr', type=float, default=0.)
    parser.add_argument('--overwrite', action='store_true')
    try:
        inject_anomalous(**vars(parser.parse_args(argv)))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
