#!/usr/bin/env python3
"""
Plasma source profiles for Hermes-3 BOUT++ grids
Complete implementation of sources_realisticgeometry_background from C++

Properly handles all regions:
- PF (Private Flux) region: ix < ixseps_inner
- SOL (Scrape-Off Layer) region: ix >= ixseps_outer
- Inter-separatrix region: ixseps_inner <= ix < ixseps_outer (double-null only)

Each region has multiple Y-sections with different normalized coordinates
"""

import numpy as np
import netCDF4 as nc
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class PlasmaSourceReader:
    """
    Read BOUT++ grid files and apply plasma sources with proper guard cell handling.
    """

    def __init__(self, grid_file: str,
                 density_var: str = 'Ne_src',
                 energy_var: str = 'Pe_src',
                 MXG: int = 2,
                 MYG: int = 0):
        """
        Initialize the plasma source reader from BOUT++ grid file.

        Parameters:
        -----------
        grid_file : str
            Path to the BOUT++ grid netCDF file
        density_var : str
            Name of the density source variable. Default 'Ne_src' is the name
            Hermes-3 reads as the electron particle source (m^-3 s^-1); the grid
            field is mapped onto the *interior* cells, with guard cells set by the
            boundary condition (see save_sources_to_grid).
        energy_var : str
            Name of the pressure source variable. Hermes-3 reads 'Pe_src' [Pa/s].
        MXG : int
            Number of guard cells in X direction (default: 2)
        MYG : int
            Number of guard cells in Y direction at each target (default: 0).
            Grid files normally carry NO y-guards, so leave this at 0; the source
            then peaks on the last interior cell adjacent to each target, exactly
            where Hermes-3 places the first plasma cell before the sheath.
        """
        self.grid_file = grid_file
        self.density_var = density_var
        self.energy_var = energy_var
        self.MXG = MXG
        self.MYG = MYG

        # Background-source shape parameters, set in apply_analytical_sources().
        self._S_nT = 0.0        # background density amplitude
        self._S_ET = 0.0        # background energy amplitude (matched-energy)
        self._y_peaking = 10.0  # poloidal peaking exponent (target/midplane ratio)

        self._load_grid()
        self._load_sources()

    def _load_grid(self):
        """Load grid information from BOUT++ netCDF file."""
        try:
            with nc.Dataset(self.grid_file, 'r') as dataset:
                # Get grid dimensions
                self.nx = dataset.dimensions['x'].size
                self.ny = dataset.dimensions['y'].size

                # Calculate dimensions without guard cells
                self.nx_nogc = self.nx - 2 * self.MXG

                # Get topology information
                self.jyseps1_1 = int(dataset.variables['jyseps1_1'][:])
                self.jyseps2_1 = int(dataset.variables['jyseps2_1'][:])
                self.jyseps1_2 = int(dataset.variables['jyseps1_2'][:])
                self.jyseps2_2 = int(dataset.variables['jyseps2_2'][:])
                self.ny_inner = int(dataset.variables['ny_inner'][:])

                # Determine topology
                if self.jyseps2_1 == self.jyseps1_2:
                    self.topology = 'singlenull'
                    self.n_targets = 2
                else:
                    self.topology = 'doublenull'
                    self.n_targets = 4

                # Calculate ny without guard cells
                self.ny_nogc = self.ny - self.n_targets * self.MYG

                # Get separatrix indices
                self.ixseps1 = int(dataset.variables['ixseps1'][:])
                self.ixseps2 = int(dataset.variables['ixseps2'][:])

                # Load coordinate arrays
                if 'Rxy' in dataset.variables and 'Zxy' in dataset.variables:
                    self.R = dataset.variables['Rxy'][:]
                    self.Z = dataset.variables['Zxy'][:]
                    self.has_RZ = True
                else:
                    self.has_RZ = False

                self.elem_vol = dataset.variables['J'][:] * dataset.variables['dx'][:] * dataset.variables['dy'][:]

                print(f"Successfully loaded grid from {self.grid_file}")
                print(f"  Grid size (with guards): nx={self.nx}, ny={self.ny}")
                print(f"  Guard cells: MXG={self.MXG}, MYG={self.MYG}")
                print(f"  Topology: {self.topology} ({self.n_targets} targets)")
                print(f"  Separatrix: ixseps1={self.ixseps1}, ixseps2={self.ixseps2}")
                print(f"  Y-separatrix: jyseps1_1={self.jyseps1_1}, jyseps2_1={self.jyseps2_1}")
                print(f"                jyseps1_2={self.jyseps1_2}, jyseps2_2={self.jyseps2_2}")
                print(f"  ny_inner={self.ny_inner}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Grid file not found: {self.grid_file}")
        except KeyError as e:
            raise KeyError(f"Required variable not found in grid file: {e}")

    def _load_sources(self):
        """Load or initialize source arrays."""
        try:
            with nc.Dataset(self.grid_file, 'r') as dataset:
                if self.density_var in dataset.variables:
                    self.S = dataset.variables[self.density_var][:]
                    print(f"  Loaded {self.density_var} from file")
                else:
                    self.S = np.zeros((self.nx, self.ny))

                if self.energy_var in dataset.variables:
                    self.S_E = dataset.variables[self.energy_var][:]
                    print(f"  Loaded {self.energy_var} from file")
                else:
                    self.S_E = np.zeros((self.nx, self.ny))

        except Exception as e:
            print(f"  Initializing sources to zeros: {e}")
            self.S = np.zeros((self.nx, self.ny))
            self.S_E = np.zeros((self.nx, self.ny))

    def _bg(self, y: float) -> float:
        """
        Poloidal background-source shape, normalised to 1 at the target (y=0,1).

        Uses exp(a*|y-0.5|)/(exp(a/2)-1) with a = self._y_peaking. The target/midplane
        ratio is ~exp(a/2): the default a=10 spikes ~148x on the last cell before the
        sheath (the STORM convention). Softening to a~4-6 moves weight upstream into
        the leg (better penetration, gentler last-cell gradient -> easier on the
        parallel solver), closer to a physical ionisation front a few cm off the plate.
        """
        a = self._y_peaking
        return np.exp(a * np.abs(y - 0.5)) / (np.exp(0.5 * a) - 1.0)

    def _add_bg(self, ix: int, iy: int, y: float):
        """Add the background density (and optional matched energy) source at one cell."""
        p = self._bg(y)
        self.S[ix, iy] += self._S_nT * p
        if self._S_ET != 0.0:
            self.S_E[ix, iy] += self._S_ET * p

    def apply_analytical_sources(self, S_n0: float = 1.0, S_E0: float = 1.0,
                                 S_nT: float = 1.0, S_ET: float = 0.0,
                                 x_Sn: float = 0.05, x_SE: float = 0.05,
                                 w_Sn: float = 0.05, w_SE: float = 0.05,
                                 y_peaking: float = 10.0,
                                 S_E_baseline: float = 0.0,
                                 apply_core_sources: bool = True,
                                 apply_background_sources: bool = True):
        """
        Apply analytical source profiles matching C++ sources_realisticgeometry_background.

        This is a complete implementation of all regions from the C++ code.

        Parameters:
        -----------
        S_n0, S_E0 : float
            Core (closed-field-line) source amplitudes (density, energy).
        S_nT : float
            Divertor background DENSITY amplitude. Calibrate against the printed
            Sp_tar [/s] (total particles/s injected outside the core).
        S_ET : float
            Divertor background ENERGY amplitude (matched-energy source with the
            same poloidal shape as the density background). Default 0.0 = pure
            particle source. Set > 0 to avoid dilution-cooling the divertor toward
            the temperature floor when raising the density.
        x_Sn, x_SE : float
            Radial peak locations of the core Gaussian (normalized).
        w_Sn, w_SE : float
            Radial widths of the core Gaussian.
        y_peaking : float
            Poloidal peaking exponent of the background source (default 10.0 = STORM
            convention; ~4-6 spreads it into the leg). See _bg().
        S_E_baseline : float
            Uniform energy-source floor (default 0.0). The original port set this to
            1.0 everywhere, which is a large unphysical floor in Hermes-3 units --
            left configurable but defaulting to 0.
        apply_core_sources : bool
            Apply Gaussian core sources
        apply_background_sources : bool
            Apply exp(a|y-0.5|) background sources
        """
        if self.topology not in ['singlenull', 'doublenull']:
            raise ValueError(f"Topology '{self.topology}' not supported")

        # Stash background-shape parameters for the per-region helpers.
        self._S_nT = S_nT
        self._S_ET = S_ET
        self._y_peaking = y_peaking

        # Initialize sources
        self.S[:, :] = 0.0
        self.S_E[:, :] = S_E_baseline

        ixseps_inner = min(self.ixseps1, self.ixseps2)
        ixseps_outer = max(self.ixseps1, self.ixseps2)

        print("\nApplying sources:")
        print(f"  ixseps_inner={ixseps_inner}, ixseps_outer={ixseps_outer}")
        print(f"  Core sources: {apply_core_sources}")
        print(f"  Background sources: {apply_background_sources} (y_peaking={y_peaking}, "
              f"target/midplane ratio ~{np.exp(0.5*y_peaking):.1f})")
        print(f"  Matched background energy: {'on' if S_ET != 0.0 else 'off'} (S_ET={S_ET})")

        # Loop over grid (excluding guard cells in X)
        xstart = self.MXG
        xend = self.nx - self.MXG - 1
        ystart = self.MYG
        yend = self.ny - self.MYG - 1

        # Loop over radial points
        for ix in range(xstart, xend + 1):
            # Get global X index (same as local for single processor)
            ix_global = ix

            # Normalized radial coordinate for core sources
            x = (ix_global - 1.5) / (self.nx - 4)

            # Determine X-region
            if apply_background_sources:
                if ix_global < ixseps_inner:
                    # PF (Private Flux) region
                    self._add_pf_sources(ix)

                elif ix_global >= ixseps_outer:
                    # SOL (Scrape-Off Layer) region
                    self._add_sol_sources(ix)

                else:
                    # Inter-separatrix region (only for double-null)
                    if self.ixseps1 < self.ixseps2:
                        # Lower double null
                        self._add_intersep_lower_dn(ix)
                    elif self.ixseps1 > self.ixseps2:
                        # Upper double null
                        self._add_intersep_upper_dn(ix)

            # Apply core sources to closed field line region
            if apply_core_sources:
                _core_rows_seen = set()
                for iy in range(ystart, yend + 1):
                    iy_nogc = self._get_global_y_no_boundaries(iy)

                    # Check if in core region
                    in_core = False
                    if ((iy_nogc > self.jyseps1_1 and iy_nogc <= self.jyseps2_1) or
                        (iy_nogc > self.jyseps1_2 and iy_nogc <= self.jyseps2_2)):
                        in_core = True

                    if in_core:
                        _core_rows_seen.add(iy_nogc)
                        self.S[ix, iy] += S_n0 * np.exp(-((x - x_Sn) / w_Sn)**2)
                        self.S_E[ix, iy] += S_E0 * np.exp(-((x - x_SE) / w_SE)**2)

                # Verify that core sources cover exactly the closed-field-line rows.
                _expect = set(range(self.jyseps1_1 + 1, self.jyseps2_1 + 1)) \
                        | set(range(self.jyseps1_2 + 1, self.jyseps2_2 + 1))
                if _core_rows_seen != _expect:
                    raise AssertionError(
                        "core mask does not match the closed-field-line rows: "
                        f"extra={sorted(_core_rows_seen - _expect)} "
                        f"missing={sorted(_expect - _core_rows_seen)} "
                        f"(jyseps={self.jyseps1_1},{self.jyseps2_1},"
                        f"{self.jyseps1_2},{self.jyseps2_2})")

        print("\nSource statistics:")
        print(f"  Density source range: [{self.S.min():.4e}, {self.S.max():.4e}]")
        print(f"  Energy source range: [{self.S_E.min():.4e}, {self.S_E.max():.4e}]")

        Sp_map = self.S * self.elem_vol * (2*np.pi)
        Sp_tot = Sp_map[2:-2, :].sum()
        Sp_core = (Sp_map[2:ixseps_inner, self.jyseps1_1 + self.MYG+1                   :self.jyseps2_1 + self.MYG+1].sum()
                  +Sp_map[2:ixseps_inner, self.jyseps1_2 + self.MYG*(self.n_targets-1)+1:self.jyseps2_2 + self.MYG*(self.n_targets-1)+1].sum()
                )
        print("  Sp_core = %.3e /s, Sp_tar = %.3e /s" % (Sp_core, Sp_tot-Sp_core))



    def _get_global_y_no_boundaries(self, iy: int) -> int:
        """Convert local Y index to global Y index without boundaries (no guard cells)."""
        if self.n_targets == 2:
            return iy - self.MYG
        elif self.n_targets == 4:
            if iy <= self.ny_inner-1+self.MYG:
                return iy - self.MYG
            else:
                return iy - self.MYG*(self.n_targets-1)

    def _add_pf_sources(self, ix: int):
        """Apply sources to Private Flux (PF) region."""
        # Lower inner divertor (0 to jyseps1_1)
        for iy in range(self.MYG, self.jyseps1_1 +self.MYG + 1):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.jyseps1_1 + 1)
            y = (iy_nogc + 0.5) * dy  # 0 < y < 0.5
            self._add_bg(ix, iy, y)

        # Upper inner divertor (jyseps2_1 to ny_inner)
        for iy in range(self.jyseps2_1 +self.MYG + 1, self.ny_inner +self.MYG):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.ny_inner - self.jyseps2_1 - 1)
            y = (iy_nogc - self.jyseps2_1 - 0.5) * dy + 0.5  # 0.5 < y < 1.0
            self._add_bg(ix, iy, y)

        # Upper outer divertor (ny_inner to jyseps1_2)
        for iy in range(self.ny_inner +self.MYG*(self.n_targets-1), self.jyseps1_2 +self.MYG*(self.n_targets-1) + 1):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.jyseps1_2 - self.ny_inner + 1)
            y = (iy_nogc - self.ny_inner + 0.5) * dy  # 0 < y < 0.5
            self._add_bg(ix, iy, y)

        # Lower outer divertor (jyseps2_2 to end)
        for iy in range(self.jyseps2_2 +self.MYG*(self.n_targets-1) + 1, self.ny):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.ny - self.jyseps2_2 - 1)
            y = (iy_nogc - self.jyseps2_2 - 0.5) * dy + 0.5  # 0.5 < y < 1.0
            self._add_bg(ix, iy, y)

    def _add_sol_sources(self, ix: int):
        """Apply sources to SOL (Scrape-Off Layer) region."""
        # SOL region on inner side
        for iy in range(self.MYG, self.ny_inner + self.MYG):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 1.0 / self.ny_inner
            y = (iy_nogc + 0.5) * dy  # 0 < y < 1
            self._add_bg(ix, iy, y)
        # SOL region on outer side
        for iy in range(self.ny_inner +self.MYG*(self.n_targets-1), self.ny):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 1.0 / (self.ny - self.ny_inner)
            y = (iy_nogc - self.ny_inner + 0.5) * dy  # 0 < y < 1
            self._add_bg(ix, iy, y)

    def _add_intersep_lower_dn(self, ix: int):
        """Apply sources to inter-separatrix region for lower double-null."""
        # Lower inner divertor leg (0 to jyseps2_1)
        for iy in range(self.MYG, self.jyseps2_1 +self.MYG + 1):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.jyseps2_1 + 1)
            y = (iy_nogc + 0.5) * dy  # 0 < y < 0.5
            self._add_bg(ix, iy, y)

        # Upper inner divertor leg (jyseps2_1 to ny_inner)
        for iy in range(self.jyseps2_1 +self.MYG + 1, self.ny_inner +self.MYG):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.ny_inner - self.jyseps2_1 - 1)
            y = (iy_nogc - self.jyseps2_1 - 0.5) * dy + 0.5  # 0.5 < y < 1.0
            self._add_bg(ix, iy, y)

        # Upper outer divertor leg (ny_inner to jyseps1_2)
        for iy in range(self.ny_inner +self.MYG*(self.n_targets-1), self.jyseps1_2 +self.MYG*(self.n_targets-1) + 1):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.jyseps1_2 - self.ny_inner + 1)
            y = (iy_nogc - self.ny_inner + 0.5) * dy  # 0 < y < 0.5
            self._add_bg(ix, iy, y)

        # Lower outer divertor leg (jyseps1_2 to end)
        for iy in range(self.jyseps1_2 +self.MYG*(self.n_targets-1), self.ny):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.ny - self.jyseps1_2 - 1)
            y = (iy_nogc - self.jyseps1_2 - 0.5) * dy + 0.5  # 0.5 < y < 1.0
            self._add_bg(ix, iy, y)

    def _add_intersep_upper_dn(self, ix: int):
        """Apply sources to inter-separatrix region for upper double-null."""
        # Lower inner divertor leg (0 to jyseps1_1)
        for iy in range(self.MYG, self.jyseps1_1 +self.MYG + 1):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.jyseps1_1 + 1)
            y = (iy_nogc + 0.5) * dy  # 0 < y < 0.5
            self._add_bg(ix, iy, y)

        # Upper inner divertor leg (jyseps1_1 to ny_inner)
        for iy in range(self.jyseps1_1 +self.MYG + 1, self.ny_inner +self.MYG):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.ny_inner - self.jyseps1_1 - 1)
            y = (iy_nogc - self.jyseps1_1 - 0.5) * dy + 0.5  # 0.5 < y < 1.0
            self._add_bg(ix, iy, y)

        # Upper outer divertor leg (ny_inner to jyseps2_2)
        for iy in range(self.ny_inner +self.MYG*(self.n_targets-1), self.jyseps2_2 +self.MYG*(self.n_targets-1) + 1):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.jyseps2_2 - self.ny_inner + 1)
            y = (iy_nogc - self.ny_inner + 0.5) * dy  # 0 < y < 0.5
            self._add_bg(ix, iy, y)

        # Lower outer divertor leg (jyseps2_2 to end)
        for iy in range(self.jyseps2_2 +self.MYG*(self.n_targets-1) + 1, self.ny):
            iy_nogc = self._get_global_y_no_boundaries(iy)
            dy = 0.5 / (self.ny - self.jyseps2_2 - 1)
            y = (iy_nogc - self.jyseps2_2 - 0.5) * dy + 0.5  # 0.5 < y < 1.0
            self._add_bg(ix, iy, y)

    def get_sources_for_bout(self, LocalNz: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get source arrays in BOUT++ format (3D with toroidal dimension).

        Returns:
        --------
        S_3d : np.ndarray
            Density source [nx, ny, nz]
        S_E_3d : np.ndarray
            Energy source [nx, ny, nz]
        """
        S_3d = np.repeat(self.S[:, :, np.newaxis], LocalNz, axis=2)
        S_E_3d = np.repeat(self.S_E[:, :, np.newaxis], LocalNz, axis=2)
        return S_3d, S_E_3d

    def plot_sources(self, figsize: Tuple[int, int] = (14, 6),
                    cmap: str = 'hot', log_scale: bool = True,
                    use_RZ: bool = True):
        """Plot the density and energy sources."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        if use_RZ and self.has_RZ:
            X, Y = self.R, self.Z
            xlabel, ylabel = 'R (m)', 'Z (m)'
        else:
            X = np.arange(self.nx)
            Y = np.arange(self.ny)
            X, Y = np.meshgrid(X, Y, indexing='ij')
            xlabel, ylabel = 'x index', 'y index'

        if log_scale:
            S_plot = np.log10(np.abs(self.S) + 1e-30)
            S_E_plot = np.log10(np.abs(self.S_E) + 1e-30)
            label_S = 'log10(|S|)'
            label_S_E = 'log10(|S_E|)'
        else:
            S_plot = self.S
            S_E_plot = self.S_E
            label_S = 'S (density source)'
            label_S_E = 'S_E (energy source)'

        im1 = ax1.contourf(X, Y, S_plot, levels=30, cmap=cmap)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title('Density Source')
        if use_RZ and self.has_RZ:
            ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label=label_S)

        im2 = ax2.contourf(X, Y, S_E_plot, levels=30, cmap=cmap)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title('Energy Source')
        if use_RZ and self.has_RZ:
            ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label=label_S_E)

        plt.tight_layout()
        return fig, (ax1, ax2)

    def save_sources_to_grid(self, output_file: Optional[str] = None):
        """Save sources to grid file."""
        if output_file is None:
            output_file = self.grid_file
            mode = 'a'
        else:
            import shutil
            shutil.copy(self.grid_file, output_file)
            mode = 'a'

        with nc.Dataset(output_file, mode) as dataset:
            if self.density_var in dataset.variables:
                dataset.variables[self.density_var][:] = self.S
            else:
                S_var = dataset.createVariable(self.density_var, 'f8', ('x', 'y'))
                S_var[:] = self.S
                S_var.long_name = 'Density source term'

            if self.energy_var in dataset.variables:
                dataset.variables[self.energy_var][:] = self.S_E
            else:
                S_E_var = dataset.createVariable(self.energy_var, 'f8', ('x', 'y'))
                S_E_var[:] = self.S_E
                S_E_var.long_name = 'Energy source term'

        print(f"Saved sources to {output_file}")
