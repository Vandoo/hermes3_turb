Scripts by purpose
==================

Paths below are relative to this directory.

fluxes/ — radial and poloidal transport
  v3_radial_flux.py           Multi-folder channel split and OMP fall-off profiles.
  v3_poloidal_flux.py         Poloidal particle and heat flux by channel.
  compare_exb_magdrift.py     Cross-check ExB and magnetic-drift expressions.
  fout_timeseries.py          Per-frame radial heat flux and autocorrelation.

balances/ — sources, inventories, and conservation
  v3_plasma_balance.py        Multi-folder global and poloidal balance checks.
  v3_balance_diagnose.py      Source, LCFS flux, and core storage terms.
  v3_radial_conservation.py   Enclosed sources versus radial surface flux.
  particle_fill_time.py       Plasma/neutral inventories and filling times.
  source_radiation_audit.py   Sources, recycling, pumping, and energy sinks.

targets/ — divertor heat loads and cross-checks
  v3_target_heatflux.py       Multi-folder averaged target analysis.
  target_poloidal_check.py    Target sheath load versus upstream poloidal power.
  validate_poloidal_efe.py    Heat-flux formulas versus saved Y-face diagnostics.

geometry/ — equilibrium and grid properties
  connection_length.py       Connection lengths and parallel residence times.
  flux_expansion.py          Poloidal flux expansion along divertor legs.
  eq_params.py               Equilibrium field, q95, and separatrix density.
  adjust_curvature.py        Check/correct grid curvature; can modify grid files.

run_diagnostics/ — run histories, health, and performance
  h2time_hermes.py            Time-trace diagnostics across restarts.
  turbulence_monitor.py      Fluctuation growth and saturation monitoring.
  inspect_hermes_run.py       Sources, field inspection, and extreme values.
  debug_hermes_run.py         Crash seeds, inventories, and solver conditioning.
  cvode_report.py            Compare CVODE solver statistics from logs.
  parse_slurm_speed.py       Parse Slurm output and simulation speed.
  run_ledger.py              Restart history, inputs, binaries, and run status.

restarts/ — tools that write restart or interpolated data files
  refine_restart.py          Refine/coarsen grids and produce restart files.
  rescale_plasma_restart.py  Rescale restart plasma density.

grid_tools/ — prepare grid source and anomalous-transport fields
  rebuild_src.py             Calibrate core and near-target plasma sources.
  inject_anomalous.py        Write species D, chi, and nu coefficient profiles.

Running a script
----------------
Install the package or put the package checkout on PYTHONPATH. Run scripts
by path, for example:
  python /path/to/hermes3_turb/scripts/targets/v3_target_heatflux.py

Keep the working directory appropriate for your case. Scripts that read
config_analysis.ini still read it from the current working directory; copy the
shared template from scripts/config_analysis.ini there. Others use a CONFIG
block or command-line arguments; see each script's header. Review case paths
and settings before running.

Preparing grid sources and anomalous diffusivities
-------------------------------------------------
Install the package first (python3 -m pip install -e .). These tools require
explicit, different input/output grid paths and preserve unrelated grid fields.
Existing output files require --overwrite; input grids are never edited.

Build sources on a guardless double-null grid:
  python scripts/grid_tools/rebuild_src.py --in grid.nc --out grid_src.nc \
    --mode cd --flux 1e22 --heating 300e3 --target-frac 0.02 --T-inj 5

--flux is the integrated particle input on the core poloidal rows [s^-1].
--mode cd sets the fraction of total particle input on non-core rows;
--mode sxd makes that contribution zero. These are source configurations,
not restrictions to one magnetic geometry. --heating is total power per
species [W], so the example gives 600 kW for electrons plus ions. The core
pressure source is adjusted to include the near-target injection energy
within that total. --T-inj is the target injection temperature [eV].
--match existing_src.nc reproduces that grid's core-row particle input and
total e+i heating instead of --flux/--heating. Matching requires the same
grid dimensions and topology. The calibration uses MXG=2 and no y guards;
guarded or single-null input grids are rejected by this driver.

Outputs: Ne_src [m^-3 s^-1], Pe_src=Pi_src [Pa s^-1]. Energy input is
3/2 times the pressure-source volume integral. Hermes-3 reads these fields
from the grid; leave the corresponding source expressions unset in BOUT.inp
and use source_only_in_core=false when retaining near-target sources.
The reusable profile class is available as:
  from hermes3_turb.plasma_source_reader import PlasmaSourceReader

Add anomalous transport profiles, selecting the actual species labels:
  python scripts/grid_tools/inject_anomalous.py \
    --in grid_src.nc --out grid_src_diff.nc --species e i \
    --D 4 --chi 4 --chi-core 10 --nu 2 --nu-pfr 0

All coefficients are physical m^2/s; Hermes-3 normalizes them internally.
D_<species> is uniform; chi_<species> uses --chi-core inside the inner
separatrix on closed poloidal bands and --chi elsewhere. nu_<species> uses
--nu-pfr inside that separatrix on private-flux bands and --nu elsewhere.
Core/PFR selection accounts for single-/double-null topology and grid y
guards. D and chi are Field2D; nu is Field3D, repeated across every z plane
(a singleton z dimension is added if absent). dv stores 2*pi*J*dx*dy [m^3].
Existing fields are updated. Species anomalous_D, anomalous_chi and
anomalous_nu input options override grid fields; leave them unset to use
these profiles. The applicable anomalous_diffusion component must be enabled.
