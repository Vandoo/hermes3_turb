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
  target_heatflux.py          Single-folder target heat flux and electron/ion split.
  target_profiles.py          Plate density, temperatures, pressure, and ion current.
  compare_target_profiles.py Overlay cached target profiles across cases.
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
  lcfs_transport_monitor.py  LCFS channel fluxes and effective turbulent transport.
  flux_limiter_check.py      Estimate where conduction/viscosity limiters act.

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

Run from the appropriate case directory. Scripts that read
config_analysis.ini still read it from the current working directory; copy the
shared template from scripts/config_analysis.ini there. Others use a CONFIG
block or command-line arguments; see each script's header. Review case paths
and settings before running.

Comparing target heat-flux profiles
----------------------------------
First run targets/v3_target_heatflux.py for each case over the desired time
window. For true midplane distances, set gridfile and fit_coordinate="omp"
before generating the caches. Check matching sheath-model
settings and appropriate averaging windows across cases yourself: the caches
do not record every calculation setting. No simulation files are opened by
the comparison script.

v3_target_heatflux.py stops before calculating if loaded dumps have no
y-boundary guards. Caches record guard presence and loaded/domain y sizes;
cache hits report that recorded status. Legacy caches without the status warn
that plate heat flux is unverified: set use_cache=False after resquashing.
Guard presence enables plate interpolation but does not certify sheath-model
choices, averaging windows, or fitted widths.

Edit root and CASES in targets/compare_target_profiles.py, or run:
  python scripts/targets/compare_target_profiles.py \
    --case "coarse" /path/to/coarse/cache/v3_target_heatflux_....npz \
    --case "fine" /path/to/fine/cache/v3_target_heatflux_....npz \
    --targets outer_lower outer_upper --xlim -5 40 --save targets.png

Each target has a separate panel. Default rows are inner_upper / outer_upper,
then inner_lower / outer_lower; --targets specifies a custom order.
The coordinate comes from the cache:
OMP/IMP mapping, physical target distance, or wetted distance. Different
coordinate labels within a panel are rejected. Old midplane-row mappings
warn because coarse grids can distort that distance; regenerate with a grid
to obtain the true midplane mapping. The script does not remap old caches.
Heat flux retains its cached definition in MW/m^2 regardless of coordinate.

Default: remove two radial guard cells at each end (--mxg 0 keeps all), plot
the arithmetic mean of per-frame heat flux and shade +/-1 sample standard
deviation over time. This band describes variation, not uncertainty in the
mean. One frame has no band. Caches without per-frame flux warn and use their
saved mean; nonfinite retained profiles are rejected. No fits are made.
--normalise peak compares shapes using each profile's peak within --xlim;
the same factor scales its band. The console reports unnormalized peaks and
their locations within that window. --no-band hides shading, --band-n changes
its width, --logy sets a log axis, and --no-show supports batch PNG output.

Comparison legends show the averaging span in milliseconds (for example,
"nx68 ny52 (Δt=15.0 ms)"). Set time_label_mode="range" in a comparison
script to show the start/end interval instead. Both forms use actual retained
timestamps; console output also reports the interval. A single snapshot is
identified explicitly. This changes labels only; means and bands retain
their existing snapshot weighting.
v3_target_heatflux.py uses the same time labels in its band legend and
profile/fit figure titles, including when the band is disabled.

Target caches already save t_s in seconds. New v3_radial_flux.py caches also
save t_s. For older caches the comparison reads only squash timestamp metadata
from the numbered folders next to cache/, applies the original restart
deduplication, and requires a matching cached frame count. Recovery warns
because the files may have changed since caching. If exact timestamps cannot
be recovered, the legend says "time unavailable"; no nominal dump interval
is assumed. Copy hermes3_turb/profile_time.py along with the updated scripts
when updating an existing package installation.

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

LCFS transport and plasma limiter diagnostics
---------------------------------------------
  python scripts/run_diagnostics/lcfs_transport_monitor.py GRID DUMP [DUMP ...] [SPECIES]
  python scripts/run_diagnostics/flux_limiter_check.py DUMP --species e i --alphas 0.1 0.3 1

The LCFS monitor loads data with xhermes and reports channel fluxes, effective
transport, and fluctuation levels. Limiter diagnostics require NumPy and
netCDF4, saved kappa_par_<species>, and a run whose limiter is disabled. They
estimate local suppression of classical flux; they do not predict a new
steady state. Ion viscosity assumes matching conduction/viscosity collision
sets; the electron viscosity estimate is an upper bound. Use --help for
species, atomic-mass, coefficient, frame, and output options.

Plate plasma profiles
---------------------
Edit CASES in targets/target_profiles.py for the data folders and grid files.
It computes sheath-plane Ne, Ni, Te, Ti, pressure, and ion saturation current
separately from the heat-flux caches. Ion reconstruction
uses each folder's archived settings before combining frames. Guardless input
is reported as a last-domain-cell estimate rather than a plate measurement.
