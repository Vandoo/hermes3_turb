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
