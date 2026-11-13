hermes3_turb
============

Python analysis routines and scripts for Hermes-3 / BOUT++ simulations.
Includes particle and heat fluxes, conservation checks, divertor target
profiles, geometry diagnostics, run monitoring, restart tools, and grid source /
anomalous-transport preparation.

Install in an analysis environment:
  python3 -m pip install -e .

Scripts that load Hermes-3 data also require xhermes. The reusable package
provides analysis functions; use xhermes, xbout, or boutdata directly for
loading. Start with scripts/README.txt for the script catalogue.

Target heat flux scripts require saved Ve ([e] diagnose=true) and retained
target guard cells. For the full sheath model, missing Vi/Ti can be
reconstructed before time averaging.

Scripts are grouped into fluxes/, balances/, targets/, geometry/,
run_diagnostics/, restarts/, and grid_tools/. Most use a CONFIG block; some use
command-line arguments or config_analysis.ini in the current working directory. Read each
script's header and adjust case paths and settings before running it.
Restart/grid tools can write simulation files; review their modes and paths.
grid_tools/rebuild_src.py calibrates core/target particle and pressure sources.
grid_tools/inject_anomalous.py writes species diffusivities. Both require
separate input/output paths; see scripts/README.txt for examples and units.

Run synthetic regression tests from this checkout:
  python3 -m pytest tests -q

The tests cover selected calculations and known regressions, not complete
validation against real runs. Check units, species, guard cells, averaging
windows, and diagnostic availability when interpreting results.

Energy and balance conventions
------------------------------
The perpendicular ExB energy flux uses (3/2) P v_E. Parallel advection,
diamagnetic drift, and magnetic flutter retain their channel coefficients.
Volume integrals exclude boundary guards by default. energy_source() converts
a saved pressure source to an internal-energy source using the factor 3/2.
Use parallel_energy_flow_ylow() to select parallel energy-flow channels; the
published ef*_tot_ylow may include diagnostic-only viscous heating.

Target heat flux can return electron/ion components and plate profiles
through components=True. Cache names include a calculation-format version.
