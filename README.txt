hermes3_turb
============

Python analysis routines and scripts for Hermes-3 / BOUT++ simulations.
Includes particle and heat fluxes, conservation checks, divertor target
profiles, geometry diagnostics, run monitoring, restart tools, and grid source /
anomalous-transport preparation.

Install in your analysis environment:
  python3 -m pip install -e .

Scripts that load Hermes-3 data also require xhermes. The reusable package
provides analysis functions; use xhermes, xbout, or boutdata directly for
loading. Start with scripts/README.txt for the script catalogue.

Target heat flux scripts require saved Ve ([e] diagnose=true). Missing Vi/Ti
is supported through the validated full-sheath reconstruction before averaging;
retain all target guard cells and use the updated target.py with the scripts.
The target cache version is algn_recVi_Ve_fxv10; older caches are recomputed.

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
