hermes3_turb
============

Python analysis routines and scripts for Hermes-3 / BOUT++ simulations.
Includes particle and heat fluxes, conservation checks, divertor target
profiles, geometry diagnostics, run monitoring, and restart tools.

Install in your analysis environment:
  python3 -m pip install -e .

Scripts that load Hermes-3 data also require xhermes. The reusable package
provides analysis functions; use xhermes, xbout, or boutdata directly for
loading. Start with scripts/README.txt for the script catalogue.

Scripts are grouped into fluxes/, balances/, targets/, geometry/,
run_diagnostics/, and restarts/. Most use a CONFIG block; some use command-line
arguments or config_analysis.ini in the current working directory. Read each
script's header and adjust case paths and settings before running it.
Restart/grid tools can write simulation files; review their modes and paths.

Run synthetic regression tests from this checkout:
  python3 -m pytest tests -q

The tests cover selected calculations and known regressions, not complete
validation against real runs. Check units, species, guard cells, averaging
windows, and diagnostic availability when interpreting results.
