"""
hermes3_turb: 3D turbulence post-processing for Hermes-3 / BOUT++ simulations.

Quick start
-----------
Load data with xhermes or xbout directly, then use this package for analysis:

    import xhermes
    import hermes3_turb as h3t

    ds = xhermes.open('/path/to/sim', gridfilepath='/path/to/grid.nc',
                      geometry='toroidal', keep_yboundaries=True)

    phi = ds['phi'].isel(t=-1)
    Ne  = ds['Ne'].isel(t=-1)
    Pe  = ds['Pe'].isel(t=-1)
    Ve  = h3t.get_parallel_velocity(ds.isel(t=-1), 'e')

    # Decompose radial particle flux into mean-flow and turbulent parts
    Gamma_mean, Gamma_turb, Gamma_total = h3t.decompose_radial_particle_flux(ds, Ne, phi)

    # Poloidal heat flux toward divertor targets
    q_pol = h3t.poloidal_heat_flux(ds, Pe, Ve)
"""

from .load import (
    get_parallel_velocity,
    SPECIES_AA,
    detect_toroidal_dim,
    detect_poloidal_dim,
    load_grid_vars,
    load_curvature_vector,
    check_normalisation,
    to_field_aligned,
    cache_path,
    existing_dirs,
    dedup_restart_times,
    dedup_restart_times,
    save_cache,
    load_cache,
)
from .decompose import (
    toroidal_mean,
    fluctuation,
    fluctuation_rms,
    relative_fluctuation,
    reynolds_decompose,
    turbulent_correlation,
    time_mean,
    skewness,
    kurtosis,
    radial_statistics,
)
from .geometry import (
    radial_area_element,
    radial_area_element_dtheta,
    poloidal_area_element,
    poloidal_area_element_dx,
    cell_volume,
)
from .blobs import (
    Blob,
    Track,
    omp_index,
    radial_coordinate,
    major_radius,
    binormal_coordinate,
    poloidal_markers,
    exb_radial_velocity_slice,
    open_time_series,
    fluctuation_field,
    detect_blobs,
    detect_frame,
    track_blobs,
    track_radial_velocity,
    track_binormal_velocity,
    exb_binormal_velocity_slice,
    velocity_statistics,
    ccf_radial_velocity,
    conditional_exb_velocity,
    ccf_feasibility,
    conditional_average,
    turbulent_flux_slice,
    blob_flux_closure,
    parallel_half_length,
    blob_velocity_regimes,
    resolvability_report,
    structure_size_report,
)
from .fluxes import (
    ExB_radial_velocity,
    magnetic_drift_radial_velocity,
    radial_magnetic_drift_heat_flux,
    radial_magnetic_drift_particle_flux,
    diamagnetic_work_radial_flux,
    poloidal_magnetic_drift_particle_flux,
    poloidal_magnetic_drift_heat_flux,
    radial_flutter_fluxes,
    radial_anomalous_heat_flux,
    radial_anomalous_particle_flux,
    ExB_poloidal_velocity,
    radial_electric_field,
    parallel_to_poloidal,
    radial_particle_flux,
    radial_heat_flux,
    decompose_radial_particle_flux,
    decompose_radial_heat_flux,
    poloidal_particle_flux,
    poloidal_heat_flux,
    parallel_conductive_heat_flux,
    total_poloidal_particle_flux,
    total_poloidal_heat_flux,
)
from .target import (
    get_target_y_indices,
    get_target_boundaries,
    field_line_incidence,
    target_heat_flux_density,
    target_heat_flux_from_diagnostics,
    target_sheath_power_from_diagnostics,
    target_physical_distance,
    get_midplane_y_indices,
    omp_midplane_distance,
    omp_mapped_distance,
    eich_profile,
    fit_eich,
    integral_width,
    double_exponential_profile,
    fit_double_exp,
    total_target_power,
)
from .balance import (
    integrate_radial_surface,
    integrate_poloidal_surface,
    volume_integral,
    radial_flux_profile,
    diamagnetic_work_identity,
    poloidal_flux_profile,
    particle_balance,
    energy_balance,
)

__version__ = "0.1.0"
__all__ = [
    # Utilities
    "get_parallel_velocity",
    "SPECIES_AA",
    "detect_toroidal_dim",
    "detect_poloidal_dim",
    "to_field_aligned",
    "check_normalisation",
    # Decomposition
    "toroidal_mean",
    "fluctuation",
    "fluctuation_rms",
    "relative_fluctuation",
    "reynolds_decompose",
    "turbulent_correlation",
    "time_mean",
    "skewness",
    "kurtosis",
    "radial_statistics",
    # Geometry
    "radial_area_element",
    "radial_area_element_dtheta",
    "poloidal_area_element",
    "poloidal_area_element_dx",
    "cell_volume",
    # Fluxes
    "ExB_radial_velocity",
    "ExB_poloidal_velocity",
    "radial_electric_field",
    "parallel_to_poloidal",
    "radial_particle_flux",
    "radial_heat_flux",
    "decompose_radial_particle_flux",
    "decompose_radial_heat_flux",
    "poloidal_particle_flux",
    "poloidal_heat_flux",
    "parallel_conductive_heat_flux",
    "total_poloidal_particle_flux",
    "total_poloidal_heat_flux",
    "radial_magnetic_drift_particle_flux",
    "diamagnetic_work_radial_flux",
    "poloidal_magnetic_drift_particle_flux",
    "poloidal_magnetic_drift_heat_flux",
    # Blob identification, tracking, radial velocity
    "Blob",
    "Track",
    "omp_index",
    "radial_coordinate",
    "major_radius",
    "binormal_coordinate",
    "poloidal_markers",
    "exb_radial_velocity_slice",
    "open_time_series",
    "fluctuation_field",
    "detect_blobs",
    "detect_frame",
    "track_blobs",
    "track_radial_velocity",
    "track_binormal_velocity",
    "exb_binormal_velocity_slice",
    "velocity_statistics",
    "ccf_radial_velocity",
    "conditional_exb_velocity",
    "conditional_average",
    "turbulent_flux_slice",
    "resolvability_report",
    "blob_velocity_regimes",
    "parallel_half_length",
    "blob_flux_closure",
    "ccf_feasibility",
    "structure_size_report",
    # Target heat flux & Eich fitting
    "get_target_y_indices",
    "get_target_boundaries",
    "field_line_incidence",
    "target_heat_flux_density",
    "target_heat_flux_from_diagnostics",
    "target_sheath_power_from_diagnostics",
    "target_physical_distance",
    "get_midplane_y_indices",
    "omp_midplane_distance",
    "omp_mapped_distance",
    "eich_profile",
    "fit_eich",
    "integral_width",
    "double_exponential_profile",
    "fit_double_exp",
    "total_target_power",
    # Balance
    "integrate_radial_surface",
    "integrate_poloidal_surface",
    "volume_integral",
    "radial_flux_profile",
    "diamagnetic_work_identity",
    "poloidal_flux_profile",
    "particle_balance",
    "energy_balance",
]
