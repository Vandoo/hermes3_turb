from setuptools import setup, find_packages

setup(
    name="hermes3_turb",
    version="0.1.0",
    description="3D turbulence post-processing for Hermes-3 / BOUT++ simulations",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy", "xarray", "xbout", "matplotlib", "scipy"],
    # 3-D rendering (scripts/plot3d_filaments.py) needs one of these; mayavi
    # does not build against vtk>=9.2, so pyvista is the portable choice.
    extras_require={"plot3d": ["pyvista"]},
    python_requires=">=3.8",
)
