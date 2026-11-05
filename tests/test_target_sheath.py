"""Analytical sheath and shifted-field regressions; no simulation jobs needed."""
import unittest
import ast
from pathlib import Path
import numpy as np
import xarray as xr
from hermes3_turb.target import (
    get_target_boundaries, recover_target_ion_velocity, target_heat_flux_density,
)

QE=1.602176634e-19
MP=1.67262192369e-27

def sheath_dataset():
    nt,nx,ny,nz=2,6,8,27
    cs=np.sqrt(QE*50/MP)
    meta={'MXG':1,'MYG':1,'ny':4,'ny_inner':2,'Nnorm':1e19,'Tnorm':50.,'Cs0':cs}
    ds=xr.Dataset(attrs={'metadata':meta,'options':{
        'hermes':{'components':'e,i,sheath_boundary'},
        'i':{'density_floor':'1e-4','adiabatic':'5/3'}}})
    shape=(nt,nx,ny,nz)
    values={'Ne':1e19,'Ni':1e19,'Te':20.,'Ti':10.,'Pe':20e19*QE,
            'Pi':10e19*QE,'phi':60.,'Ve':-2e4,'Vi':0.,'NVi':0.}
    for name,value in values.items():
        ds[name]=xr.DataArray(np.full(shape,value),dims=('t','x','y','z'),
                             attrs={'units_type':'SI'})
    ds['NVi'].attrs['conversion']=MP*1e19*cs
    sonic=np.sqrt(QE*((5/3)*10+20)/(2*MP))
    for domain,guard in get_target_boundaries(ds)[0].values():
        sign=-1 if guard<domain else 1
        # First frame subsonic, second supersonic: the latter must not be capped.
        interior=sign*sonic*np.array([.5,1.8])[:,None,None]
        face=sign*sonic*np.array([1.,1.8])[:,None,None]
        ds['Vi'].values[:,:,domain,:]=interior
        ds['Vi'].values[:,:,guard,:]=2*face-interior
        ds['NVi'].values[:,:,domain,:]=2*MP*1e19*interior
        # This misleading momentum guard is deliberately NOT a velocity guard.
        ds['NVi'].values[:,:,guard,:]=-ds['NVi'].values[:,:,domain,:]
    for name,value in [('dx',1.),('g_11',1.),('g_22',4.),('g_33',1.),('Bxy',1.)]:
        ds[name]=xr.DataArray(np.full((nx,ny),value),dims=('x','y'))
    for name in ['g_13','g_23']:
        ds[name]=xr.DataArray(np.zeros((nx,ny)),dims=('x','y'))
    ds['zShift']=xr.DataArray(np.broadcast_to(np.linspace(-.7,.9,ny),(nx,ny)),dims=('x','y'))
    ds['dz']=xr.DataArray(2*np.pi/nz)
    return ds

def heat(ds,target):
    return target_heat_flux_density(ds,ds.Pe,ds.Pi,ds.Ve,ds.Vi,
                                   phi=ds.phi,target=target,bohm_cap_v=False)[1]

class TestTargetSheath(unittest.TestCase):
    def test_missing_vi_and_ti_recover_sonic_and_supersonic_sheaths(self):
        saved=sheath_dataset()
        old=saved.drop_vars(['Vi','Ti'])
        recovered=recover_target_ion_velocity(old)
        for target,(domain,guard) in get_target_boundaries(saved)[0].items():
            np.testing.assert_allclose(recovered.Vi.isel(y=[domain,guard]),
                                       saved.Vi.isel(y=[domain,guard]),rtol=2e-14)
            np.testing.assert_allclose(heat(recovered,target),heat(saved,target),rtol=2e-14)
        self.assertNotIn('Vi',old)

    def test_shifted_correlated_fields_match_the_known_aligned_profile(self):
        physical=sheath_dataset()
        z=2*np.pi*np.arange(27)/27
        for name,amplitude in [('Pe',.5),('Ve',.4),('Pi',.4),('Vi',.35)]:
            physical[name]=physical[name]*(1+amplitude*xr.DataArray(np.cos(3*z),dims='z'))
        raw=physical.copy(deep=True)
        phase=np.exp(-1j*raw.zShift.values[None,:,:,None]*np.arange(14)[None,None,None,:])
        for name in ['Pe','Pi','Ve','Vi','Te','Ti','Ne','Ni','phi']:
            raw[name]=raw[name].copy(data=np.fft.irfft(np.fft.rfft(physical[name],axis=-1)*phase,n=27,axis=-1))
        for target in get_target_boundaries(raw)[0]:
            truth=target_heat_flux_density(physical,physical.Pe,physical.Pi,physical.Ve,
                    physical.Vi,phi=physical.phi,target=target,bohm_cap_v=False,
                    align_fields=False)[1]
            np.testing.assert_allclose(heat(raw,target),truth,rtol=3e-14)

    def test_recover_before_mixing_diagnosed_and_undiagnosed_frames(self):
        saved=sheath_dataset()
        first=saved.isel(t=slice(0,1)).drop_vars(['Vi','Ti'])
        second=saved.isel(t=slice(1,2))
        joined=xr.concat([recover_target_ion_velocity(first),second],dim='t')
        for target in get_target_boundaries(saved)[0]:
            np.testing.assert_allclose(heat(joined,target),heat(saved,target),rtol=2e-14)

    def test_direct_api_accepts_missing_vi(self):
        saved=sheath_dataset();old=saved.drop_vars(['Vi','Ti'])
        for target in get_target_boundaries(saved)[0]:
            q=target_heat_flux_density(old,old.Pe,old.Pi,old.Ve,phi=old.phi,
                                      target=target,bohm_cap_v=False)[1]
            np.testing.assert_allclose(q,heat(saved,target),rtol=2e-14)

    def test_bout_options_object_supplies_floor_and_adiabatic(self):
        from boutdata.data import BoutOptions
        ds=sheath_dataset().drop_vars(['Vi','Ti'])
        options=BoutOptions()
        options.getSection('i')['density_floor']='1e-4'
        options['i']['adiabatic']='5/3'
        ds.attrs['options']=options
        recover_target_ion_velocity(ds)

    def test_existing_saved_velocity_is_preserved(self):
        ds=sheath_dataset()
        self.assertIs(recover_target_ion_velocity(ds),ds)

    def test_named_coordinates_and_lazy_chunks(self):
        ds=sheath_dataset().rename({'y':'theta','z':'zeta'})
        ds=ds.assign_coords(theta=np.linspace(.1,.8,8),zeta=np.arange(27))
        lazy=ds.drop_vars(['Vi','Ti']).chunk({'t':1,'theta':2,'zeta':9})
        recovered=recover_target_ion_velocity(lazy)
        for target in get_target_boundaries(ds)[0]:
            np.testing.assert_allclose(heat(recovered,target),heat(ds,target),rtol=2e-14)

    def test_script_q_for_handles_reconstructed_fields(self):
        from hermes3_turb.load import get_parallel_velocity
        from hermes3_turb.target import target_heat_flux_from_diagnostics
        root=Path(__file__).resolve().parents[1]
        script=next(p for p in [root/'scripts/v3_target_heatflux.py',
                                root/'scripts/targets/v3_target_heatflux.py',
                                Path(__file__).with_name('v3_target_heatflux.py')] if p.exists())
        tree=ast.parse(script.read_text())
        compute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_compute')
        qfor=next(n for n in compute.body if isinstance(n,ast.FunctionDef) and n.name=='_q_for')
        scope={'np':np,'prefer_diagnostics':False,'ion_species':'i','use_phi_gamma':True,
               'phi_wall':0.,'AA_ion':2.,'Z_ion':1.,'bohm_cap_v':False,
               'get_parallel_velocity':get_parallel_velocity,
               'target_heat_flux_density':target_heat_flux_density,
               'target_heat_flux_from_diagnostics':target_heat_flux_from_diagnostics}
        exec(compile(ast.Module(body=[qfor],type_ignores=[]),str(script),'exec'),scope)
        saved=sheath_dataset()
        recovered=recover_target_ion_velocity(saved.drop_vars(['Vi','Ti']))
        for target in get_target_boundaries(saved)[0]:
            np.testing.assert_allclose(scope['_q_for'](recovered,target),heat(saved,target),rtol=2e-14)

    def test_simple_sheath_cannot_silently_use_full_sheath_reconstruction(self):
        ds=sheath_dataset().drop_vars('Vi')
        ds.attrs['options']['hermes']['components']='sheath_boundary_simple'
        with self.assertRaisesRegex(ValueError,'sheath_boundary_simple'):
            recover_target_ion_velocity(ds)

    def test_bohm_smoothing_refuses_max_rule_reconstruction(self):
        ds=sheath_dataset().drop_vars('Vi')
        ds.attrs['options']['sheath_boundary']={'bohm_smoothing':'0.1'}
        with self.assertRaisesRegex(ValueError,'bohm_smoothing'):
            recover_target_ion_velocity(ds)

    def test_identity_transform_is_not_shifted_again(self):
        # Data stored aligned (identity transform) must give the same q as the
        # explicit align_fields=False call, even though zShift is present.
        physical=sheath_dataset()
        z=2*np.pi*np.arange(27)/27
        for name,amplitude in [('Pe',.5),('Ve',.4),('Pi',.4),('Vi',.35)]:
            physical[name]=physical[name]*(1+amplitude*xr.DataArray(np.cos(3*z),dims='z'))
        physical.attrs['options']['mesh']={'paralleltransform':{'type':'identity'}}
        for target in get_target_boundaries(physical)[0]:
            truth=target_heat_flux_density(physical,physical.Pe,physical.Pi,physical.Ve,
                    physical.Vi,phi=physical.phi,target=target,bohm_cap_v=False,
                    align_fields=False)[1]
            np.testing.assert_allclose(heat(physical,target),truth,rtol=1e-14)

    def test_flat_identity_options_are_not_shifted_again(self):
        physical=sheath_dataset()
        z=2*np.pi*np.arange(27)/27
        for name,amplitude in [('Pe',.5),('Ve',.4),('Pi',.4),('Vi',.35)]:
            physical[name]=physical[name]*(1+amplitude*xr.DataArray(np.cos(3*z),dims='z'))
        physical.attrs['options']['mesh:paralleltransform']={'type':'"identity"'}
        for target in get_target_boundaries(physical)[0]:
            truth=target_heat_flux_density(physical,physical.Pe,physical.Pi,physical.Ve,
                    physical.Vi,phi=physical.phi,target=target,bohm_cap_v=False,
                    align_fields=False)[1]
            np.testing.assert_allclose(heat(physical,target),truth,rtol=1e-14)

    def test_saved_vi_does_not_require_a_max_rule_reconstruction(self):
        ds=sheath_dataset()
        ds.attrs['options']['sheath_boundary']={'bohm_smoothing':'0.1'}
        self.assertIs(recover_target_ion_velocity(ds),ds)

    def test_unsupported_transform_is_not_silently_treated_as_identity(self):
        ds=sheath_dataset()
        ds.attrs['options']['mesh']={'paralleltransform':{'type':'shiftedinterp'}}
        with self.assertRaisesRegex(ValueError,'does not implement transform'):
            heat(ds,'outer_lower')

    def test_scripts_refuse_missing_ve_before_ion_reconstruction(self):
        root=Path(__file__).resolve().parents[1]
        candidates=[root/'scripts',root/'scripts/targets',Path(__file__).parent]
        checked=0
        for name in ['v3_target_heatflux.py','target_poloidal_check.py','target_heatflux.py']:
            paths=[p/name for p in candidates if (p/name).exists()]
            if not paths:continue
            tree=ast.parse(paths[0].read_text())
            gate=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                      and isinstance(n.test,ast.Compare)
                      and isinstance(n.test.left,ast.Constant) and n.test.left.value=='Ve'
                      and isinstance(n.test.ops[0],ast.NotIn))
            dataset_name=gate.test.comparators[0].id
            code=compile(ast.Module(body=[gate],type_ignores=[]),str(paths[0]),'exec')
            with self.assertRaisesRegex(SystemExit,r'\[e\] diagnose = true'):
                exec(code,{dataset_name:{'NVe':1.,'Ne':1.}})
            exec(code,{dataset_name:{'Ve':1.}})
            recover_lines=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call)
                           and isinstance(n.func,ast.Name)
                           and n.func.id=='recover_target_ion_velocity']
            self.assertLess(gate.lineno,min(recover_lines))
            checked+=1
        self.assertGreaterEqual(checked,2)

if __name__=='__main__':unittest.main()
