#!/usr/bin/env python3
"""
Poloidal flux expansion f_x of a BOUT++ grid, per divertor leg.

Needed for `blobs.blob_velocity_regimes`: Tsui et al 2018 (PoP 25, 072506)
eq. (1.5) sets the field-line fanning parameter `eps_x ~ 1/f_x_max` from "the
maximum POLOIDAL flux expansion typically found for those shapes WHEN
AVERAGED ACROSS THE SOL", and quotes ~0.3 for diverted / ~0.5 for limited
plasmas. In the Ci (connected ideal-interchange) regime eps_x multiplies the
blob velocity envelope directly, so for a Super-X it is not a detail.

f_x = (R Bp)_upstream / (R Bp)(theta), because dpsi = R Bp dr_perp so the
perpendicular spacing of flux surfaces goes as 1/(R Bp). And **sqrt(g11) is
R*Bp exactly** in the BOUT++ tokamak metric -- verified here against the
independent route R = sqrt(g_33), Bp = B*sqrt(1 - g_23^2/(g_22 g_33)), which
agrees to 1e-6 -- so this needs only the DUMP, no grid file and no geometry.

TWO NUMBERS, AND ONLY ONE OF THEM IS TSUI'S. `R Bp -> 0` at the X-point, so
the maximum ANYWHERE along a leg diverges and is set by how finely the grid
resolves the X-point cell (7.94 on the nx244 SXD mesh). That is not what
eps_x ~ 0.3 can mean. Use the value **at the target** -- the standard
divertor-community "flux expansion", and the one that reproduces Tsui's
reference magnitude.

Measured on sxd_MU/from_archer_6kW (nx244, connected double null, up-down
symmetric to 1.3e-6):
    outer targets  f_x = 4.64 SOL-averaged (5.47 at the separatrix)  -> eps_x 0.216
    inner targets  f_x = 3.56                                        -> eps_x 0.281
    4-leg mean     f_x = 4.10                                        -> eps_x 0.244
    X-point max    f_x = 7.94  <- grid-dependent, NOT eps_x
The separatrix 5.47 independently reproduces the f_x ~ 5.5 measured in the
lambda_q work from the OMP-mapped cell widths.

NB `jyseps*` are the INCLUSIVE last index of the preceding band, so they need
`+1` when used as a slice END as well as as a slice start. Getting that wrong
gave legs of 19 and 20 rows and made a symmetric grid look asymmetric.
"""
import numpy as np
from netCDF4 import Dataset
p = ("/home/eric/Downloads/BOUT_code/work_dir/github_tools/debug/pit_workdir/"
     "sxd_MU/from_archer_6kW/data/408/boutdata.nc")
d = Dataset(p)
RBp = np.sqrt(np.asarray(d["g11"][:], float))     # = R*Bp exactly (verified)
dy  = np.asarray(d["dy"][:], float)
g22 = np.asarray(d["g_22"][:], float)
rho = float(d["rho_s0"][:])
ixs=int(d["ixseps1"][:]); MXG=int(d["MXG"][:]); MYG=int(d["MYG"][:])
ny=int(d["ny"][:]); nyi=int(d["ny_inner"][:]); nyf=RBp.shape[1]; nx=RBp.shape[0]
g = MYG if nyf>ny else 0
j11,j21,j12,j22=[int(d[k][:]) for k in ("jyseps1_1","jyseps2_1","jyseps1_2","jyseps2_2")]
d.close()

# jyseps* are INCLUSIVE last indices of the preceding band -> +1 as a slice end
A0 = g                      # region A domain start
B0 = nyi + 3*g              # region B domain start
legs = {
  "inner_lower": (slice(A0,        j11+g+1),   (j11+j21)//2+g,   "lo"),
  "inner_upper": (slice(j21+g+1,   nyi+g),   (j11+j21)//2+g,   "hi"),
  "outer_upper": (slice(B0,        j12+3*g+1), (j12+j22)//2+3*g, "lo"),
  "outer_lower": (slice(j22+3*g+1, ny+3*g),    (j12+j22)//2+3*g, "hi"),
}
sol = slice(ixs, nx-MXG)
print(f"OMP y={(j12+j22)//2+3*g}  IMP y={(j11+j21)//2+g}   "
      f"SOL x {ixs}..{nx-MXG-1} ({nx-MXG-ixs} cells)")
print(f"{'leg':>12}{'y rows':>10}{'n':>4}{'f_x target':>12}{'f_x max':>10}"
      f"{'s/L of max':>12}{'L_leg [m]':>11}")
res={}
for name,(ysl,yref,end) in legs.items():
    ys=np.arange(ysl.start, ysl.stop)
    fx = RBp[sol, yref][:,None] / RBp[sol][:, ys]
    it = -1 if end=="hi" else 0
    fx_t = float(np.mean(fx[:,it]))
    fx_m = float(np.mean(fx.max(axis=1)))
    # s/L measured FROM THE TARGET
    im = np.argmax(fx,axis=1)
    sl = im/(len(ys)-1) if end=="hi" else 1.0-im/(len(ys)-1)
    L = float(np.mean((dy[sol][:,ys]*np.sqrt(g22[sol][:,ys])*rho).sum(axis=1)))
    print(f"{name:>12}{f'{ys[0]}-{ys[-1]}':>10}{len(ys):>4}{fx_t:>12.2f}"
          f"{fx_m:>10.2f}{float(np.mean(sl)):>12.2f}{L:>11.2f}")
    res[name]=(fx_t,fx_m,fx,it)

print("\n### symmetry check on R*Bp (mirror the poloidal axis)")
for tag, a, b in (("outer", "outer_upper", "outer_lower"),
                  ("inner", "inner_lower", "inner_upper")):
    A = RBp[sol, legs[a][0]]
    B_ = RBp[sol, legs[b][0]][:, ::-1]
    if A.shape[1] != B_.shape[1]:
        print(f"  {tag}: lengths {A.shape[1]} vs {B_.shape[1]} -> not symmetric")
    else:
        print(f"  {tag}: n={A.shape[1]}  max rel diff = "
              f"{np.abs(A-B_).max()/np.abs(A).max():.2e}")

print("\n### f_x at target vs radius (outer legs)")
fx=res['outer_lower'][2]; it=res['outer_lower'][3]
rr=np.arange(nx)[sol]
print("   " + "  ".join(f"x+{rr[k]-ixs}:{fx[k,it]:.2f}"
                        for k in (0,2,5,10,20,40,80,120,160) if k<fx.shape[0]))

tgt4=np.mean([v[0] for v in res.values()])
tgtO=np.mean([res[k][0] for k in ("outer_upper","outer_lower")])
mx4 =np.mean([v[1] for v in res.values()])
print(f"\n===== f_x,max FOR THE SXD GRID =====")
print(f"  at the TARGET, 4-leg mean, SOL-averaged : {tgt4:.2f}  -> eps_x = {1/tgt4:.3f}")
print(f"  at the TARGET, outer legs only          : {tgtO:.2f}  -> eps_x = {1/tgtO:.3f}")
print(f"  max ANYWHERE in the leg (X-point)       : {mx4:.2f}  -> eps_x = {1/mx4:.3f}")
print(f"  separatrix value at the outer target    : {fx[0,it]:.2f}")
print(f"  Tsui's diverted reference               : 3.3   -> eps_x = 0.30")
