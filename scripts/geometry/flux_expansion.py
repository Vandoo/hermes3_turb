#!/usr/bin/env python3
"""Compute poloidal flux expansion along each divertor leg.

    f_x(theta) = (R Bp)_upstream / (R Bp)(theta)

For the BOUT++ tokamak metric, sqrt(g11) = R Bp. The calculation uses that
metric from the dump and reports target expansion, the maximum along the
leg, and parallel leg length. Target and X-point maxima are distinct: the
latter depends on resolution near Bp=0.

Tsui et al. 2018, Phys. Plasmas 25, 072506, motivates the field-line fanning
parameter used in blob-velocity estimates. jyseps* indices are inclusive;
half-open Python region slices include the endpoint with +1.
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
