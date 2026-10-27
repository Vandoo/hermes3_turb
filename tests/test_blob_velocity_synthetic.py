"""Synthetic-blob regression test for the three velocity estimators.

The point of this file: the blob module had no ground truth. Every estimator
was checked against the *other* estimators on real data, where all three can
be wrong together. Here Gaussian filaments are advected at a velocity that is
known exactly, sampled on a grid and cadence matching the MAST-U SXD OMP
slice, and each estimator is asked to recover it.

This is what exposed the matched-z cross-correlation bug: with ``v_b = 0``
the 1-D CCF was fine, but switching on a realistic binormal drift (which is
10-100x the radial one in a real SOL) made it return 113 m/s against a true
400 m/s *while reporting ``resolved=True``* — a silent 3.5x under-estimate in
exactly the direction of the "simulated blobs are too slow" puzzle. Keep the
``v_b != 0`` cases.

Run with ``pytest hermes3_turb/tests/`` (or as a script).
"""

import numpy as np
import pytest

import hermes3_turb as h3t

# grid and cadence of the SXD OMP slice
NX, NZ, NT = 80, 81, 40
DR = 0.28e-3            # radial cell [m]
B_PERIOD = 0.30         # toroidal arc held by the slice [m]
IXSEP = 10
R = (np.arange(NX) - IXSEP) * DR
B = np.arange(NZ) * B_PERIOD / NZ
N0 = 2e18 * np.exp(-R / 8e-3)


def make_blobs(v_r, v_b, dt, dw_r=1.5e-3, dw_b=12e-3, nblob=8, noise=0.05,
               seed=3):
    """Density field (t,x,z) holding filaments advected at exactly (v_r, v_b)."""
    g = np.random.default_rng(seed)
    t = np.arange(NT) * dt
    n = np.tile(N0[None, :, None], (NT, 1, NZ)).astype(float)
    for _ in range(nblob):
        t0 = g.uniform(-5, NT) * dt
        r0 = g.uniform(-2e-3, 4e-3)
        b0 = g.uniform(0, B_PERIOD)
        amp = g.uniform(0.6, 1.4)
        for i in range(NT):
            rc = r0 + v_r * (t[i] - t0)
            bc = b0 + v_b * (t[i] - t0)
            db = (B[None, :] - bc + 0.5 * B_PERIOD) % B_PERIOD - 0.5 * B_PERIOD
            n[i] += amp * N0[:, None] * np.exp(
                -((R[:, None] - rc) / dw_r) ** 2 - (db / dw_b) ** 2)
    n *= 1.0 + noise * g.standard_normal(n.shape)
    return t, n


@pytest.mark.parametrize("v_b", [0.0, 6000.0, 20000.0])
def test_estimators_recover_known_velocity(v_b):
    v_r, dt = 400.0, 0.69e-6
    t, n = make_blobs(v_r, v_b, dt)
    f, _, _ = h3t.fluctuation_field(n, normalise="sigma")
    xs = slice(IXSEP, NX - 2)

    # (1) tracking
    frames = h3t.detect_blobs(f, R, B, B_PERIOD, t=t, threshold=2.5, x_slice=xs)
    tracks = h3t.track_blobs(frames, B_PERIOD, max_dr=3000 * dt,
                             max_db=40000 * dt, max_gap=1, min_length=4, dt=dt)
    st = h3t.velocity_statistics(tracks, b_period=B_PERIOD, r_grid=R,
                                 min_length=4)
    assert st["n_tracks"] >= 3
    assert st["v_r_median"] == pytest.approx(v_r, rel=0.25)
    if v_b:
        assert np.nanmedian(st["v_b"]) == pytest.approx(v_b, rel=0.1)

    # (2) 2-D cross-correlation. MUST scan the binormal shift: the matched-z
    # version fails outright for v_b != 0.
    ccf = h3t.ccf_radial_velocity(f, R, t, x_ref=IXSEP + 10,
                                  x_offsets=(4, 8, 12), b=B,
                                  b_period=B_PERIOD, binormal_scan=True)
    ok = np.asarray(ccf["resolved"], dtype=bool)
    assert ok.any(), "no resolved offsets even with the binormal scan"
    assert np.median(ccf["v_r"][ok]) == pytest.approx(v_r, rel=0.3)
    if v_b:
        assert np.median(ccf["v_b"][ok]) == pytest.approx(v_b, rel=0.1)

    # (3) conditional ExB, fed the exact advecting field
    cond = h3t.conditional_exb_velocity(np.full_like(n, v_r), f,
                                        threshold=2.5, x_slice=xs)
    assert cond["v_mean"] == pytest.approx(v_r, rel=1e-6)


def test_matched_z_ccf_is_biased_low_when_blobs_drift():
    """The bug this file exists for. Keep it as documentation."""
    v_r, v_b, dt = 400.0, 6000.0, 5.0e-6
    t, n = make_blobs(v_r, v_b, dt, dw_r=1e-3)
    f, _, _ = h3t.fluctuation_field(n, normalise="sigma")
    old = h3t.ccf_radial_velocity(f, R, t, x_ref=IXSEP + 10,
                                  x_offsets=(4, 8, 12), binormal_scan=False)
    new = h3t.ccf_radial_velocity(f, R, t, x_ref=IXSEP + 10,
                                  x_offsets=(4, 8, 12), b=B,
                                  b_period=B_PERIOD, binormal_scan=True)
    ok_old = np.asarray(old["resolved"], dtype=bool)
    ok_new = np.asarray(new["resolved"], dtype=bool)
    if ok_old.any():
        # it does not merely fail: it reports a plausible, too-small number
        assert np.median(old["v_r"][ok_old]) < 0.5 * v_r
    assert np.median(new["v_r"][ok_new]) == pytest.approx(v_r, rel=0.3)


def test_turbulent_flux_is_outward_for_outward_blobs():
    """<dn dv_r> must be outward when the blobs carry the outward velocity.

    NB the advecting field has to FLUCTUATE with the blobs. Handing
    ``turbulent_flux_slice`` a uniform v_r gives dv_r = 0, so the product is
    zero up to catastrophic cancellation on a 1e18 background — the first
    version of this test did exactly that and the "flux" came back as float
    noise of either sign.
    """
    v_r, dt = 400.0, 0.69e-6
    t, n = make_blobs(v_r, 6000.0, dt)
    f, _, _ = h3t.fluctuation_field(n, normalise="sigma")
    v = np.where(f > 1.0, v_r, 0.0)        # velocity lives in the structures
    Gt = h3t.turbulent_flux_slice(n, v)
    xs = slice(IXSEP, NX - 2)
    assert (Gt[xs] > 0).mean() > 0.9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_flux_closure_brackets_unity():
    """Mancini 2026 PPCF eq (3) vs eq (5): the two flux estimates must agree.

    Eq (5) does not state whether ``<n>_fil`` is the total density or the
    excess, and the two differ by O(1) for ``dn/n ~ 1`` blobs — so the test
    is that the truth is BRACKETED by the two variants, not that either hits
    1 on its own. On synthetic blobs with an exactly known flux the pair comes
    out 1.42 (total n) / 0.60 (excess), and ``f_p = 0.035`` against the 0.059
    - 0.072 that paper reports for TCV.
    """
    v_r, dt = 400.0, 0.69e-6
    t, n = make_blobs(v_r, 6000.0, dt)
    f, _, _ = h3t.fluctuation_field(n, normalise="sigma")
    v = np.where(f > 1.0, v_r, 0.0)
    xs = slice(IXSEP, NX - 2)
    out = h3t.blob_flux_closure(n, v, f, R, B, B_PERIOD, threshold=2.5,
                                x_slice=xs, verbose=False)
    assert 0.0 < out["f_p"] < 0.3
    assert out["ratio_dn"] < 1.0 < out["ratio_n"], (
        f"unity not bracketed: {out['ratio_dn']:.3f} .. {out['ratio_n']:.3f}")
    assert out["v_fil"] == pytest.approx(v_r, rel=1e-6)


def test_regimes_reproduce_myra_and_tsui():
    """Pin the published normalisations and the TCV reproduction.

    Myra, Russell & D'Ippolito, PoP 13, 112502 (2006) eqs (28)-(29) and the
    tabulated form in Tsui et al., PoP 25, 072506 (2018) eqs (1.1)-(1.9).
    Both PDFs are under Fusion/Filament_blob/.
    """
    # a* = L^(2/5) rho_s^(4/5) / R^(1/5) exactly (Myra eq 29 / Tsui eq 1.2)
    o = h3t.blob_velocity_regimes(Te_eV=20, Ti_eV=20, L_par=10.0, R=0.88,
                                  B=1.4, ne=1e19, delta=[7e-3], verbose=False)
    a_ref = 10.0 ** 0.4 * o["rho_s"] ** 0.8 / 0.88 ** 0.2
    assert o["a_star"] == pytest.approx(a_ref, rel=1e-12)
    # v* = c_s (2 L rho_s^2 / R^3)^(1/5) (Tsui eq 1.1)
    v_ref = o["c_s"] * (2.0 * 10.0 * o["rho_s"] ** 2 / 0.88 ** 3) ** 0.2
    assert o["v_star"] == pytest.approx(v_ref, rel=1e-12)

    # TCV: a_b = 3-11 mm must map onto a_hat ~ 0.3-1.5 (paper quotes 0.3-1.1)
    o = h3t.blob_velocity_regimes(Te_eV=20, Ti_eV=20, L_par=10.0, R=0.88,
                                  B=1.4, ne=1e19, delta=[3e-3, 11e-3],
                                  verbose=False)
    assert 0.25 < o["a_hat"][0] < 0.6
    assert 1.0 < o["a_hat"][1] < 1.8

    # ...and at TCV's higher-density conditions the whole range is RB, as the
    # paper reports ("most filaments populate the resistive-ballooning regime")
    o = h3t.blob_velocity_regimes(Te_eV=15, Ti_eV=15, L_par=10.0, R=0.88,
                                  B=1.4, ne=5e19,
                                  delta=[3e-3, 7e-3, 11e-3], verbose=False)
    assert o["Lambda"] > 1.0
    assert o["regime"][:2] == ["RB", "RB"]
    # envelope must sit at or above the measured 0.5-2 km/s band
    assert np.all(o["v_envelope"] > 2.0e3)

    # eps_x ~ 1/f_x_max (Tsui eq 1.5), and it acts ONLY on the Ci branch
    a = h3t.blob_velocity_regimes(Te_eV=30, Ti_eV=30, L_par=20.0, R=1.4,
                                  B=0.45, delta=[5e-3], f_x_max=5.5,
                                  verbose=False)
    b = h3t.blob_velocity_regimes(Te_eV=30, Ti_eV=30, L_par=20.0, R=1.4,
                                  B=0.45, delta=[5e-3], eps_x=0.3,
                                  verbose=False)
    assert a["eps_x"] == pytest.approx(1 / 5.5)
    assert a["v_Ci"][0] < b["v_Ci"][0]
    assert a["v_RB"][0] == pytest.approx(b["v_RB"][0])   # RB carries no eps_x
