"""
JAX port of the Lake-M3 vertical 1D lake model.

SCOPE: this module ports the full chain of modules `run_wq_model()`
actually invokes (per timestep, in order):

    dynamic kd_light (from DOC/POC) -> OC river loading
      -> heating_module -> ice_module -> boundary_module_oxygen (O2 only)
      -> eddy_diffusivity_hendersonSellers
      -> diffusion_module_dAdK_v2 (temperature + O2 + DOCr + DOCl)
      -> advection_diffusion_module_mylake (POCr/POCl settling)
      -> mixing_module_minlake_RL (temperature + all 5 WQ tracers)
      -> convection_module (temperature only)
      -> prodcons_module_woDOCL (production/consumption reactions)
      -> hydrologic outflow (DOCr/DOCl/POCr/POCl only)

Phase 1 (temperature only: `temperature_step`/`run_temperature_model`) and
Phase 2 (full water quality: `full_step`/`run_full_model`) both live here;
Phase 1 is kept as-is (with its `kd_light` held constant) since it's a
useful, faster, already-validated standalone engine when WQ state isn't
needed. Everything is written in pure JAX (no in-place mutation, no
Python-level branching on traced values) so a whole multi-year
simulation can be `jax.jit`-compiled and differentiated end-to-end with
`jax.grad` / `jax.vjp`.

Known, documented simplifications relative to the numpy reference:

1. Iterative/while-loop constructs in the original code (the bulk
   sensible/latent heat Monin-Obukhov stability iteration, the wind
   mixing search loop, the convective-overturn bubble merge) do not have
   a statically-known number of iterations, which JAX's reverse-mode
   autodiff cannot differentiate through (`lax.while_loop` only supports
   forward-mode AD). Each is replaced with a fixed number of iterations
   (`lax.fori_loop`/`lax.scan`, unrolled at trace time) that is generous
   enough to reach the same fixed point in all realistic conditions.
   These bounds are exposed as keyword arguments so they can be
   tightened (for speed) or loosened (for extreme forcing) and are
   documented at each call site below.
2. `ice_module`'s `supercooled` state is threaded through the original
   code but never actually read back anywhere -- it is dropped from the
   JAX state here with no effect on results.
3. Several dead parameters/branches in the numpy reference that have no
   effect on any returned value (confirmed by reading, not assumed) are
   dropped here rather than faithfully-but-pointlessly reproduced:
   `boundary_module_oxygen`'s and `prodcons_module_woDOCL`'s `H`/`albedo`
   diagnostic recomputation (`prodcons_step` computes its own `H`,
   matching the formula, from the ice state), `da_dz`/`dv_da`/`sed_flux`/
   `do_consumption_old` in the O2 boundary flux (superseded by
   `sed_oxygen_loss` before being used), `IP`/`IP_m`/`growth`/`temp` in
   the production-consumption reaction (computed but never read), and
   `sw_to_par`'s and `LIGHTUSEBYPHOTOS`'s caller-supplied values (the
   reference silently shadows both with hardcoded constants inside
   `solve_mprk`).
4. `prodcons_module_woDOCL`'s reaction step loops `for dep in range(0,
   nx-1)` in the reference -- the bottom layer (`nx-1`) is never reached
   by the reaction step. This is very likely an off-by-one bug, but it's
   reproduced faithfully here (`prodcons_step` masks the last layer to a
   no-op) rather than silently "fixed", since the goal is a numerically
   matching port.
5. `eddy_diffusivity_hendersonSellers` is called with a hardcoded
   latitude (43.100948) rather than the lake's configured latitude, in
   every `diffusion_method` branch of `run_wq_model` (line 4572) --
   reproduced here rather than "fixed" to the real latitude.
6. `eddy_diffusivity_hendersonSellers`'s Richardson-number term divides
   by `exp(-2*k_star*depth)`, which underflows to an exact float64 zero
   for deep/weakly-mixed columns (routine here: Ravn reaches 30+ m and
   winter winds push `k_star` into the tens). The *value* is unaffected
   (`Ri` legitimately saturates at +inf, which correctly drives `kz` to
   ~0), but the *gradient* is not: differentiating a ratio through an
   exact-zero denominator is a `0 * inf = NaN`, and this genuinely
   reproduced (not hypothetical -- see below) as a NaN partway through a
   `jax.grad` over a multi-month Ravn run. Fixed by clamping the decay
   exponent (`min(k_star*depth, 30)`) so the denominator stays a tiny
   but finite double; this changes `kz` by far less than float64
   noise at the depths/timesteps where it triggers, so it is a
   gradient-safety fix, not a value change.
7. `mixing_step`'s wind-mixing energy input, `KE0`, takes
   `sqrt(max(tau**3 / calc_dens(u[0]), 0.0))` where `tau` is the wind
   stress. On a genuinely calm hour (`Uw == 0.0` exactly -- routine in
   real hourly meteorology, not a corner case), `tau**3` is exactly 0,
   so the `sqrt` argument is exactly 0. The value is fine (no wind -> no
   mixing energy), but `sqrt`'s local derivative at exactly 0 is
   infinite, and `inf * 0 = NaN` regardless of the upstream derivative
   also being zero -- the same pattern as item 6, this time found via a
   NaN gradient at a real calm-wind Ravn timestep (~step 7961 of the
   2020-01-01 run). Fixed the same way: floor the `sqrt` argument at
   `1e-30` instead of `0.0` (changes `KE0` by ~1e-15 in the affected
   case -- physically negligible).

VALIDATION NOTES (see the project's validation scripts/conversation for
full detail): every function in this module (`calc_dens`,
`eddy_diffusivity_hendersonSellers`, `heating_module`, `ice_module`,
`diffusion_step(_wq)`, `mixing_step`, `convection_step`,
`oxygen_boundary_step`, `poc_settling_step`, `prodcons_step`,
`do_sat_calc`/`k_vachon`/`k600_to_kgas_o2`) matches its numpy counterpart
to floating-point precision (1e-8 to 1e-15) when tested in isolation
with identical inputs, and a single full `full_step` matches the
reference's per-iteration state to ~1e-8 given identical starting state.
Over a multi-day *coupled* run, however, the trajectories diverge by a
few percent (temperature/O2/DOC) to a few tens of percent (POC) of the
reference's typical magnitude. Tracing this down (see `mixing_step`'s
`zb` search) shows the cause: `mixing_step` picks the shallowest unstable
interface via `jnp.argmax` on a boolean mask, and in a weakly-stratified
water column (as this Ravn winter case mostly is) many interfaces sit
within floating-point noise of each other, so which one gets selected as
`zb` is genuinely sensitive to summation order -- `jnp.sum` (XLA) and
`np.sum` (numpy) need not accumulate in the same order, even though both
are equally valid floating-point reductions of the same formula. Since a
full-water-column redistribution happens whenever `zb` changes, tiny
(~1e-7) upstream differences occasionally cascade into a materially
different mixed profile within a handful of steps -- and the reference
model shows the same step-to-step volatility in its own `thermo_dep`
diagnostic, confirming this is a property of the physics (a hard,
discontinuous branch decision) rather than a discrepancy between the two
implementations. `jax.grad` remains well-defined and finite through this
-- `argmax`/`jnp.where` branch selection just contributes zero gradient
at the selection itself, which is the standard, expected subgradient
behavior for this kind of discrete decision.

Gradient horizon, revised: an earlier version of this module documented
a NaN gradient appearing somewhere between ~7500-8000 hourly steps into
a continuous `jax.grad` over the real Ravn forcing, and attributed it to
"accumulated sensitivity through chaotic mixing-branch flips" as an
inherent, unfixable property of the physics. Chasing it down properly
(prompted by needing a real fix rather than a workaround) found this
was wrong: the NaN at that specific horizon was item 7 above (a calm
wind hour), a concrete, fixable bug, not chaos-driven blowup. With items
6 and 7 both fixed, isolated ~2000-step windows spot-checked across the
*entire* multi-year Ravn record (steps 0, ~6000-8000, ~15000, ~28000,
~40000) all give finite gradients. That said, this is a spot-check, not
an exhaustive proof for every possible timestep/parameter combination --
there is no guarantee a third such singularity doesn't exist somewhere
unexplored. For that reason, `calibrate_M3_jax.py` still differentiates
via fixed-size chunked/truncated backpropagation through time (see its
module docstring) rather than one continuous multi-year `jax.grad`: it
costs nothing when gradients would have been fine anyway, and it caps
the blast radius (and the memory) if a future, undiscovered edge case
does turn up.

Everything below uses `jax.numpy` and is safe to `jax.jit` /
`jax.vmap` / `jax.grad`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax

# ---------------------------------------------------------------------------
# Basic physics: density, radiation, bulk fluxes
# ---------------------------------------------------------------------------


def calc_dens(wtemp):
    """Water density from temperature (deg C). Vectorized, matches
    `calc_dens` in processBased_lakeModel_functions.py line 29."""
    wtemp = jnp.asarray(wtemp)
    return (
        999.842594
        + 6.793952e-2 * wtemp
        - 9.095290e-3 * wtemp ** 2
        + 1.001685e-4 * wtemp ** 3
        - 1.120083e-6 * wtemp ** 4
        + 6.536336e-9 * wtemp ** 5
    )


def longwave(cc, sigma, Tair, ea, emissivity, Jlw):
    """Incoming longwave radiation. Matches `longwave()` line 664."""
    Tair_k = Tair + 273.15
    p = 1.33 * ea / Tair_k
    Ea = 1.24 * (1 + 0.17 * cc ** 2) * p ** (1 / 7)
    return emissivity * Ea * sigma * Tair_k ** 4


def backscattering(emissivity, sigma, Twater, eps):
    """Outgoing longwave (backscattering). Matches `backscattering()` line 671."""
    Twater_k = Twater + 273.15
    return -1.0 * (eps * sigma * Twater_k ** 4)


def _psim(zeta):
    """Stability function for momentum, vectorized over `zeta`.
    Matches `PSIM()` line 680, rewritten as a `jnp.where` cascade so it
    is safe to trace/differentiate (no Python `if` on traced values).

    NOTE: every branch is fed a *clamped* copy of `zeta` so that its
    formula stays well-defined (no negative base to a fractional power,
    no division by exactly zero) even when that branch is not the one
    selected by the outer `jnp.where`. This matters because `jnp.where`'s
    gradient multiplies each branch's local Jacobian by 0/1 *after*
    computing it -- `0 * nan == nan`, so an unselected branch that is
    merely finite-valued but has a NaN/Inf local derivative would still
    poison the overall gradient."""
    zeta_neg = jnp.minimum(zeta, 0.0)  # keeps 1 - 16*zeta_neg >= 1, always a safe base
    X = (1 - 16 * zeta_neg) ** 0.25
    unstable = 2 * jnp.log((1 + X) / 2) + jnp.log((1 + X * X) / 2) - 2 * jnp.arctan(X) + jnp.pi / 2
    stable_far = jnp.log(jnp.abs(zeta) + 1e-12) - 0.76 * zeta - 12.093
    stable_mid = 0.5 / (zeta * zeta + 1e-12) - 4.25 / (zeta + 1e-12) - 7.0 * jnp.log(jnp.abs(zeta) + 1e-12) - 0.852
    stable_near = -5 * zeta
    stable = jnp.where(zeta > 10.0, stable_far, jnp.where(zeta > 0.5, stable_mid, stable_near))
    return jnp.where(zeta < 0.0, unstable, jnp.where(zeta > 0.0, stable, 0.0))


def _psite(zeta):
    """Stability function for sensible/latent heat. Matches `PSITE()` line 698.
    See the note in `_psim` about clamping every branch's input."""
    zeta_neg = jnp.minimum(zeta, 0.0)
    X = (1 - 16 * zeta_neg) ** 0.25
    unstable = 2 * jnp.log((1 + X * X) / 2)
    stable_far = jnp.log(jnp.abs(zeta) + 1e-12) - 0.76 * zeta - 12.093
    stable_mid = 0.5 / (zeta * zeta + 1e-12) - 4.25 / (zeta + 1e-12) - 7.0 * jnp.log(jnp.abs(zeta) + 1e-12) - 0.852
    stable_near = -5 * zeta
    stable = jnp.where(zeta > 10.0, stable_far, jnp.where(zeta > 0.5, stable_mid, stable_near))
    return jnp.where(zeta < 0.0, unstable, jnp.where(zeta > 0.0, stable, 0.0))


def bulk_fluxes(Tair, Twater, Uw, pa, RH, Cd=0.0013, z0_iters=50, l_iters=20):
    """Sensible + latent heat flux via the Monin-Obukhov bulk-transfer
    iteration. This single function replaces the near-duplicate
    `sensible()` (line 722) and `latent()` (line 890) -- both compute
    identical intermediate quantities (H and E) and only differ in
    which one they return, so we compute both once.

    The two `while` loops in the original (roughness-length iteration,
    then Monin-Obukhov length iteration capped at 20 in the reference
    too) are replaced with fixed-count `lax.fori_loop`s -- `z0_iters`
    and `l_iters` -- since JAX cannot reverse-mode-differentiate
    through a data-dependent-length `while_loop`. Both are contraction
    mappings that converge in a handful of iterations for realistic
    meteorology; the defaults are generous.

    Returns (sensible, latent), matching the sign convention of the
    original (`sensible()`/`latent()` both return the negative of the
    internally-computed H / E).
    """
    const_cp = 1005.0
    k_von = 0.41
    g = 9.81

    U_Z = jnp.where(Uw <= 0, 1e-3, Uw)
    T = jnp.where(Tair == 0, 1e-6, Tair)
    T0 = jnp.where(Twater == 0, 1e-6, Twater)
    Rh = RH
    p = pa / 100.0
    z = 2.0

    e_s = 6.11 * jnp.exp(17.27 * T / (237.3 + T))
    e_a = Rh * e_s / 100.0
    q_z = 0.622 * e_a / p
    e_sat = 6.11 * jnp.exp(17.27 * T0 / (237.3 + T0))
    q_s = 0.622 * e_sat / p
    L_v = 2.501e6 - 2370 * T0
    R_a = 287 * (1 + 0.608 * q_z)
    rho_a = 100 * p / (R_a * (T + 273.16))
    v = (1.0 / rho_a) * (4.94e-8 * T + 1.7184e-5)
    T_v = (T + 273.16) * (1 + 0.61 * q_z)

    u_star0 = U_Z * jnp.sqrt(0.00104 + 0.0015 / (1 + jnp.exp((-U_Z + 12.5) / 1.56)))
    u_star0 = jnp.where(u_star0 == 0, 1e-6, u_star0)

    def z0_body(_, carry):
        u_star, z_0 = carry
        u_star_new = k_von * U_Z / jnp.log(z / z_0)
        z_0_new = (Cd * u_star_new ** 2 / g) + (0.11 * v / u_star_new)
        return (u_star_new, z_0_new)

    z_0_init = (Cd * u_star0 ** 2 / g) + (0.11 * v / u_star0)
    u_star, z_0 = lax.fori_loop(0, z0_iters, z0_body, (u_star0, z_0_init))

    C_DN = (u_star ** 2) / (U_Z ** 2)
    Re_star = u_star * z_0 / v
    z_T = z_0 * jnp.exp(-2.67 * Re_star ** 0.25 + 2.57)
    z_E = z_T
    C_HN = k_von * jnp.sqrt(C_DN) / jnp.log(z / z_T)
    C_EN = k_von * jnp.sqrt(C_DN) / jnp.log(z / z_E)

    H_init = rho_a * const_cp * C_HN * U_Z * (T0 - T)
    E_init = rho_a * L_v * C_EN * U_Z * (q_s - q_z)
    L_init = (-rho_a * u_star ** 3 * T_v) / (
        k_von * g * (H_init / const_cp + 0.61 * E_init * (T + 273.16) / L_v)
    )

    def l_body(_, carry):
        u_star, L, H, E = carry
        zeta = jnp.clip(z / L, -15.0, 15.0)
        psim = _psim(zeta)
        psit = _psite(zeta)
        psie = _psite(zeta)
        z_0_ = (Cd * u_star ** 2 / g) + (0.11 * v / u_star)
        Re_star_ = u_star * z_0_ / v
        z_T_ = z_0_ * jnp.exp(-2.67 * Re_star_ ** 0.25 + 2.57)
        z_E_ = z_T_
        C_D = k_von * k_von / (jnp.log(z / z_0_) - psim) ** 2
        C_H = k_von * jnp.sqrt(C_D) / (jnp.log(z / z_T_) - psit)
        C_E = k_von * jnp.sqrt(C_D) / (jnp.log(z / z_E_) - psie)
        H_new = rho_a * const_cp * C_H * U_Z * (T0 - T)
        E_new = rho_a * L_v * C_E * U_Z * (q_s - q_z)
        u_star_new = jnp.sqrt(C_D * U_Z ** 2)
        L_new = (-rho_a * u_star_new ** 3 * T_v) / (
            k_von * g * (H_new / const_cp + 0.61 * E_new * (T + 273.16) / L_v)
        )
        return (u_star_new, L_new, H_new, E_new)

    _, _, H, E = lax.fori_loop(0, l_iters, l_body, (u_star, L_init, H_init, E_init))

    return -H, -E


# ---------------------------------------------------------------------------
# Eddy diffusivity (Henderson-Sellers), matches `eddy_diffusivity_hendersonSellers`
# line 67. `area` is accepted for API symmetry but (as in the original) is
# unused by this closure formulation.
# ---------------------------------------------------------------------------


def eddy_diffusivity_hendersonSellers(rho, depth, g, rho_0, ice, Uw, latitude, T0, kzn_prev, Cd, km, weight_kz,
                                       return_ri=False):
    k = 0.4
    Pr = 1.0
    z0 = 0.0002
    xi = 1.0 / 3.0

    U2 = jnp.maximum(Uw * jnp.log((2 - 1e-5) / z0) / jnp.log((10 - 1e-5) / z0), 1e-2)

    Cd_calc = jnp.where(
        U2 < 2.2,
        1.08 * U2 ** (-0.15) * 1e-3,
        jnp.where(
            U2 < 5.0,
            (0.771 + 0.858 * U2 ** (-0.15)) * 1e-3,
            jnp.where(
                U2 < 8.0,
                (0.867 + 0.0667 * U2 ** (-0.15)) * 1e-3,
                jnp.where(
                    U2 < 25.0,
                    (1.2 + 0.025 * U2 ** (-0.15)) * 1e-3,
                    jnp.where(U2 < 50.0, 0.073 * U2 ** (-0.15) * 1e-3, Cd),
                ),
            ),
        ),
    )

    w_star = Cd_calc * U2
    k_star = 6.6 * jnp.sin(jnp.deg2rad(latitude)) ** 0.5 * U2 ** (-1.84)

    diff_rho = jnp.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy = jnp.concatenate([diff_rho, diff_rho[-1:]])
    buoy = jnp.maximum(buoy, 7e-5)

    # `k_star` can be an order-10 decay rate (it scales as U2**(-1.84), so it
    # blows up for the near-calm winds typical of a weakly-mixed winter
    # column), and `depth` reaches 30+ m here, so `k_star * depth` routinely
    # exceeds ~700 -- past the point where `exp(-2*k_star*depth)` underflows
    # to an *exact* float64 zero rather than a tiny-but-nonzero number. The
    # forward value of `Ri` is still perfectly well-defined then (it's just
    # +inf, which correctly drives kz to ~0 at depth via `1/(1+37*Ri**2)`),
    # but its *gradient* is not: d(Ri)/d(buoy) works out to
    # `0.5 * (1+X)**-0.5 * dX/d(buoy)` with `X = buoy * ... / exp(-2*k_star*depth)`.
    # At the exact-zero denominator, `X = inf` (so `(1+X)**-0.5 = 0`) while
    # `dX/d(buoy) = inf` (division by the exact zero) -- a `0 * inf = NaN`
    # that poisons the whole eddy-diffusivity Jacobian (confirmed: this is
    # what turns `jax.grad` NaN partway through a multi-year Ravn run, at
    # the first winter timestep whose column is deep and weakly stratified
    # enough to hit the `buoy` floor at every layer). Clamping the exponent
    # keeps `decay` a tiny-but-nonzero double (exp(-700) ~= 1e-304, still a
    # normal float64) so the division stays finite -- `Ri` becomes a large
    # finite number instead of +inf, and `kz` is unaffected to far beyond
    # float64 precision, since both give kz ~= 0 at these depths anyway.
    decay = jnp.exp(-jnp.minimum(k_star * depth, 30.0))
    Ri = (
        -1
        + jnp.sqrt(1 + 40 * (buoy * k ** 2 * depth ** 2) / (w_star ** 2 * decay ** 2))
    ) / 20

    kz = (k * w_star * depth) / (Pr * (1 + 37 * Ri ** 2)) * decay

    LST = T0[0]
    kz = jnp.where(LST > 4, kz * 1e2, jnp.where(LST > 0, kz * 1e4, kz * 0.0))

    weight = jnp.where(jnp.mean(kzn_prev) == 0.0, 1.0, weight_kz)
    kz = weight * kz + (1 - weight) * kzn_prev

    if return_ri:
        # `Ri` is always >= 0 by construction (`buoy` is floored above 0, so
        # the sqrt argument is always >= 1) -- a depth-resolved Richardson
        # number `run_M3_mcl_jax.py` can optionally fit its own stability
        # function against (`kz = K0*(1+alpha*Ri)**(-n)`, the classical
        # Munk-Anderson form) instead of directly correcting this
        # function's own `kz` output. Not part of the default return value
        # so every existing caller (this function's numeric-matching
        # contract with the reference model) is completely unaffected.
        return kz + km, Ri
    return kz + km


# ---------------------------------------------------------------------------
# Heating module: matches `heating_module` line 1065 (temperature-only path).
# ---------------------------------------------------------------------------


def heating_module(
    u, area, volume, depth, dt, dx, ice, Tair, CC, ea, Jsw, Jlw, Uw, Pa, RH,
    kd_light, Hi, Hs, rho_snow, kd_ice=0.7, kd_snow=0.9, rho_fw=1000.0,
    sigma=5.67e-8, eps=0.97, emissivity=0.97, p2=1.0, Cd=0.0013,
    sw_factor=1.0, at_factor=1.0, turb_factor=1.0, Hgeo=0.1,
):
    albedo = jnp.where(ice, 0.3, 0.1)
    IceSnowAttCoeff = jnp.where(
        ice, jnp.exp(-kd_ice * Hi) * jnp.exp(-kd_snow * (rho_fw / rho_snow) * Hs), 1.0
    )

    Tair_eff = Tair * at_factor
    Jsw_eff = Jsw * sw_factor

    sens, lat = bulk_fluxes(Tair_eff, u[0], Uw, Pa, RH, Cd=Cd)
    Q = (
        longwave(CC, sigma, Tair_eff, ea, emissivity, Jlw)
        + backscattering(emissivity, sigma, u[0], eps)
        + lat / turb_factor
        + sens / turb_factor
    )

    dens0 = calc_dens(u[0])
    u_new = u.at[0].add((Q * area[0]) / (4184 * dens0 * volume[0]) * dt)

    Qsw = kd_light * (1 - albedo) * Jsw_eff * jnp.exp(-kd_light * depth)
    u_new = u_new + (Qsw / (calc_dens(u) * 4184)) * dt

    dens_bot = calc_dens(u[-1])
    u_new = u_new.at[-1].add(Hgeo / (dens_bot * 4184 * dx) * dt)

    # Net surface heat flux (W/m^2) diagnostic: the non-solar terms already
    # summed in `Q` (net longwave + backscattering + sensible + latent),
    # plus the net (post-albedo) incident shortwave -- i.e. the standard
    # surface energy-balance total, evaluated before `Qsw` is spread out
    # with depth via Beer-Lambert attenuation for the actual heating below.
    # Not used elsewhere in the physics -- purely a diagnostic passed up
    # through `full_step`'s returned dict for external analysis/plotting
    # (see run_M3_mcl_jax.py's --target ri diagnostics).
    Q_net = Q + (1 - albedo) * Jsw_eff

    return u_new, IceSnowAttCoeff, Q_net


# ---------------------------------------------------------------------------
# Ice module: matches `ice_module` line 1762. All four mutually-exclusive
# branches (ice forms / persists+Tair>0 / persists+Tair<=0 / melts away /
# no ice) are evaluated and combined with `jnp.where` since the branch
# taken depends on traced state (ice flag, Hi, Tair).
# ---------------------------------------------------------------------------


def ice_module(
    u, dt, dx, area, Tair, CC, ea, Jsw, Jlw, Uw, Pa, RH, PP, IceSnowAttCoeff,
    ice, dt_iceon_avg, iceT, rho_snow, Hi, Hs, Hsi, eps, emissivity=0.97,
    sigma=5.67e-8, p2=1.0, Cd=0.0013, rho_new_snow=250.0, rho_ice=910.0,
    rho_fw=1000.0, Ice_min=0.1, Cw=4.18e6, L_ice=333500.0, meltP=1.0,
    K_ice=2.1,
):
    icep = jnp.maximum(dt_iceon_avg, dt / 86400)
    x = (dt / 86400) / icep
    iceT_new = iceT * (1 - x) + u[0] * x

    # ---- candidate A: ice starts forming this step -----------------------
    supercooled = u < 0
    initEnergy = jnp.sum(jnp.where(supercooled, (0 - u) * area * dx * Cw, 0.0))
    Hi_A = Ice_min + (initEnergy / (910 * L_ice)) / jnp.max(area)
    u_A = jnp.where(Hi_A >= 0, jnp.where(supercooled, 0.0, u), u)
    u_A = u_A.at[0].set(jnp.where(Hi_A >= 0, 0.0, u_A[0]))

    condA = (iceT_new <= 0) & (Hi < Ice_min) & (Tair <= 0) & jnp.logical_not(ice)

    # ---- candidate B: ice==True and Hi>=Ice_min (persists / grows / melts) --
    Q_surf = u[0] * Cw * dx
    u_B_base = u.at[0].set(0.0)
    Twater0 = 0.0  # aliasing in the reference makes Twater==u[0]==0 here

    sens0, lat0 = bulk_fluxes(Tair, Twater0, Uw, Pa, RH, Cd=Cd)
    flux_sum = (
        (1 - IceSnowAttCoeff) * Jsw
        + longwave(CC, sigma, Tair, ea, emissivity, Jlw)
        + backscattering(emissivity, sigma, Twater0, eps)
        + lat0
        + sens0
    )

    Hs_gt0 = Hs > 0
    # Tair > 0 sub-branch
    dHs_ifpos = -jnp.maximum(0.0, meltP * dt * flux_sum / (rho_fw * L_ice))
    Hi_ifpos = jnp.where(Hs + dHs_ifpos < 0, Hi + (Hs + dHs_ifpos) * (rho_fw / rho_ice), Hi)
    melt_ice = jnp.maximum(0.0, meltP * dt * flux_sum / (rho_ice * L_ice))
    dHs_ifzero = 0.0
    Hi_ifzero = Hi - melt_ice
    Hsi_ifzero = jnp.maximum(0.0, Hsi - melt_ice)

    Tice_B1 = 0.0
    dHs_B1 = jnp.where(Hs_gt0, dHs_ifpos, dHs_ifzero)
    Hi_B1 = jnp.where(Hs_gt0, Hi_ifpos, Hi_ifzero)
    Hsi_B1 = jnp.where(Hs_gt0, Hsi, Hsi_ifzero)
    dHsnew_B1 = 0.0

    # Tair <= 0 sub-branch
    K_snow = 2.22362 * (rho_snow / 1000) ** 1.885
    Hi_safe = jnp.where(Hi > 0, Hi, 1.0)
    p_ifpos = (K_ice / K_snow) * (((rho_fw / rho_snow) * Hs) / Hi_safe)
    dHsi_ifpos = jnp.maximum(0.0, Hi * (rho_ice / rho_fw - 1) + Hs)
    Hsi_ifpos = Hsi + dHsi_ifpos
    p_ifzero = 1.0 / (10 * Hi_safe)
    dHsi_ifzero = 0.0
    Hsi_ifzero2 = Hsi

    p_B2 = jnp.where(Hs_gt0, p_ifpos, p_ifzero)
    dHsi_B2 = jnp.where(Hs_gt0, dHsi_ifpos, dHsi_ifzero)
    Hsi_B2 = jnp.where(Hs_gt0, Hsi_ifpos, Hsi_ifzero2)

    Tice_B2 = Tair / (1 + p_B2)
    Hi_B2 = jnp.sqrt(jnp.maximum((Hi + dHsi_B2) ** 2 + 2 * K_ice / (rho_ice * L_ice) * (0 - Tice_B2) * dt, 0.0))
    dHsnew_B2 = PP * (1 / (1000 * 86400)) * dt
    dHs_B2 = dHsnew_B2 - dHsi_B2 * (rho_ice / rho_fw)

    Tair_gt0 = Tair > 0
    dHs_B = jnp.where(Tair_gt0, dHs_B1, dHs_B2)
    Hi_B_pre = jnp.where(Tair_gt0, Hi_B1, Hi_B2)
    Hsi_B_pre = jnp.where(Tair_gt0, Hsi_B1, Hsi_B2)
    dHsnew_B = jnp.where(Tair_gt0, dHsnew_B1, dHsnew_B2)

    Hi_B = Hi_B_pre - jnp.maximum(0.0, Q_surf / (rho_ice * L_ice))
    Hs_B = Hs + dHs_B
    Hsi_B = jnp.where(Hi_B < Hsi_B_pre, jnp.maximum(0.0, Hi_B), Hsi_B_pre)

    Hs_le0 = Hs_B <= 0
    Hs_B_safe = jnp.where(Hs_le0, 1.0, Hs_B)
    rho_snow_B = jnp.where(
        Hs_le0, rho_new_snow, rho_snow * (Hs_B - dHsnew_B) / Hs_B_safe + rho_new_snow * dHsnew_B / Hs_B_safe
    )
    Hs_B = jnp.where(Hs_le0, 0.0, Hs_B)

    condB = ice & (Hi >= Ice_min)

    # ---- candidate C: ice was on, thickness collapsed below Ice_min -----
    condC = ice & (Hi < Ice_min)

    # ---- combine -----------------------------------------------------------
    ice_final = jnp.where(condA, True, jnp.where(condB, True, jnp.where(condC, False, ice)))
    u_final = jnp.where(condA, u_A, jnp.where(condB, u_B_base, u))
    Hi_sel = jnp.where(condA, Hi_A, jnp.where(condB, Hi_B, Hi))
    Hs_sel = jnp.where(condA, Hs, jnp.where(condB, Hs_B, Hs))
    Hsi_sel = jnp.where(condA, Hsi, jnp.where(condB, Hsi_B, Hsi))
    rho_snow_sel = jnp.where(condA, rho_snow, jnp.where(condB, rho_snow_B, rho_snow))

    Hi_final = jnp.where(ice_final, Hi_sel, 0.0)
    Hs_final = jnp.where(ice_final, Hs_sel, 0.0)
    Hsi_final = jnp.where(ice_final, Hsi_sel, 0.0)

    return u_final, Hi_final, Hs_final, Hsi_final, ice_final, iceT_new, rho_snow_sel


# ---------------------------------------------------------------------------
# Diffusion: flux-form Crank-Nicolson, matches the temperature path of
# `diffusion_module_dAdK_v2` (line 2933), solved with a differentiable
# Thomas (tridiagonal) algorithm instead of `scipy.linalg.solve_banded`.
# ---------------------------------------------------------------------------


def _thomas_solve(a, b, c, d):
    """Solve a tridiagonal system. `a[0]` and `c[-1]` are ignored (as is
    conventional): row i is a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i].
    Implemented with `lax.scan` (static length) so it is safe for
    reverse-mode `jax.grad`."""

    def forward(carry, elems):
        c_prev, d_prev = carry
        a_i, b_i, c_i, d_i = elems
        denom = b_i - a_i * c_prev
        c_star = c_i / denom
        d_star = (d_i - a_i * d_prev) / denom
        return (c_star, d_star), (c_star, d_star)

    _, (c_star, d_star) = lax.scan(forward, (0.0, 0.0), (a, b, c, d))

    def backward(x_next, elems):
        c_star_i, d_star_i = elems
        x_i = d_star_i - c_star_i * x_next
        return x_i, x_i

    _, x = lax.scan(backward, 0.0, (c_star, d_star), reverse=True)
    return x


def _flux_form_coeffs(area, kz, dx, dt):
    """Build the flux-form Crank-Nicolson tridiagonal coefficients shared
    by every diffused tracer (temperature, O2, DOCr, DOCl all share the
    same `area`/`kz`/`dx`/`dt` -- only the right-hand side differs), so
    this is computed once per step and reused via `_diffuse_tracer`."""
    A_face = 0.5 * (area[:-1] + area[1:])
    K_face = 0.5 * (kz[:-1] + kz[1:])

    A_l = jnp.concatenate([jnp.zeros(1), A_face])
    K_l = jnp.concatenate([jnp.zeros(1), K_face])
    A_r = jnp.concatenate([A_face, jnp.zeros(1)])
    K_r = jnp.concatenate([K_face, jnp.zeros(1)])

    denom = area * dx * dx
    sub = (A_l * K_l) / denom
    sup = (A_r * K_r) / denom
    diag = -(sup + sub)

    a = -0.5 * dt * sub
    b = 1.0 - 0.5 * dt * diag
    c = -0.5 * dt * sup
    return sub, sup, a, b, c


def _diffuse_tracer(x, sub, sup, a, b, c, dt):
    x_left = jnp.concatenate([x[:1], x[:-1]])
    x_right = jnp.concatenate([x[1:], x[-1:]])
    Lx = sup * (x_right - x) + sub * (x_left - x)
    rhs = x + 0.5 * dt * Lx
    return _thomas_solve(a, b, c, rhs)


def diffusion_step(u, kz, area, dx, dt):
    """Temperature-only part of `diffusion_module_dAdK_v2`. Flux-form
    operator with natural (zero-flux) Neumann boundaries; solved
    implicitly (Crank-Nicolson)."""
    sub, sup, a, b, c = _flux_form_coeffs(area, kz, dx, dt)
    return _diffuse_tracer(u, sub, sup, a, b, c, dt)


def diffusion_step_wq(u, o2, docr, docl, kz, area, dx, dt):
    """Full `diffusion_module_dAdK_v2`: diffuses temperature, O2, DOCr and
    DOCl together with one shared set of tridiagonal coefficients (O2/DOC
    are diffused as *concentrations*, i.e. mass/volume, matching the
    numpy reference's `o2c = o2n/vol_arr` conversion -- the caller passes
    already-converted concentrations and converts back to mass after)."""
    sub, sup, a, b, c = _flux_form_coeffs(area, kz, dx, dt)
    u_new = _diffuse_tracer(u, sub, sup, a, b, c, dt)
    o2_new = _diffuse_tracer(o2, sub, sup, a, b, c, dt)
    docr_new = _diffuse_tracer(docr, sub, sup, a, b, c, dt)
    docl_new = _diffuse_tracer(docl, sub, sup, a, b, c, dt)
    return u_new, o2_new, docr_new, docl_new


# ---------------------------------------------------------------------------
# Wind mixing: temperature-only part of `mixing_module_minlake_RL`
# (line 3939). The data-dependent `while` search loop is replaced with a
# fixed-length `lax.fori_loop` over at most `nx` merge events (a safe
# upper bound: each successful full merge consumes wind energy against
# a genuinely stable interface, and there are at most nx-1 interfaces).
# ---------------------------------------------------------------------------


def mixing_step(u, depth, area, volume, dx, dt, Uw, ice, g=9.81, Cd=0.0013, W_str=None, tracers=()):
    """Wind-mixing search from `mixing_module_minlake_RL`. `tracers` is an
    optional tuple of extra state arrays *already expressed as
    concentrations* (state/volume -- matching the numpy reference's
    `o2 = o2n/volume` conversion before the mixing loop) that get merged
    with exactly the same zb/KP_ratio decisions as temperature; the
    caller is responsible for converting mass <-> concentration and for
    the `o2`/`docr`/... <-> `tracers` tuple bookkeeping. Returns
    `(u_out, tracers_out)`."""
    nx = u.shape[0]
    if W_str is None:
        W_str = 1.0 - jnp.exp(-0.3 * jnp.max(area) / 1e6)

    tau = 1.225 * Cd * Uw ** 2
    # On a genuinely calm hour (Uw == 0.0 exactly -- common in real hourly
    # meteorology, and reproduced here: found via a NaN gradient at a real
    # Ravn timestep), `tau**3` is exactly 0, so the `sqrt` argument is
    # exactly 0 too. The *value* is fine (0 energy input when there's no
    # wind), but `sqrt`'s local derivative at exactly 0 is infinite, and
    # `inf * 0 = NaN` in the chain rule regardless of the fact that the
    # upstream derivative is genuinely zero (tau**3 doesn't depend on
    # `u[0]` at all here) -- the same "unselected/degenerate branch
    # poisons the gradient" pattern documented elsewhere in this module
    # (`bulk_fluxes`'s zeta clamp, `eddy_diffusivity_hendersonSellers`'s
    # decay clamp). Fixed the same way: floor the sqrt argument at a
    # tiny-but-nonzero epsilon instead of exactly 0, which changes KE0 by
    # an utterly negligible amount (sqrt(1e-30) ~ 1e-15) but keeps
    # `sqrt`'s local derivative finite.
    KE0 = W_str * jnp.max(area) * jnp.sqrt(jnp.maximum(tau ** 3 / calc_dens(u[0]), 1e-30)) * dt
    KE0 = jnp.where(ice, 0.0, KE0)

    idx = jnp.arange(nx)
    n_tracers = len(tracers)

    def body(_, state):
        u, tracers, KE, active = state
        rho = calc_dens(u)
        d_rho = rho[1:] - rho[:-1]
        mask_pos = d_rho > 0
        any_pos = jnp.any(mask_pos)
        zb = jnp.argmax(mask_pos)
        is_last = zb == (nx - 2)
        stop_now = jnp.logical_or(jnp.logical_not(any_pos), is_last)
        do_update = jnp.logical_and(active, jnp.logical_not(stop_now))

        mask_le_zb = idx <= zb
        mask_le_zb1 = idx <= (zb + 1)

        MLD = depth[zb] + dx / 2
        dD = d_rho[zb]
        Zg_num = jnp.sum(jnp.where(mask_le_zb, area * depth * rho * dx, 0.0))
        Zg_den = jnp.sum(jnp.where(mask_le_zb, area * rho * dx, 0.0))
        Zg = jnp.ceil((Zg_num / Zg_den) * 100) / 100
        volume_epi = jnp.sum(jnp.where(mask_le_zb, volume, 0.0))
        vol_zb1 = volume[zb + 1]
        V_weight = vol_zb1 * volume_epi / (vol_zb1 + volume_epi)
        POE = dD * g * V_weight * (MLD + dx / 2 - Zg)
        KP_ratio = KE / jnp.where(POE != 0, POE, 1.0)

        branch_gt1 = KP_ratio > 1
        denom2 = jnp.sum(jnp.where(mask_le_zb, volume, 0.0)) + KP_ratio * vol_zb1

        def merge(x):
            # KP_ratio > 1: fully mix layers [0, zb+1]
            xmix1 = jnp.sum(jnp.where(mask_le_zb1, volume * x, 0.0)) / jnp.sum(jnp.where(mask_le_zb1, volume, 0.0))
            x_branch1 = jnp.where(mask_le_zb1, xmix1, x)
            # KP_ratio <= 1: partially entrain layer zb+1, then stop
            x_zb1 = x[zb + 1]
            sum_low = jnp.sum(jnp.where(mask_le_zb, volume * x, 0.0))
            xmix2 = (sum_low + KP_ratio * vol_zb1 * x_zb1) / denom2
            x_branch2 = jnp.where(mask_le_zb, xmix2, x)
            x_branch2 = x_branch2.at[zb + 1].set(KP_ratio * xmix2 + (1 - KP_ratio) * x_zb1)
            x_upd = jnp.where(branch_gt1, x_branch1, x_branch2)
            return jnp.where(do_update, x_upd, x)

        new_u = merge(u)
        new_tracers = tuple(merge(t) for t in tracers)

        KE_branch1 = KE - POE
        KE_branch2 = 0.0
        new_KE_upd = jnp.where(branch_gt1, KE_branch1, KE_branch2)
        new_active_upd = jnp.where(branch_gt1, active, False)
        new_KE = jnp.where(do_update, new_KE_upd, KE)
        new_active = jnp.where(do_update, new_active_upd, jnp.where(stop_now, False, active))
        return (new_u, new_tracers, new_KE, new_active)

    u_out, tracers_out, _, _ = lax.fori_loop(0, nx, body, (u, tuple(tracers), KE0, True))
    if n_tracers == 0:
        return u_out, ()
    return u_out, tracers_out


# ---------------------------------------------------------------------------
# Convective overturn: matches `convection_module` (line 1724). The
# original repeatedly bubble-merges adjacent unstable layers until the
# column is stable; replaced with a fixed `max_outer` x (nx-1) sequential
# sweep. Because a stabilized profile is a fixed point of the sweep,
# extra passes beyond convergence are harmless no-ops, so this is exact
# as long as `max_outer` is large enough. Empirically (200 adversarial
# random profiles at nx=64, see validation notes) `max_outer = nx` is
# NOT always enough -- merging volumetrically-weighted cells is not the
# same contraction as a plain bubble sort -- but `4 * nx` reproduced the
# numpy reference to floating-point precision (~1e-14) in every trial,
# so that is the default. Real physical profiles (already smoothed by
# heating/diffusion/mixing) need far fewer passes in practice; lower
# `max_outer` for speed only after re-checking against the reference.
# ---------------------------------------------------------------------------


def convection_step(u, volume, denThresh=1e-3, max_outer=None):
    nx = u.shape[0]
    if max_outer is None:
        max_outer = 4 * nx

    def inner_body(dep, u):
        dens_u = calc_dens(u)
        d_lo = dens_u[dep]
        d_hi = dens_u[dep + 1]
        unstable = jnp.logical_and(d_hi < d_lo, jnp.abs(d_hi - d_lo) >= denThresh)
        v_lo = volume[dep]
        v_hi = volume[dep + 1]
        u_lo = u[dep]
        u_hi = u[dep + 1]
        merged = (u_lo * v_lo + u_hi * v_hi) / (v_lo + v_hi)
        u = u.at[dep].set(jnp.where(unstable, merged, u_lo))
        u = u.at[dep + 1].set(jnp.where(unstable, merged, u_hi))
        return u

    def outer_body(_, u):
        return lax.fori_loop(0, nx - 1, inner_body, u)

    return lax.fori_loop(0, max_outer, outer_body, u)


# ---------------------------------------------------------------------------
# One full physical time step, and the scan driver over a whole simulation.
# ---------------------------------------------------------------------------


class LakeState(dict):
    """Thin dict subclass so the pytree carry has readable field access.

    Missing-key attribute lookups must raise `AttributeError` (not
    `KeyError`) or JAX's internal `hasattr(...)`-style duck typing checks
    (e.g. for `__jax_array__`) break."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    __setattr__ = dict.__setitem__


jax.tree_util.register_pytree_node(
    LakeState,
    lambda d: (list(d.values()), list(d.keys())),
    lambda keys, values: LakeState(zip(keys, values)),
)


def make_initial_state(u0, nx, ice=False, Hi=0.0, Hs=0.0, Hsi=0.0, iceT=6.0, rho_snow=250.0):
    return LakeState(
        u=jnp.asarray(u0, dtype=jnp.float64),
        kz=jnp.zeros(nx, dtype=jnp.float64),
        ice=jnp.array(bool(ice)),
        Hi=jnp.array(float(Hi)),
        Hs=jnp.array(float(Hs)),
        Hsi=jnp.array(float(Hsi)),
        iceT=jnp.array(float(iceT)),
        rho_snow=jnp.array(float(rho_snow)),
    )


def temperature_step(state, forcing, geometry, params):
    """One hourly (or `dt`-length) physical step of the thermal engine.

    `geometry` = dict(area, depth, volume, dx, dt, latitude)
    `params`   = dict of physical constants (see `default_params()`)
    `forcing`  = dict of scalars for this step (Tair, CC, ea, Jsw, Jlw, Uw, Pa, RH, PP)
    """
    area, depth, volume = geometry["area"], geometry["depth"], geometry["volume"]
    dx, dt = geometry["dx"], geometry["dt"]
    # params["wind_factor"] scales the wind speed used everywhere in this
    # step -- see default_params()'s note on why it's applied here rather
    # than baked into the forcing series.
    Uw_eff = forcing["Uw"] * params["wind_factor"]

    u, IceSnowAttCoeff, _ = heating_module(
        state.u, area, volume, depth, dt, dx, state.ice,
        forcing["Tair"], forcing["CC"], forcing["ea"], forcing["Jsw"], forcing["Jlw"],
        Uw_eff, forcing["Pa"], forcing["RH"], params["kd_light"], state.Hi, state.Hs,
        state.rho_snow, kd_ice=params["kd_ice"], kd_snow=params["kd_snow"], rho_fw=params["rho_fw"],
        sigma=params["sigma"], eps=params["eps"], emissivity=params["emissivity"], p2=params["p2"],
        Cd=params["Cd"], sw_factor=params["sw_factor"], at_factor=params["at_factor"],
        turb_factor=params["turb_factor"], Hgeo=params["Hgeo"],
    )

    u, Hi, Hs, Hsi, ice, iceT, rho_snow = ice_module(
        u, dt, dx, area, forcing["Tair"], forcing["CC"], forcing["ea"], forcing["Jsw"], forcing["Jlw"],
        Uw_eff, forcing["Pa"], forcing["RH"], forcing["PP"], IceSnowAttCoeff,
        state.ice, params["dt_iceon_avg"], state.iceT, state.rho_snow, state.Hi, state.Hs, state.Hsi,
        eps=params["eps"], emissivity=params["emissivity"], sigma=params["sigma"], p2=params["p2"],
        Cd=params["Cd"], rho_new_snow=params["rho_new_snow"], rho_ice=params["rho_ice"],
        rho_fw=params["rho_fw"], Ice_min=params["Ice_min"], Cw=params["Cw"], L_ice=params["L_ice"],
        meltP=params["meltP"], K_ice=params["K_ice"],
    )

    dens_u = calc_dens(u)
    kz = eddy_diffusivity_hendersonSellers(
        # NOTE: the reference's `run_wq_model` calls this with a hardcoded
        # latitude (43.100948) instead of the lake's configured latitude,
        # in every diffusion_method branch (line 4572) -- reproduced here
        # rather than "fixed" to `geometry["latitude"]`, since the goal is
        # a numerically matching port.
        dens_u, depth, params["g"], jnp.mean(dens_u), ice, Uw_eff, geometry["latitude"], u,
        state.kz, params["Cd"], params["km"], params["weight_kz"],
    )

    u = diffusion_step(u, kz, area, dx, dt)
    u, _ = mixing_step(u, depth, area, volume, dx, dt, Uw_eff, ice, g=params["g"], Cd=params["Cd"],
                        W_str=params["W_str"])
    u = convection_step(u, volume, denThresh=params["denThresh"], max_outer=params["max_conv_passes"])

    new_state = LakeState(u=u, kz=kz, ice=ice, Hi=Hi, Hs=Hs, Hsi=Hsi, iceT=iceT, rho_snow=rho_snow)
    return new_state


def run_temperature_model(u0, forcing_series, geometry, params, ice_state=None):
    """Run the full thermal engine with `lax.scan`.

    `forcing_series` is a dict of 1-D arrays (n_steps,) for keys
    Tair, CC, ea, Jsw, Jlw, Uw, Pa, RH, PP -- one entry per time step,
    already interpolated onto the model's time grid (see
    `run_M3_jax.py` for how these are built from the Ravn meteorology
    file with the *same* `scipy.interp1d` calls the numpy model uses,
    so the forcing seen by both models is numerically identical).

    `ice_state`, if given, is a dict with any of ice/Hi/Hs/Hsi/iceT/rho_snow
    (as returned by `get_ice_and_snow()` in the numpy code) to seed the
    initial ice state; defaults match a fully ice-free lake.

    Returns (final_state, per_step_temperature) where
    per_step_temperature has shape (n_steps, nx).
    """
    nx = u0.shape[0]
    ice_state = ice_state or {}
    state0 = make_initial_state(u0, nx, **ice_state)

    def body(state, forcing_t):
        new_state = temperature_step(state, forcing_t, geometry, params)
        return new_state, new_state.u

    final_state, u_all = lax.scan(body, state0, forcing_series)
    return final_state, u_all


# ---------------------------------------------------------------------------
# Phase 2: water quality (O2, DOCr/DOCl, POCr/POCl).
# ---------------------------------------------------------------------------
# Gas-exchange helpers: matches `do_sat_calc` (line 1921), `k_vachon`
# (line 5350) and `k600_to_kgas`/`get_schmidt` (lines 5358-5392), the
# latter specialized to gas="O2" (the only gas ever requested).
# ---------------------------------------------------------------------------


def do_sat_calc(temp, baro, altitude=0.0, salinity=0.0):
    """Dissolved-oxygen saturation concentration. Matches `do_sat_calc()`
    line 1921 (the `baro is None` branch is never exercised by
    `run_wq_model` -- `Pa` is always supplied -- so it's omitted here)."""
    mgL_mlL = 1.42905
    mmHg_mb = 0.750061683
    baro_mb = jnp.where(baro > 2000, baro / 100.0, baro)  # Pa -> hPa, as in the reference
    u = 10 ** (8.10765 - 1750.286 / (235 + temp))
    press_corr = (baro_mb * mmHg_mb - u) / (760 - u)
    ts = jnp.log((298.15 - temp) / (273.15 + temp))
    o2_sat = (
        2.00907 + 3.22014 * ts + 4.05010 * ts ** 2 + 4.94457 * ts ** 3 - 2.56847e-1 * ts ** 4 + 3.88767 * ts ** 5
        - salinity * (6.24523e-3 + 7.37614e-3 * ts + 1.03410e-2 * ts ** 2 + 8.17083e-3 * ts ** 3)
        - 4.88682e-7 * salinity ** 2
    )
    return jnp.exp(o2_sat) * mgL_mlL * press_corr


def k_vachon(wind, area, param1=2.51, param2=1.48, param3=0.39):
    """Gas transfer velocity k600. Matches `k_vachon()` line 5350."""
    k600 = param1 + param2 * wind + param3 * wind * (jnp.log(area / 1e6) / jnp.log(10.0))
    return k600 * 24 / 100  # cm/hr -> m/day


def k600_to_kgas_o2(k600, temperature):
    """k600 -> kO2 via the Schmidt number. Matches `k600_to_kgas()`/
    `get_schmidt()` (lines 5358-5392) specialized to gas="O2"
    (a=1568, b=-86.04, c=2.142, d=-0.0216)."""
    schmidt = 1568.0 - 86.04 * temperature + 2.142 * temperature ** 2 - 0.0216 * temperature ** 3
    return k600 * (schmidt / 600.0) ** -0.5


# ---------------------------------------------------------------------------
# O2 boundary flux: matches the O2-only part of `boundary_module_oxygen`
# (line 3537) -- atmospheric exchange plus sediment oxygen demand. DOC/POC
# pass through this module unchanged in the reference, so it isn't
# reproduced here at all.
# ---------------------------------------------------------------------------


def oxygen_boundary_step(u, o2, area, volume, dt, altitude, ice, Pa, Tair, Uw, theta_r, f_sod, d_thick):
    piston_velocity_ice = 1e-5 / 86400
    k600 = k_vachon(Uw, area[0])
    piston_velocity_open = k600_to_kgas_o2(k600, Tair) / 86400
    piston_velocity = jnp.where(ice, piston_velocity_ice, piston_velocity_open)

    o2_sat = do_sat_calc(u[0], baro=Pa, altitude=altitude)
    atm_flux = piston_velocity * (o2_sat - o2[0] / volume[0]) * area[0]
    o2 = o2.at[0].add(atm_flux * dt)

    u_k = u + 273.15
    d_sod = 10 ** (-4.410 + 773.8 / u_k - (506.4 / u_k) ** 2) / 10000

    A_sed_interior = jnp.maximum(area[:-1] - area[1:], 0.0)
    A_sed = jnp.concatenate([A_sed_interior, area[-1:]])
    sed_oxygen_loss = A_sed * f_sod

    C_o2 = o2 / volume
    C_sed = o2 / volume / 2
    F_sed_diff = d_sod[-1] * (C_o2[-1] - C_sed[-1]) / d_thick
    sed_oxygen_loss = sed_oxygen_loss.at[-1].add(A_sed[-1] * F_sed_diff)

    do_consumption = sed_oxygen_loss * dt * theta_r ** (u - 20)
    o2 = o2 - jnp.minimum(o2, do_consumption)
    return o2, atm_flux, do_consumption


# ---------------------------------------------------------------------------
# POC settling: matches `advection_diffusion_module_mylake` (line 3846),
# an exponentially-fitted (Scharfetter-Gummel-style) advection-diffusion
# tridiagonal, solved with the same differentiable Thomas algorithm as
# the Crank-Nicolson diffusion. Note the reference's discretization does
# NOT area/volume-weight this operator (only `area`/`volume` in the
# signature aren't referenced in the body) -- reproduced as-is.
# ---------------------------------------------------------------------------


def poc_settling_step(pocr, pocl, kz, settling_rate_refractory, settling_rate_labile, dx, dt):
    def solve_settling(poc, U):
        Kz_face = 0.5 * (kz[:-1] + kz[1:])
        theta = U * dt / dx
        Pe_face = jnp.clip(U * dx / Kz_face, 1e-12, 700.0)
        expPe = jnp.exp(Pe_face)
        az_interior = theta * (1.0 + 1.0 / (expPe - 1.0))  # az[1:]
        cz_interior = theta / (expPe - 1.0)  # cz[:-1]
        az = jnp.concatenate([jnp.zeros(1), az_interior])
        cz = jnp.concatenate([cz_interior, jnp.zeros(1)])
        bz = 1.0 + az + cz
        bz = bz.at[0].set(1.0 + theta + cz[0])
        bz = bz.at[-1].set(1.0 + az[-1])
        a = jnp.concatenate([jnp.zeros(1), -cz[:-1]])
        c = jnp.concatenate([-az[1:], jnp.zeros(1)])
        return _thomas_solve(a, bz, c, poc)

    pocr_new = solve_settling(pocr, settling_rate_refractory)
    pocl_new = solve_settling(pocl, settling_rate_labile)
    return pocr_new, pocl_new


# ---------------------------------------------------------------------------
# Production/consumption reactions: matches `prodcons_module_woDOCL`
# (line 2117) -- a modified Patankar-Runge-Kutta (MPRK) predictor-corrector
# that solves a small (5x5) linear system twice per layer per step. Each
# depth layer is independent given the shared scalars (consumption, npp),
# so this vmaps cleanly over depth instead of a Python loop.
# ---------------------------------------------------------------------------

_PRODCONS_EPS = 1e-12


def _prodcons_pd_matrices(y, consumption, npp, resp, beta):
    """Build the (5x5) production and destruction matrices for state
    `y = (o2, docr, docl, pocr, pocl)`. Matches `fun()` inside
    `prodcons_module_woDOCL` line 2231."""
    o2v, docrv, doclv, pocrv, poclv = y[0], y[1], y[2], y[3], y[4]
    resp_docr, resp_docl, resp_pocr, resp_pocl = resp
    oxygen, carbon = 32.0, 12.0
    carbon_oxygen = oxygen / carbon

    p = jnp.zeros((5, 5))
    p = p.at[0, 0].set(npp * oxygen)
    p = p.at[1, 3].set(0.1 * (pocrv * resp_pocr * consumption))
    p = p.at[2, 4].set((1 - beta) * npp * carbon)
    p = p.at[4, 4].set(beta * npp * carbon)

    d = jnp.zeros((5, 5))
    d = d.at[0, 1].set(carbon_oxygen * (docrv * resp_docr * consumption))
    d = d.at[0, 2].set(carbon_oxygen * (doclv * resp_docl * consumption))
    d = d.at[0, 3].set(0.9 * carbon_oxygen * (pocrv * resp_pocr * consumption))
    d = d.at[0, 4].set(carbon_oxygen * (poclv * resp_pocl * consumption))
    d = d.at[1, 1].set(docrv * resp_docr * consumption)
    d = d.at[2, 2].set(doclv * resp_docl * consumption)
    d = d.at[3, 3].set(pocrv * resp_pocr * consumption)
    d = d.at[4, 4].set(poclv * resp_pocl * consumption)
    return p, d


def _prodcons_single_layer(y0, u_i, volume_i, area_i, H_i, TP, dt, theta_r, k_half, theta_npp, resp, beta,
                            sw_to_par):
    y0 = jnp.stack(y0)

    o2c = y0[0] / volume_i
    consumption = theta_r ** (u_i - 20) * o2c / (k_half + o2c)

    PAR = H_i * sw_to_par / 1e6  # mol/m2/s
    P_max = 1.5e-6  # mol C/m2/s
    alpha_P = 0.03
    LIGHTUSEBYPHOTOS = 0.3  # matches the reference's hardcoded shadow of the caller's value
    P_I = P_max * (1 - jnp.exp(-(alpha_P * LIGHTUSEBYPHOTOS) * PAR / P_max))
    k_TP = 60.0  # 0.06 * 1000
    f_TP = TP / (k_TP + TP)
    temp_factor = theta_npp ** (u_i - 20)
    npp = P_I * f_TP * temp_factor * area_i

    eye = jnp.eye(5, dtype=bool)

    def safe(v):
        return jnp.where(v == 0, _PRODCONS_EPS, v)

    p0, d0 = _prodcons_pd_matrices(y0, consumption, npp, resp, beta)
    y0_safe = safe(y0)
    a0 = jnp.where(eye, jnp.diag(dt * jnp.sum(d0, axis=1) / y0_safe + 1.0), -dt * p0 / y0_safe[None, :])
    r0 = y0 + dt * jnp.diag(p0)
    c0 = jnp.linalg.solve(a0, r0)

    p1, d1 = _prodcons_pd_matrices(c0, consumption, npp, resp, beta)
    p_avg = 0.5 * (p0 + p1)
    d_avg = 0.5 * (d0 + d1)
    c0_safe = safe(c0)
    a1 = jnp.where(eye, jnp.diag(dt * jnp.sum(d_avg, axis=1) / c0_safe + 1.0), -dt * p_avg / c0_safe[None, :])
    r1 = y0 + dt * jnp.diag(p_avg)
    y_new = jnp.linalg.solve(a1, r1)

    resp_docr, resp_docl, resp_pocr, resp_pocl = resp
    diagnostics = jnp.stack([
        86400 * resp_docr * consumption, 86400 * resp_docl * consumption,
        86400 * resp_pocr * consumption, 86400 * resp_pocl * consumption, npp * 86400 * 12.0,
    ])
    return y_new, diagnostics


def prodcons_step(u, o2, docr, docl, pocr, pocl, area, volume, H, TP, dt, theta_r, k_half, theta_npp, resp,
                   beta, sw_to_par=2.114):
    """`resp` = (resp_docr, resp_docl, resp_pocr, resp_pocl). Returns
    (o2, docr, docl, pocr, pocl, diagnostics_dict). The bottom layer
    (index nx-1) is left unchanged -- see module docstring point 4."""
    nx = u.shape[0]

    def per_layer(o2_i, docr_i, docl_i, pocr_i, pocl_i, u_i, volume_i, area_i, H_i):
        return _prodcons_single_layer(
            (o2_i, docr_i, docl_i, pocr_i, pocl_i), u_i, volume_i, area_i, H_i, TP, dt,
            theta_r, k_half, theta_npp, resp, beta, sw_to_par,
        )

    y_new, diagnostics = jax.vmap(per_layer)(o2, docr, docl, pocr, pocl, u, volume, area, H)

    mask = jnp.arange(nx) < (nx - 1)
    o2_new = jnp.where(mask, y_new[:, 0], o2)
    docr_new = jnp.where(mask, y_new[:, 1], docr)
    docl_new = jnp.where(mask, y_new[:, 2], docl)
    pocr_new = jnp.where(mask, y_new[:, 3], pocr)
    pocl_new = jnp.where(mask, y_new[:, 4], pocl)

    diag = dict(
        docr_respiration=jnp.where(mask, diagnostics[:, 0], 0.0),
        docl_respiration=jnp.where(mask, diagnostics[:, 1], 0.0),
        pocr_respiration=jnp.where(mask, diagnostics[:, 2], 0.0),
        pocl_respiration=jnp.where(mask, diagnostics[:, 3], 0.0),
        npp_production=jnp.where(mask, diagnostics[:, 4], 0.0),
    )
    return o2_new, docr_new, docl_new, pocr_new, pocl_new, diag


def make_initial_state_full(u0, o2_0, docr_0, docl_0, pocr_0, pocl_0, nx, ice=False, Hi=0.0, Hs=0.0, Hsi=0.0,
                             iceT=6.0, rho_snow=250.0):
    state = make_initial_state(u0, nx, ice=ice, Hi=Hi, Hs=Hs, Hsi=Hsi, iceT=iceT, rho_snow=rho_snow)
    state["o2"] = jnp.asarray(o2_0, dtype=jnp.float64)
    state["docr"] = jnp.asarray(docr_0, dtype=jnp.float64)
    state["docl"] = jnp.asarray(docl_0, dtype=jnp.float64)
    state["pocr"] = jnp.asarray(pocr_0, dtype=jnp.float64)
    state["pocl"] = jnp.asarray(pocl_0, dtype=jnp.float64)
    return state


def full_step(state, forcing, geometry, params, kz_override=None):
    """One full timestep: temperature + O2/DOCr/DOCl/POCr/POCl, matching
    `run_wq_model`'s per-iteration order (see module docstring).

    `geometry` extends the Phase-1 dict with: `hypso_weight`, `depth_mask`
    (= `depth < mean_depth`, static), `outflow_frac` (static -- see
    `default_geometry_wq()`), and `altitude`.
    `forcing` extends the Phase-1 dict with `carbon` and `TP`.

    `kz_override`, if given, replaces the process-based eddy diffusivity
    (`eddy_diffusivity_hendersonSellers`'s output, still always computed
    and reported as `kz_process` in the returned diagnostics dict) before
    diffusion/settling. It is either a length-`nx` array used directly, or
    a callable `kz_override(kz_process, u, ice, dens_u, forcing, ri) -> kz`
    called with this step's *post*-heating/ice-module temperature (`u`),
    ice flag (`ice`), density (`dens_u`), and the same depth-resolved
    Richardson number `eddy_diffusivity_hendersonSellers` computed
    `kz_process` from (also reported as `ri_process` in diagnostics) --
    i.e. exactly the intermediates `kz_process` itself was computed from,
    which a plain array can't give a caller access to (they aren't
    otherwise exposed before this point in the step). `run_M3_mcl_jax.py`
    uses the callable form so its NN's per-step input features
    (surface/bottom temperature, ice state) reflect this step's own state
    rather than the previous step's, and so it can fit a stability function
    directly against `ri` instead of (or as well as) correcting
    `kz_process`. The default (`kz_override=None`) code path is numerically
    identical to before this parameter was added."""
    area, depth, volume = geometry["area"], geometry["depth"], geometry["volume"]
    dx, dt = geometry["dx"], geometry["dt"]
    docr, docl, pocr, pocl, o2 = state.docr, state.docl, state.pocr, state.pocl, state.o2
    # params["wind_factor"] scales the wind speed used everywhere in this
    # step -- see default_params()'s note on why it's applied here rather
    # than baked into the forcing series.
    Uw_eff = forcing["Uw"] * params["wind_factor"]

    # 1. dynamic kd_light from the *previous* step's WQ state
    depth_mask = geometry["depth_mask"]
    n_mask = jnp.sum(depth_mask)
    sum_doc = jnp.where(depth_mask, (docr + docl) / volume, 0.0)
    sum_poc = jnp.where(depth_mask, (pocr + pocl) / volume, 0.0)
    kd_light = (
        params["light_water"]
        + params["light_doc"] * jnp.sum(sum_doc) / n_mask
        + params["light_poc"] * jnp.sum(sum_poc) / n_mask
    )

    # 2. OC river loading (`oc_load_factor` scales the measured "oc"
    # concentration in oc_load_file.csv -- see default_params() -- treating
    # the overall carbon-loading boundary condition as a calibratable unknown)
    perdepth_oc = forcing["carbon"] * params["oc_load_factor"] * geometry["hypso_weight"]
    docr = docr + perdepth_oc * params["prop_oc_docr"]
    docl = docl + perdepth_oc * params["prop_oc_docl"]
    pocr = pocr + perdepth_oc * params["prop_oc_pocr"]
    pocl = pocl + perdepth_oc * params["prop_oc_pocl"]

    # 3. heating (uses the dynamic kd_light, not a held-constant value)
    u, IceSnowAttCoeff, Q_net = heating_module(
        state.u, area, volume, depth, dt, dx, state.ice,
        forcing["Tair"], forcing["CC"], forcing["ea"], forcing["Jsw"], forcing["Jlw"],
        Uw_eff, forcing["Pa"], forcing["RH"], kd_light, state.Hi, state.Hs, state.rho_snow,
        kd_ice=params["kd_ice"], kd_snow=params["kd_snow"], rho_fw=params["rho_fw"],
        sigma=params["sigma"], eps=params["eps"], emissivity=params["emissivity"], p2=params["p2"],
        Cd=params["Cd"], sw_factor=params["sw_factor"], at_factor=params["at_factor"],
        turb_factor=params["turb_factor"], Hgeo=params["Hgeo"],
    )

    # 4. ice
    u, Hi, Hs, Hsi, ice, iceT, rho_snow = ice_module(
        u, dt, dx, area, forcing["Tair"], forcing["CC"], forcing["ea"], forcing["Jsw"], forcing["Jlw"],
        Uw_eff, forcing["Pa"], forcing["RH"], forcing["PP"], IceSnowAttCoeff,
        state.ice, params["dt_iceon_avg"], state.iceT, state.rho_snow, state.Hi, state.Hs, state.Hsi,
        eps=params["eps"], emissivity=params["emissivity"], sigma=params["sigma"], p2=params["p2"],
        Cd=params["Cd"], rho_new_snow=params["rho_new_snow"], rho_ice=params["rho_ice"],
        rho_fw=params["rho_fw"], Ice_min=params["Ice_min"], Cw=params["Cw"], L_ice=params["L_ice"],
        meltP=params["meltP"], K_ice=params["K_ice"],
    )

    # 5. O2 boundary (atmospheric exchange + sediment oxygen demand)
    o2, atm_flux, do_consumption = oxygen_boundary_step(
        u, o2, area, volume, dt, geometry["altitude"], ice, forcing["Pa"], forcing["Tair"], Uw_eff,
        params["theta_r"], params["f_sod"], params["d_thick"],
    )

    # 6. eddy diffusivity
    dens_u = calc_dens(u)
    kz_process, ri_process = eddy_diffusivity_hendersonSellers(
        # NOTE: the reference's `run_wq_model` calls this with a hardcoded
        # latitude (43.100948) instead of the lake's configured latitude,
        # in every diffusion_method branch (line 4572) -- reproduced here
        # rather than "fixed" to `geometry["latitude"]`, since the goal is
        # a numerically matching port.
        dens_u, depth, params["g"], jnp.mean(dens_u), ice, Uw_eff, geometry["latitude"], u,
        state.kz, params["Cd"], params["km"], params["weight_kz"], return_ri=True,
    )
    if kz_override is None:
        kz = kz_process
    elif callable(kz_override):
        kz = kz_override(kz_process, u, ice, dens_u, forcing, ri_process)
    else:
        kz = kz_override

    # 7. diffusion (temperature + O2 + DOCr + DOCl, as concentrations)
    o2c, docrc, doclc = o2 / volume, docr / volume, docl / volume
    u, o2c, docrc, doclc = diffusion_step_wq(u, o2c, docrc, doclc, kz, area, dx, dt)
    o2, docr, docl = o2c * volume, docrc * volume, doclc * volume

    # 8. POC settling
    pocr, pocl = poc_settling_step(
        pocr, pocl, kz, params["settling_rate_refractory"], params["settling_rate_labile"], dx, dt,
    )

    # 9. mixing (temperature + all 5 WQ tracers, as concentrations)
    o2c, docrc, doclc, pocrc, poclc = o2 / volume, docr / volume, docl / volume, pocr / volume, pocl / volume
    u, (o2c, docrc, doclc, pocrc, poclc) = mixing_step(
        u, depth, area, volume, dx, dt, Uw_eff, ice, g=params["g"], Cd=params["Cd"],
        W_str=params["W_str"], tracers=(o2c, docrc, doclc, pocrc, poclc),
    )
    o2, docr, docl = o2c * volume, docrc * volume, doclc * volume
    pocr, pocl = pocrc * volume, poclc * volume

    # 10. convection (temperature only), this checks for density instabilities
    u = convection_step(u, volume, denThresh=params["denThresh"], max_outer=params["max_conv_passes"])

    # 11. production/consumption reactions
    albedo = jnp.where(ice, 0.3, 0.1)
    H = jnp.where(
        ice, IceSnowAttCoeff * forcing["Jsw"] * jnp.exp(-kd_light * depth),
        (1 - albedo) * forcing["Jsw"] * jnp.exp(-kd_light * depth),
    )
    o2, docr, docl, pocr, pocl, diag = prodcons_step(
        u, o2, docr, docl, pocr, pocl, area, volume, H, forcing["TP"], dt,
        params["theta_r"], params["k_half"], params["theta_npp"],
        (params["resp_docr"], params["resp_docl"], params["resp_pocr"], params["resp_pocl"]),
        params["beta"],
    )

    # 12. hydrologic outflow (DOC/POC only, not O2)
    outflow_frac = geometry["outflow_frac"]
    docr = docr * (1 - outflow_frac)
    docl = docl * (1 - outflow_frac)
    pocr = pocr * (1 - outflow_frac)
    pocl = pocl * (1 - outflow_frac)

    new_state = LakeState(
        u=u, kz=kz, ice=ice, Hi=Hi, Hs=Hs, Hsi=Hsi, iceT=iceT, rho_snow=rho_snow,
        o2=o2, docr=docr, docl=docl, pocr=pocr, pocl=pocl,
    )
    diagnostics = dict(
        kd_light=kd_light, atm_flux=atm_flux, kz_process=kz_process, ri_process=ri_process,
        Q_net=Q_net, **diag,
    )
    return new_state, diagnostics


def default_geometry_wq(area, depth, volume, dx, dt, latitude, altitude, hypso_weight, mean_depth,
                         hydro_res_time_hr):
    """Build the `geometry` dict `full_step` expects: the Phase-1 fields
    plus the WQ-only static arrays (all time-invariant, so computed once
    here rather than every scan step)."""
    depth = jnp.asarray(depth)
    volume = jnp.asarray(volume)
    depth_mask = depth < mean_depth
    total_outflow = (1.0 / hydro_res_time_hr) * jnp.sum(volume)
    volume_out = total_outflow * jnp.asarray(hypso_weight)
    outflow_frac = volume_out / volume
    return dict(
        area=jnp.asarray(area), depth=depth, volume=volume, dx=dx, dt=dt, latitude=latitude,
        altitude=altitude, hypso_weight=jnp.asarray(hypso_weight), depth_mask=depth_mask,
        outflow_frac=outflow_frac,
    )


def run_full_model(u0, o2_0, docr_0, docl_0, pocr_0, pocl_0, forcing_series, geometry, params, ice_state=None):
    """Run the full temperature + water-quality model with `lax.scan`.

    `forcing_series` extends the Phase-1 dict with `carbon` and `TP`
    (both interpolated onto the model's time grid the same way as the
    meteorology, see `run_M3_jax.py`).

    Returns (final_state, per_step) where `per_step` is a dict with keys
    u, o2, docr, docl, pocr, pocl (each shape (n_steps, nx)).
    """
    nx = u0.shape[0]
    ice_state = ice_state or {}
    state0 = make_initial_state_full(u0, o2_0, docr_0, docl_0, pocr_0, pocl_0, nx, **ice_state)

    def body(state, forcing_t):
        new_state, _ = full_step(state, forcing_t, geometry, params)
        outputs = dict(
            u=new_state.u, o2=new_state.o2, docr=new_state.docr, docl=new_state.docl,
            pocr=new_state.pocr, pocl=new_state.pocl,
        )
        return new_state, outputs

    final_state, per_step = lax.scan(body, state0, forcing_series)
    return final_state, per_step


def default_params(model_params: dict, ice_and_snow: dict = None) -> dict:
    """Build the `params` dict `temperature_step`/`run_temperature_model`
    expect, from a Lake-M3 `model_params` row (as returned by
    `get_model_params()` in the numpy code) plus the ice/snow config row
    (as returned by `get_ice_and_snow()`, for `dt_iceon_avg`/`Ice_min`)."""
    import math

    if ice_and_snow is None:
        ice_and_snow = {}
    w_str_raw = model_params["W_str"]
    try:
        w_str_is_nan = w_str_raw is None or (isinstance(w_str_raw, float) and math.isnan(w_str_raw))
    except TypeError:
        w_str_is_nan = True

    return dict(
        kd_light=float(model_params["kd_light"]),
        kd_ice=0.7,
        kd_snow=0.9,
        rho_fw=1000.0,
        sigma=float(model_params["sigma"]),
        eps=float(model_params["eps"]),
        emissivity=float(model_params["emissivity"]),
        p2=float(model_params["p2"]),
        Cd=float(model_params["Cd"]),
        sw_factor=float(model_params["sw_factor"]),
        at_factor=float(model_params["at_factor"]),
        turb_factor=float(model_params["turb_factor"]),
        # NOTE: the reference `run_wq_model` applies this as a second wind
        # multiplier *on top of* lake_config.csv's own "WindSpeed" (already
        # baked into `forcing["Uw"]` when it's built -- see
        # `build_forcing_series()`/`provide_meteorology()`): run_M3.py passes
        # `wind_factor=model_params["wind_factor"]` into run_wq_model, which
        # rebuilds its own Uw series as `wind_factor * daily_meteo.wind`
        # (itself already scaled by lake_config's WindSpeed upstream). Ported
        # here the same way sw_factor/at_factor/turb_factor are -- applied to
        # the forcing value inside `temperature_step`/`full_step`, not baked
        # into the forcing series -- so it stays a plain entry of `params`
        # and is differentiable/calibratable like any other CANDIDATE_PARAMS
        # entry (unlike lake_config's WindSpeed, which is fixed at forcing-
        # construction time and not part of `params` at all).
        wind_factor=float(model_params["wind_factor"]),
        Hgeo=float(model_params["Hgeo"]),
        dt_iceon_avg=float(ice_and_snow.get("dt_iceon_avg", 0.8)),
        rho_new_snow=250.0,
        rho_ice=910.0,
        Ice_min=float(ice_and_snow.get("Ice_min", 0.1)),
        Cw=4.18e6,
        L_ice=333500.0,
        meltP=float(model_params["meltP"]),
        K_ice=2.1,
        g=float(model_params["g"]),
        km=float(model_params["km"]),
        weight_kz=float(model_params["weight_kz"]),
        W_str=(None if w_str_is_nan else float(w_str_raw)),
        denThresh=float(model_params["denThresh"]),
        max_conv_passes=None,  # defaults to nx inside convection_step
        # --- Phase 2 (water quality) only, below ---
        light_water=float(model_params["light_water"]),
        light_doc=float(model_params["light_doc"]),
        light_poc=float(model_params["light_poc"]),
        theta_r=float(model_params["theta_r"]),
        theta_npp=float(model_params["theta_npp"]),
        k_half=float(model_params["k_half"]),
        # resp_*/settling_rate_* are day^-1 / m day^-1 in model_params.csv; run_wq_model (via
        # run_M3.py) always divides by 86400 before use, so we do the same here.
        resp_docr=float(model_params["resp_docr"]) / 86400,
        resp_docl=float(model_params["resp_docl"]) / 86400,
        resp_pocr=float(model_params["resp_pocr"]) / 86400,
        resp_pocl=float(model_params["resp_pocl"]) / 86400,
        prop_oc_docr=float(model_params["prop_oc_docr"]),
        prop_oc_docl=float(model_params["prop_oc_docl"]),
        prop_oc_pocr=float(model_params["prop_oc_pocr"]),
        prop_oc_pocl=float(model_params["prop_oc_pocl"]),
        # Multiplier on the OC loading forcing (`forcing["carbon"]`, built from
        # oc_load_file.csv's "oc" concentration column times "discharge" --
        # see `provide_carbon()`). Scaling `oc` itself before that product is
        # computed would give exactly the same result (multiplication is
        # linear/commutative here), so applying it to the forcing in
        # `full_step` instead is equivalent without needing to touch
        # `provide_carbon`/`oc_load_file.csv` parsing at all. Not a column
        # the original numpy `run_wq_model` reads -- JAX/calibration-only,
        # default 1.0 (no change from the measured concentration), added so
        # the overall scale of the carbon loading -- often one of the more
        # uncertain boundary conditions -- can be treated as an unknown and
        # calibrated like any other parameter.
        oc_load_factor=float(model_params.get("oc_load_factor", 1.0)),
        settling_rate_labile=float(model_params["settling_rate_labile"]) / 86400,
        settling_rate_refractory=float(model_params["settling_rate_refractory"]) / 86400,
        f_sod=float(model_params["f_sod"]),
        d_thick=float(model_params["d_thick"]),
        beta=0.8,  # not in model_params.csv; matches run_wq_model's own default, never overridden by run_M3.py
    )
