"""full_step (standard) and full_step_mcl (experimental MCL copy) must stay
physically identical: with no override, or an override that returns
kz_process unchanged, both give the same state. Run from the project root:
    python -m pytest tests/test_full_step_mcl.py   (or: python tests/test_full_step_mcl.py)
"""
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from processBased_lakeModel_functions import (  # noqa: E402
    get_hypsography, provide_meteorology, initial_profile, wq_initial_profile, provide_phosphorus,
    provide_carbon, get_lake_config, get_model_params, get_run_config, get_ice_and_snow,
)
from jax_lakeModel_functions import (  # noqa: E402
    make_initial_state_full, full_step, full_step_mcl, default_params, default_geometry_wq,
)
from run_M3_jax import build_forcing_series, add_wq_forcing_series  # noqa: E402

N_STEPS = 200


def _setup(lake="Ravn"):
    cwd = os.getcwd()
    os.chdir(ROOT / lake)
    try:
        lc = get_lake_config("./lake_config.csv", 1); mp = get_model_params("./model_params.csv", 1)
        rc = get_run_config("./run_config.csv", 1); ias = get_ice_and_snow("./ice_and_snow.csv", 1)
        wf = float(lc["WindSpeed"]); nx = int(rc["nx"]); dt = float(rc["dt"]); dx = float(rc["dx"])
        area, depth, volume, hw = get_hypsography(hypsofile=rc["hypso_ini_file"], dx=dx, nx=nx,
                                                  outflow_depth=float(lc["outflow_depth"]))
        start = pd.Timestamp(rc["start_time"]); st = np.arange(1, (N_STEPS + 1) * dt, dt)[:N_STEPS]
        met = provide_meteorology(meteofile=rc["meteo_ini_file"], windfactor=wf, lat=lc["Latitude"],
                                  lon=lc["Longitude"], elev=lc["Elevation"], startDate=start)
        u_ini = initial_profile(initfile=rc["u_ini_file"], nx=nx, dx=dx, depth=depth, startDate=start)
        wq = wq_initial_profile(initfile=rc["wq_ini_file"], nx=nx, dx=dx, depth=depth, volume=volume, startDate=start)
        tp = provide_phosphorus(tpfile=rc["tp_ini_file"], startingDate=start, startTime=1)
        cd = provide_carbon(ocloadfile=rc["oc_load_file"], startingDate=start, startTime=1).dropna(subset=["oc"])
        forcing = add_wq_forcing_series(build_forcing_series(met, st, wf), tp, cd, st)
        geom = default_geometry_wq(area=area, depth=depth, volume=volume, dx=dx, dt=dt,
                                   latitude=float(lc["Latitude"]), altitude=float(lc["Elevation"]),
                                   hypso_weight=hw, mean_depth=float(np.sum(volume) / np.max(area)),
                                   hydro_res_time_hr=float(mp["hydro_res_time"]) * 8760)
        params = default_params(mp, ias, diffusion_method=str(rc.get("diffusion_method", "hendersonSellers")))
        tb = lambda x: x.strip().lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x)
        s0 = make_initial_state_full(
            jnp.asarray(deepcopy(u_ini)), jnp.asarray(wq[0]), jnp.asarray(wq[1] * 0.75), jnp.asarray(wq[1] * 0.25),
            jnp.asarray(0.5 * volume), jnp.asarray(0.5 * volume), nx,
            ice=tb(ias["ice"]), Hi=ias["Hi"], Hs=ias["Hs"], Hsi=ias["Hsi"], iceT=ias["iceT"], rho_snow=ias["rho_snow"])
    finally:
        os.chdir(cwd)
    return s0, forcing, geom, params


def _run(step, s0, forcing):
    def body(s, f):
        ns, _ = step(s, f)
        return ns, (ns.u, ns.kz, ns.o2, ns.pocl, ns.uvel)
    return jax.jit(lambda s, f: lax.scan(body, s, f))(s0, forcing)[1]


def test_full_step_mcl_matches_full_step():
    s0, forcing, geom, params = _setup()
    ref = _run(lambda s, f: full_step(s, f, geom, params), s0, forcing)
    none = _run(lambda s, f: full_step_mcl(s, f, geom, params), s0, forcing)
    ident = _run(lambda s, f: full_step_mcl(
        s, f, geom, params, kz_override=lambda kz, u, ice, dens, fo, ri: kz), s0, forcing)
    for a, b, c in zip(ref, none, ident):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(np.asarray(a), np.asarray(c), rtol=1e-12, atol=0)


def test_full_step_has_no_override_hook():
    import inspect
    assert "kz_override" not in inspect.signature(full_step).parameters
    assert "kz_override" in inspect.signature(full_step_mcl).parameters


if __name__ == "__main__":
    test_full_step_has_no_override_hook()
    test_full_step_mcl_matches_full_step()
    print("OK: full_step and full_step_mcl agree")
