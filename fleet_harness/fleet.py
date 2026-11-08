"""Fleet runs: house inventory, strata and groups, per-house result files.

Everything here except `run_and_save` works on the configs or on saved results only, so it
needs no EnergyPlus or Gurobi.

Result layout, one folder per run (e.g. results/fleet/baseline_realistic_2023/):
    house_{n}_power.parquet       one row per 15-min step, stamped at step START (UTC)
                                  raw asset kW (as extract_results) + derived columns
    house_{n}_attributes.parquet  logged attributes, stamped at step END (UTC)
    house_{n}_solves.parquet      Gurobi status and runtime per MPC step
    house_{n}_signal.csv          real day-ahead price and the control signal the MPC planned
                                  on, per 15-min step over the simulated window (UTC), for plots
    house_{n}.json                settings, versions, KPIs, energy totals. Written LAST:
                                  its presence means the house is done.
    house_{n}.error.txt           traceback of a failed run (removed when it later succeeds)
"""

import datetime
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from fleet_harness.runner import PAPER_ROOT

REPO_ROOT = Path(__file__).resolve().parent.parent
FLEET_DIR = REPO_ROOT / "results" / "fleet"

# simugrid sign convention (checked on the house-0 full-year run): PublicGrid_0 > 0 is
# offtake; consuming assets are < 0; SolarPv_0 is >= 0; each step's columns sum to 0.
ASSET_COLUMNS = {
    "appliance_kw": "Consumer_0",  # uncontrolled household load, an input CSV
    "heating_kw": "EnergyPlus_0",  # heat pump (EnergyPlus building)
    "dhw_kw": "WaterHeater_0",  # electric water heater
    "ev_kw": "Charger_0",
}
FLEXIBLE = ["heating_kw", "dhw_kw", "ev_kw"]
STRATUM_KEYS = ["has_heating", "has_ev", "has_pv"]


# ---------------------------------------------------------------------------
# Inventory, strata, groups (configs only)
# ---------------------------------------------------------------------------

def house_inventory(tot_houses=500):
    """One row per house, read from the JSON configs (no simulation).

    has_heating means the house has an EnergyPlus building (heat pump). In the 500-house
    set the water heater always comes with it (447 houses each); the 53 houses without
    either keep only appliances, plus EV and/or PV.
    """
    folder = PAPER_ROOT / "data" / f"houses_belgium_{tot_houses}"
    rows = []
    for house in range(tot_houses):
        cfg = json.loads((folder / f"house_{house}.json").read_text())
        assets = {a["name"]: a for a in cfg["Assets"].values()}
        env = cfg["Environments"]["Environment_0"]
        eplus = assets.get("EnergyPlus_0", {})
        rows.append(
            {
                "house": house,
                "region": env["region"],
                "has_heating": "EnergyPlus_0" in assets,
                "has_dhw": "WaterHeater_0" in assets,
                "has_ev": "Charger_0" in assets,
                "has_pv": "SolarPv_0" in assets,
                "building": Path(eplus["idf_model"]).stem if eplus else "",
                "appliance_profile": Path(env["Consumer_0_electric"]).stem,
                "charger_kw": assets.get("Charger_0", {}).get("max_charge_cp", 0.0),
                "pv_kwp": assets.get("SolarPv_0", {}).get("size", 0.0),
                "dhw_volume_m3": assets.get("WaterHeater_0", {}).get("volume", 0.0),
                "capacity_tariff": env["capacity_tariff"],
                "offtake_extra": env["offtake_extra"],
            }
        )
    inv = pd.DataFrame(rows).set_index("house")
    inv["stratum"] = stratum_labels(inv)
    return inv


def stratum_labels(inv):
    """Label like 'heat+ev+nopv' from the three stratum flags."""
    def label(row):
        return "+".join(
            [
                "heat" if row.has_heating else "noheat",
                "ev" if row.has_ev else "noev",
                "pv" if row.has_pv else "nopv",
            ]
        )
    return inv.apply(label, axis=1)


def _allocate(counts, n):
    """Split n over cells in proportion to counts (largest remainder).

    Every non-empty cell gets at least 1 when n allows, so small strata (e.g. the 3
    no-heating, no-EV PV houses) are represented. Never gives a cell more than it has.
    """
    counts = counts[counts > 0]
    if n > counts.sum():
        raise ValueError(f"asked for {n} houses, only {counts.sum()} available")
    floor = min(1, n // len(counts)) if len(counts) else 0
    alloc = pd.Series(floor, index=counts.index)
    remaining = n - alloc.sum()
    exact = counts / counts.sum() * remaining
    extra = np.floor(exact).astype(int)
    alloc += extra
    remainder = (exact - extra).sort_values(ascending=False)
    for cell in remainder.index[: n - alloc.sum()]:
        alloc[cell] += 1
    # cap at availability, then hand any overflow to cells with room, biggest first
    over = (alloc - counts).clip(lower=0)
    alloc -= over
    spare = over.sum()
    for cell in counts.sort_values(ascending=False).index:
        if spare == 0:
            break
        room = counts[cell] - alloc[cell]
        take = min(room, spare)
        alloc[cell] += take
        spare -= take
    return alloc.astype(int)


def select_groups(inv, n_train, n_val, seed=0):
    """Pick disjoint, stratified training and validation groups.

    Strata = (has_heating, has_ev, has_pv). Returns a Series house -> 'train' / 'val' /
    'rest'. The 2023 test set is all houses (PLAN.md Phase 5), so 'rest' is not a test set.
    """
    rng = np.random.default_rng(seed)
    group = pd.Series("rest", index=inv.index, name="group")
    available = inv.copy()
    for name, n in [("train", n_train), ("val", n_val)]:
        alloc = _allocate(available["stratum"].value_counts(), n)
        for cell, k in alloc.items():
            members = available.index[available["stratum"] == cell].to_numpy()
            chosen = rng.choice(members, size=k, replace=False)
            group[chosen] = name
        available = available.drop(group.index[group == name], errors="ignore")
    return group


# ---------------------------------------------------------------------------
# One run -> tables
# ---------------------------------------------------------------------------

def power_table(power):
    """Raw per-asset kW (extract_results()["power"]) plus derived, non-negative columns.

    grid_kw      PublicGrid_0 (+ offtake, - injection)
    offtake_kw   max(grid, 0)          injection_kw  max(-grid, 0)
    appliance_kw, heating_kw, dhw_kw, ev_kw   consumption of each asset (0 if absent)
    pv_kw        PV production (0 if absent)
    flexible_kw  heating + dhw + ev    consumption_kw  appliance + flexible
    """
    out = power.copy()
    grid = power["PublicGrid_0"]
    out["grid_kw"] = grid
    out["offtake_kw"] = grid.clip(lower=0)
    out["injection_kw"] = (-grid).clip(lower=0)
    for col, asset in ASSET_COLUMNS.items():
        out[col] = -power[asset] if asset in power else 0.0
    out["pv_kw"] = power["SolarPv_0"] if "SolarPv_0" in power else 0.0
    out["flexible_kw"] = out[FLEXIBLE].sum(axis=1)
    out["consumption_kw"] = out["appliance_kw"] + out["flexible_kw"]
    out.index.name = "datetime"
    return out


def energy_totals(table, time_step_h):
    """kWh totals of the derived columns."""
    cols = ["offtake_kw", "injection_kw", "consumption_kw", "flexible_kw", "pv_kw",
            *ASSET_COLUMNS]
    return {c.replace("_kw", "_kwh"): float(table[c].sum() * time_step_h) for c in cols}


def solve_stats(solves):
    """Counts of Gurobi statuses (2 optimal, 9 time limit, 17 memory limit) and runtimes."""
    if solves.empty:
        return {"n_solves": 0}
    status = solves["status"]
    return {
        "n_solves": int(len(solves)),
        "n_optimal": int((status == 2).sum()),
        "n_time_limit": int((status == 9).sum()),
        "n_mem_limit": int((status == 17).sum()),
        "n_other": int((~status.isin([2, 9, 17])).sum()),
        "runtime_mean_s": float(solves["runtime_s"].mean()),
        "runtime_max_s": float(solves["runtime_s"].max()),
    }


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def house_paths(out_dir, house):
    out_dir = Path(out_dir)
    return {
        "power": out_dir / f"house_{house}_power.parquet",
        "attributes": out_dir / f"house_{house}_attributes.parquet",
        "solves": out_dir / f"house_{house}_solves.parquet",
        "signal": out_dir / f"house_{house}_signal.csv",
        "meta": out_dir / f"house_{house}.json",
        "error": out_dir / f"house_{house}.error.txt",
    }


def is_done(out_dir, house):
    return house_paths(out_dir, house)["meta"].exists()


# Settings that must be identical for a saved result to count as "done" for a new run.
RUN_KEYS = ("year", "mode", "days", "gurobi_threads", "disable_enforced", "tot_houses")


def settings_mismatch(out_dir, house, settings):
    """{key: (saved, wanted)} for the RUN_KEYS (and signal name) where the saved result of
    a done house differs from `settings`; empty if they match. Stops e.g. a 7-day smoke
    test saved in a folder from being taken as done by the full-year run."""
    meta = json.loads(house_paths(out_dir, house)["meta"].read_text())
    wanted = {k: settings.get(k) for k in RUN_KEYS}
    wanted["signal"] = settings.get("signal_name", "real_price")
    return {k: (meta.get(k), v) for k, v in wanted.items() if meta.get(k) != v}


def _atomic_write(path, write):
    """Write via a temp file in the same folder + os.replace (safe on shared filesystems)."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_house_result(out_dir, house, result, meta):
    """Write one run's tables, then its JSON (the done marker). Returns the JSON dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = house_paths(out_dir, house)
    table = power_table(result["power"])
    _atomic_write(paths["power"], table.to_parquet)
    _atomic_write(paths["attributes"], result["attributes"].to_parquet)
    _atomic_write(paths["solves"], result["solves"].to_parquet)
    signal = result.get("signal")
    if signal is not None:
        _atomic_write(paths["signal"], signal.to_csv)

    record = {
        **meta,
        "house": house,
        "time_step_h": result["time_step_h"],
        "start": str(table.index[0]),
        "end": str(table.index[-1]),
        "n_steps": int(len(table)),
        "kpis": result["kpis"],
        "energy_kwh": energy_totals(table, result["time_step_h"]),
        "solves": solve_stats(result["solves"]),
    }
    _atomic_write(paths["meta"], lambda p: Path(p).write_text(json.dumps(record, indent=1)))
    if paths["error"].exists():
        paths["error"].unlink()
    return record


def load_house(out_dir, house):
    """(power table, meta dict) of a finished house."""
    paths = house_paths(out_dir, house)
    return pd.read_parquet(paths["power"]), json.loads(paths["meta"].read_text())


def load_signal(path):
    """Read a saved signal CSV (house_{n}_signal.csv) back with its tz-aware UTC index."""
    table = pd.read_csv(path, index_col="datetime")
    table.index = pd.to_datetime(table.index, utc=True)
    return table


def load_run_summary(out_dir):
    """One row per finished house: settings, KPIs, energy totals and solve stats, flattened."""
    rows = []
    for meta_path in sorted(Path(out_dir).glob("house_*.json")):
        rec = json.loads(meta_path.read_text())
        row = {k: v for k, v in rec.items() if not isinstance(v, dict)}
        for section in ("kpis", "energy_kwh", "solves"):
            row.update({f"{section}.{k}": v for k, v in rec.get(section, {}).items()})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("house").sort_index()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _git(*args, cwd=REPO_ROOT):
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def provenance():
    """Code versions and machine, stored with every result."""
    import gurobipy

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        # untracked files count too: new, uncommitted code must not look like git_commit
        "git_dirty": bool(_git("status", "--porcelain")),
        "paper_commit": _git("rev-parse", "HEAD", cwd=PAPER_ROOT),
        "python": platform.python_version(),
        "gurobi": ".".join(map(str, gurobipy.gurobi.version())),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_and_save(house, out_dir, settings, prov):
    """Run one house with run_isolated and save it. Runs in a worker process.

    settings: keyword arguments for run_one_house_mpc (year, mode, days, gurobi_threads,
    disable_enforced, signal_series ...); signal_name labels the signal in the JSON.
    Never raises: a failure is written to house_{n}.error.txt and reported in the return
    value, so one bad house doesn't stop the others.
    """
    from fleet_harness.mpc_runner import calibration_year, run_isolated

    settings = dict(settings)
    signal_name = settings.pop("signal_name", "real_price")
    started = datetime.datetime.now(datetime.UTC)
    t0 = time.perf_counter()
    try:
        result = run_isolated(house, **settings)
        meta = {
            "signal": signal_name,
            **{k: v for k, v in settings.items() if k != "signal_series"},
            "calibration_year": calibration_year(settings["mode"], settings["year"]),
            "started_utc": started.isoformat(),
            "elapsed_s": time.perf_counter() - t0,
            **prov,
        }
        save_house_result(out_dir, house, result, meta)
        return {"house": house, "ok": True, "elapsed_s": meta["elapsed_s"]}
    except Exception:
        tb = traceback.format_exc()
        path = house_paths(out_dir, house)["error"]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path.write_text(f"started {started.isoformat()}\n{tb}")
        return {"house": house, "ok": False, "elapsed_s": time.perf_counter() - t0, "error": tb}
