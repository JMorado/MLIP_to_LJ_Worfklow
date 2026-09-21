"""benchmark_batch.py

Batch thermodynamic benchmark runner that reads run metadata from a
``combined_dataset_meta.json`` produced by ``aggregate_dataset.py`` (step 03)
and runs density/Hvap/Hmix predictions for every selected run.

Run selection
-------------
By default **all** runs in the meta file are used.  Two mutually exclusive
filters are available:

* ``--max-run RUN_NAME``  
  Keep only runs whose trailing numeric suffix is <= the suffix of *RUN_NAME*.
  For example ``--max-run run_0050`` keeps ``run_0000`` through ``run_0050``.

* ``--runs-file FILE``  
  A plain-text file with one run name per line (blank lines and ``#`` comments
  are ignored).  Only those runs are processed.

Typical usage
-------------
    # All runs up to run_0050
    python benchmark_batch.py \\
        --dataset-meta /path/to/combined_dataset_meta.json \\
        --forcefield openff-2.0.0.offxml \\
        --max-run run_0050 \\
        --output-dir output/batch_benchmark

    # Specific runs listed in a file
    python benchmark_batch.py \\
        --dataset-meta /path/to/combined_dataset_meta.json \\
        --forcefield openff-2.0.0.offxml \\
        --runs-file my_runs.txt \\
        --output-dir output/batch_benchmark

Outputs
-------
For each run ``<name>``, results are written to ``<output-dir>/<name>/``:

* ``predictions.json`` – predicted values
* ``summary.txt``      – human-readable per-run summary

A combined ``batch_summary.csv`` is written to ``<output-dir>/``.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import logging
import traceback
from pathlib import Path

import descent.targets.thermo
import descent.utils.molecule
import openmm
import smee.mm
from descent.targets.thermo import SimulationConfig as ThermoSimulationConfig
from openff.toolkit import Molecule

from scalej.simulation import create_system_from_smiles

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# TIP3P partial charges (O = -0.834 e, H = +0.417 e)
_TIP3P_CHARGES = {8: -0.834, 1: 0.417}
_WATER_MOL = Molecule.from_smiles("O")


def assign_mbis_charges(mol, library_charges=False) -> None:
    """Assign NAGL MBIS partial charges to an OpenFF Molecule (in-place)."""
    from openff.toolkit import Quantity, unit

    if library_charges:
        if mol.is_isomorphic_with(_WATER_MOL):
            charges = [_TIP3P_CHARGES[a.atomic_number] for a in mol.atoms]
            mol.partial_charges = Quantity(charges, unit.elementary_charge)
            return

    from naglmbis.models import load_charge_model
    from naglmbis.models.base_model import ComputePartialPolarised

    gas_model = load_charge_model(charge_model="nagl-gas-charge-dipole-esp-wb-default")
    water_model = load_charge_model(
        charge_model="nagl-water-charge-dipole-esp-wb-default"
    )
    polarised_model = ComputePartialPolarised(
        model_gas=gas_model, model_water=water_model, alpha=0.55
    )
    charges = polarised_model.compute_polarised_charges(mol.to_rdkit())
    charges = charges.detach().numpy().astype(float).squeeze()
    mol.partial_charges = Quantity(charges, unit.elementary_charge)
    mol._normalize_partial_charges()


# ---------------------------------------------------------------------------
# CSV conditions loader
# ---------------------------------------------------------------------------

def _load_conditions_csv(csv_path: Path) -> dict[int, dict]:
    """Load per-run thermodynamic conditions from a CSV file.

    The CSV must have ``Temperature (K)`` and ``Pressure (kPa)`` columns
    (standard OpenFF Evaluator format).  Row *i* (0-based, after the header)
    corresponds to ``run_<i:04d>``.

    Returns a dict mapping the 0-based row index to a dict with keys
    ``temperature`` (K) and ``pressure`` (bar).
    """
    conditions: dict[int, dict] = {}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            temp_k = float(row["Temperature (K)"])
            press_kpa = float(row["Pressure (kPa)"])
            conditions[i] = {
                "temperature": temp_k,
                "pressure": press_kpa / 100.0,  # kPa -> bar
            }
    return conditions


# ---------------------------------------------------------------------------
# Run selection helpers
# ---------------------------------------------------------------------------

def _parse_runs_file(path: Path) -> list[str]:
    """Read run names from a text file (one per line; # comments stripped)."""
    names: list[str] = []
    with open(path) as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if line:
                names.append(line)
    return names


def _numeric_suffix(run_name: str) -> int | None:
    """Return the trailing integer of a run name, e.g. 'run_0042' -> 42."""
    suffix = run_name.split("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _select_runs(
    all_runs: list[dict],
    max_run: str | None,
    runs_file: Path | None,
) -> list[dict]:
    """Filter the list of run metadata dicts according to the selection criteria."""
    if max_run is not None and runs_file is not None:
        raise ValueError("--max-run and --runs-file are mutually exclusive.")

    if runs_file is not None:
        wanted = set(_parse_runs_file(runs_file))
        log.info(f"Loaded {len(wanted)} run name(s) from '{runs_file}'.")
        selected = [r for r in all_runs if r["run"] in wanted]
        missing = wanted - {r["run"] for r in selected}
        if missing:
            log.warning(
                f"{len(missing)} run name(s) from the file were not found in "
                f"the dataset meta: {sorted(missing)}"
            )
        return selected

    if max_run is not None:
        limit = _numeric_suffix(max_run)
        if limit is None:
            raise ValueError(
                f"--max-run value '{max_run}' does not have a numeric suffix "
                f"(expected something like 'run_0050')."
            )
        selected = [
            r for r in all_runs
            if (s := _numeric_suffix(r["run"])) is not None and s <= limit
        ]
        log.info(
            f"--max-run {max_run}: keeping {len(selected)} / {len(all_runs)} run(s)."
        )
        return selected

    # No filter – use all runs
    log.info(f"No run filter specified – using all {len(all_runs)} run(s).")
    return list(all_runs)


# ---------------------------------------------------------------------------
# Single-run benchmark
# ---------------------------------------------------------------------------

def _benchmark_one(run_meta: dict, args: argparse.Namespace) -> list[dict]:
    """Run the benchmark for a single entry from combined_dataset_meta.json.

    Parameters
    ----------
    run_meta:
        One element of the ``"runs"`` list in combined_dataset_meta.json.
    args:
        Parsed CLI args (provides force field, sim config, flags, etc.).

    Returns
    -------
    list[dict]
        Results list (same schema as benchmark_scalej.py).
    """
    run_name = run_meta["run"]
    name = run_meta["name"]
    smiles = run_meta["smiles"]
    components = list(run_meta["components"])  # shallow copy

    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute mole fractions from nmol counts
    total_nmol = sum(c["nmol"] for c in components)
    for comp in components:
        comp["x"] = comp["nmol"] / total_nmol

    # Per-run T/P from CSV (injected by benchmark_batch) or global defaults
    temperature = run_meta.get("_temperature", args.temperature)
    pressure = run_meta.get("_pressure", args.pressure)

    log.info("=" * 60)
    log.info(f"Benchmark run : {name}")
    log.info(f"SMILES        : {smiles}")
    log.info(f"Temperature   : {temperature} K")
    log.info(f"Pressure      : {pressure} bar")
    for comp in components:
        log.info(
            f"  component   : {comp['smiles']}  x{comp['nmol']}"
            f"  (x={comp['x']:.4f})"
        )

    # Create system using scalej
    smiles_list = [c["smiles"] for c in components]
    nmol_list = [c["nmol"] for c in components]

    charge_callback = (
        functools.partial(assign_mbis_charges, library_charges=args.library_charges)
        if args.nagl_mbis
        else None
    )
    if charge_callback:
        log.info("Using NAGL MBIS partial charges.")

    _tensor_system, tensor_ff, topologies = create_system_from_smiles(
        smiles_list=smiles_list,
        nmol_list=nmol_list,
        forcefield_name=args.forcefield,
        charge_assignment_callback=charge_callback,
    )
    tensor_ff = tensor_ff.to(args.device)

    # Build topology dict keyed by mapped SMILES
    topology_map = {}
    for comp, topo in zip(components, topologies):
        mapped = descent.utils.molecule.map_smiles(comp["smiles"])
        comp["mapped_smiles"] = mapped
        topology_map[mapped] = topo.to(args.device)

    # Simulation config for descent.targets.thermo
    custom_config = {
        "bulk": ThermoSimulationConfig(
            max_mols=args.n_molecules,
            gen_coords=smee.mm.GenerateCoordsConfig(),
            equilibrate=[
                smee.mm.MinimizationConfig(),
                smee.mm.SimulationConfig(
                    temperature=temperature * openmm.unit.kelvin,
                    pressure=None,
                    n_steps=args.equilibration_steps,
                    timestep=args.timestep_fs * openmm.unit.femtosecond,
                ),
                smee.mm.SimulationConfig(
                    temperature=temperature * openmm.unit.kelvin,
                    pressure=pressure * openmm.unit.bar,
                    n_steps=args.equilibration_steps,
                    timestep=args.timestep_fs * openmm.unit.femtosecond,
                ),
            ],
            production=smee.mm.SimulationConfig(
                temperature=temperature * openmm.unit.kelvin,
                pressure=pressure * openmm.unit.bar,
                n_steps=args.production_steps,
                timestep=args.timestep_fs * openmm.unit.femtosecond,
            ),
            production_frequency=args.report_interval,
        )
    }

    # Build descent thermo dataset entries
    smiles_a = components[0]["smiles"]
    x_a = components[0]["x"]
    smiles_b = components[1]["smiles"] if len(components) > 1 else None
    x_b = components[1]["x"] if len(components) > 1 else None
    is_mixture = len(components) > 1

    entries: list[dict] = []
    results_list: list[dict] = []

    if not args.no_density:
        entries.append(
            {
                "type": "density",
                "smiles_a": smiles_a,
                "x_a": x_a,
                "smiles_b": smiles_b,
                "x_b": x_b,
                "temperature": temperature,
                "pressure": pressure,
                "value": 0.0,
                "std": 0.0,
                "units": "g/mL",
                "source": None,
            }
        )

    if not args.no_hvap:
        hvap_type = "hmix" if is_mixture else "hvap"
        entries.append(
            {
                "type": hvap_type,
                "smiles_a": smiles_a,
                "x_a": x_a,
                "smiles_b": smiles_b,
                "x_b": x_b,
                "temperature": temperature,
                "pressure": pressure,
                "value": 0.0,
                "std": 0.0,
                "units": "kcal/mol",
                "source": None,
            }
        )

    if not entries:
        raise ValueError("All property types disabled – nothing to predict.")

    log.info(f"Running predictions for {len(entries)} propert(ies) ...")
    dataset = descent.targets.thermo.create_dataset(*entries)
    ref_vals, ref_stds, pred_vals, pred_stds = descent.targets.thermo.predict(
        dataset,
        tensor_ff,
        topology_map,
        output_dir=output_dir / "predictions",
        cached_dir=output_dir / "cache",
        verbose=True,
        simulation_config=custom_config,
    )

    for i, entry in enumerate(entries):
        results_list.append(
            {
                "type": entry["type"],
                "units": entry["units"],
                "reference": None,
                "reference_std": None,
                "predicted": float(pred_vals[i]),
                "predicted_std": float(pred_stds[i]),
            }
        )

    if not results_list:
        raise ValueError("All property types disabled – nothing to predict.")

    # ------------------------------------------------------------------ #
    # Save per-run outputs
    # ------------------------------------------------------------------ #
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as fh:
        json.dump(
            {"name": name, "smiles": smiles, "results": results_list}, fh, indent=2
        )
    log.info(f"Saved predictions -> '{predictions_path}'")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as fh:
        fh.write("=== SCALeJ Thermodynamic Benchmark Summary ===\n\n")
        fh.write(f"Run name      : {name}\n")
        fh.write(f"SMILES        : {smiles}\n")
        fh.write(f"Force field   : {args.forcefield}\n")
        fh.write(f"Temperature   : {temperature} K\n")
        fh.write(
            f"Pressure      : {pressure} bar\n\n"
        )
        fh.write(
            f"{'Property':<24} {'Units':<10} {'Predicted':>12} {'Pred. Std':>12}\n"
        )
        fh.write("-" * 62 + "\n")
        for r in results_list:
            fh.write(
                f"{r['type']:<24} {r['units']:<10}"
                f" {r['predicted']:>12.4f} {r['predicted_std']:>12.4f}\n"
            )
    log.info(f"Saved summary  -> '{summary_path}'")

    return results_list


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def benchmark_batch(args: argparse.Namespace) -> None:
    dataset_meta_path = Path(args.dataset_meta)
    log.info(f"Loading dataset metadata from '{dataset_meta_path}' ...")
    with open(dataset_meta_path) as fh:
        dataset_meta = json.load(fh)

    all_runs: list[dict] = dataset_meta.get("runs", [])
    if not all_runs:
        raise ValueError("No 'runs' entries found in the dataset metadata.")

    log.info(f"Dataset meta: {dataset_meta.get('n_runs', len(all_runs))} total run(s).")

    selected_runs = _select_runs(
        all_runs,
        max_run=args.max_run,
        runs_file=Path(args.runs_file) if args.runs_file else None,
    )

    if not selected_runs:
        raise ValueError("No runs matched the selection criteria.")

    log.info(f"Selected {len(selected_runs)} run(s) for benchmarking.")

    # Inject per-run T/P from CSV if provided
    if args.conditions_csv:
        conditions = _load_conditions_csv(Path(args.conditions_csv))
        log.info(
            f"Loaded thermodynamic conditions for {len(conditions)} row(s) "
            f"from '{args.conditions_csv}'."
        )
        for rm in selected_runs:
            run_idx = _numeric_suffix(rm["run"])
            if run_idx is not None and run_idx in conditions:
                rm["_temperature"] = conditions[run_idx]["temperature"]
                rm["_pressure"] = conditions[run_idx]["pressure"]
            else:
                log.warning(
                    f"No conditions row for '{rm['run']}' in CSV; "
                    f"using global defaults ({args.temperature} K, {args.pressure} bar)."
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Iterate over selected runs
    # ------------------------------------------------------------------ #
    batch_rows: list[dict] = []
    failed: list[str] = []

    for i, run_meta in enumerate(selected_runs, 1):
        run_name = run_meta["run"]
        log.info(f"[{i}/{len(selected_runs)}] Starting run '{run_name}' ...")
        try:
            results = _benchmark_one(run_meta, args)
            for r in results:
                batch_rows.append(
                    {
                        "run": run_name,
                        "name": run_meta["name"],
                        "smiles": run_meta["smiles"],
                        "type": r["type"],
                        "units": r["units"],
                        "predicted": r["predicted"],
                        "predicted_std": r["predicted_std"],
                    }
                )
            log.info(f"[{i}/{len(selected_runs)}] Finished run '{run_name}'.")
        except Exception:
            log.error(
                f"[{i}/{len(selected_runs)}] Run '{run_name}' FAILED:\n"
                + traceback.format_exc()
            )
            failed.append(run_name)

    # ------------------------------------------------------------------ #
    # Write combined batch summary CSV
    # ------------------------------------------------------------------ #
    csv_path = output_dir / "batch_summary.csv"
    if batch_rows:
        fieldnames = ["run", "name", "smiles", "type", "units", "predicted", "predicted_std"]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(batch_rows)
        log.info(f"Saved batch summary -> '{csv_path}'")
    else:
        log.warning("No successful results to write to batch_summary.csv.")

    # ------------------------------------------------------------------ #
    # Final report
    # ------------------------------------------------------------------ #
    n_ok = len(selected_runs) - len(failed)
    log.info(
        f"Batch complete: {n_ok}/{len(selected_runs)} run(s) succeeded"
        + (f", {len(failed)} failed: {failed}" if failed else ".")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Batch thermodynamic benchmark runner.  Reads run metadata from a "
            "combined_dataset_meta.json produced by aggregate_dataset.py and runs "
            "density/Hvap/Hmix predictions for every selected run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Dataset / run selection ----------------------------------------
    sel = p.add_argument_group("dataset & run selection")
    sel.add_argument(
        "--dataset-meta",
        required=True,
        metavar="JSON",
        help="Path to combined_dataset_meta.json.",
    )

    excl = sel.add_mutually_exclusive_group()
    excl.add_argument(
        "--max-run",
        default=None,
        metavar="RUN_NAME",
        help=(
            "Only benchmark runs up to and including this run name "
            "(by numeric suffix, e.g. 'run_0050')."
        ),
    )
    excl.add_argument(
        "--runs-file",
        default=None,
        metavar="FILE",
        help=(
            "Text file with run names to benchmark, one per line "
            "(blank lines and # comments are ignored)."
        ),
    )

    sel.add_argument(
        "--conditions-csv",
        default=None,
        metavar="CSV",
        help=(
            "CSV file with per-run 'Temperature (K)' and 'Pressure (kPa)' columns "
            "(OpenFF Evaluator format). Row i corresponds to run_<i:04d>. "
            "When provided, overrides --temperature/--pressure per run."
        ),
    )

    # ---- Force field ---------------------------------------------------
    p.add_argument(
        "--forcefield",
        required=True,
        metavar="OFFXML",
        help="OpenFF force field name or path (e.g. 'openff-2.0.0.offxml').",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/batch_benchmark"),
        metavar="DIR",
        help="Root output directory; per-run results go into subdirectories.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for force field tensors (default: cpu).",
    )

    # ---- Thermodynamic state -------------------------------------------
    thermo = p.add_argument_group("thermodynamic state")
    thermo.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        metavar="K",
        help="Simulation temperature in Kelvin.",
    )
    thermo.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        metavar="bar",
        help="Simulation pressure in bar.",
    )

    # ---- Simulation config ---------------------------------------------
    sim = p.add_argument_group("simulation config")
    sim.add_argument(
        "--n-molecules",
        type=int,
        default=1000,
        metavar="N",
        help="Number of molecules to pack.",
    )
    sim.add_argument(
        "--equilibration-steps",
        type=int,
        default=100_000,
        metavar="N",
        help="Number of equilibration steps.",
    )
    sim.add_argument(
        "--production-steps",
        type=int,
        default=1_000_000,
        metavar="N",
        help="Number of production steps.",
    )
    sim.add_argument(
        "--timestep-fs",
        type=float,
        default=1.0,
        metavar="FS",
        help="Integration timestep in femtoseconds.",
    )
    sim.add_argument(
        "--report-interval",
        type=int,
        default=2000,
        metavar="N",
        help="Frame reporting interval in steps.",
    )

    # ---- Property flags ------------------------------------------------
    props = p.add_argument_group("property flags")
    props.add_argument(
        "--no-density",
        action="store_true",
        help="Skip density prediction.",
    )
    props.add_argument(
        "--no-hvap",
        action="store_true",
        help="Skip Hvap/Hmix prediction.",
    )
    props.add_argument(
        "--nagl-mbis",
        action="store_true",
        help="Use NAGL MBIS partial charges instead of AM1-BCC.",
    )
    props.add_argument(
        "--library-charges",
        action="store_true",
        help="Assign TIP3P library charges to water when using NAGL-MBIS.",
    )

    return p.parse_args()


if __name__ == "__main__":
    benchmark_batch(_parse_args())
