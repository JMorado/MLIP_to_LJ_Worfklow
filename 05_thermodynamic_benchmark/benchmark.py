"""benchmark.py

Run thermodynamic property predictions (density, Hvap) for a mixture using an
OpenFF force field and ``descent.targets.thermo``.

The system is described by a JSON file.  The ``energies_forces_meta.json``
produced by ``calculate_mlp_data.py`` in step 02 can be used directly as
``--input`` – it already has the required schema:

    {
      "name": "run_0000",
      "smiles": "CO",
      "components": [
        {"smiles": "CO", "nmol": 500}
      ]
    }

Typical usage
-------------
    python benchmark.py \\
        --input ../02_calculate_mlp_data/output/run_0000/energies_forces_meta.json \\
        --forcefield openff-2.0.0.offxml \\
        --output-dir output/benchmark/run_0000

Outputs written to ``--output-dir``:
* ``predictions.json``  – predicted density and Hvap values
* ``summary.txt``       – human-readable benchmark summary
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import descent.targets.thermo
import descent.utils.molecule
import openmm
import smee.mm
from descent.targets.thermo import SimulationConfig

from scalej.simulation import create_system_from_smiles

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _load_config(config_path: Path) -> dict:
    """Load and validate the JSON run configuration.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to the JSON file describing the system.

    Returns
    -------
    dict
        Parsed configuration dict with keys ``name``, ``smiles``,
        and ``components``.
    """
    with open(config_path) as fh:
        config = json.load(fh)

    required = {"name", "smiles", "components"}
    missing = required - config.keys()
    if missing:
        raise ValueError(
            f"JSON config is missing required key(s): {', '.join(sorted(missing))}"
        )

    for i, comp in enumerate(config["components"]):
        if "smiles" not in comp or "nmol" not in comp:
            raise ValueError(f"Component {i} must have 'smiles' and 'nmol' keys.")

    return config


def assign_mbis_charges(mol) -> None:
    """Assign NAGL MBIS partial charges to an OpenFF Molecule (in-place)."""
    from openff.toolkit import Quantity, unit

    from naglmbis.models import load_charge_model
    from naglmbis.models.base_model import ComputePartialPolarised

    gas_model = load_charge_model(charge_model="nagl-gas-charge-dipole-esp-wb-default")
    water_model = load_charge_model(
        charge_model="nagl-water-charge-dipole-esp-wb-default"
    )
    polarised_model = ComputePartialPolarised(
        model_gas=gas_model, model_water=water_model, alpha=0.5
    )
    charges = polarised_model.compute_polarised_charges(mol.to_rdkit())
    charges = charges.detach().numpy().astype(float).squeeze()
    mol.partial_charges = Quantity(charges, unit.elementary_charge)
    mol._normalize_partial_charges()


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


def _run_index_from_name(name: str) -> int | None:
    """Extract a trailing integer from a run name, e.g. 'run_0042' -> 42."""
    suffix = name.split("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def benchmark(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = config["name"]
    smiles = config["smiles"]
    components = config["components"]

    # Resolve per-run T/P: CSV overrides > CLI defaults
    temperature = args.temperature
    pressure = args.pressure

    if args.conditions_csv:
        conditions = _load_conditions_csv(Path(args.conditions_csv))
        run_idx = _run_index_from_name(name)
        if run_idx is not None and run_idx in conditions:
            temperature = conditions[run_idx]["temperature"]
            pressure = conditions[run_idx]["pressure"]
            log.info(
                f"Conditions from CSV row {run_idx}: "
                f"T={temperature} K, P={pressure} bar"
            )
        else:
            log.warning(
                f"No conditions row for '{name}' (index {run_idx}) in CSV; "
                f"using CLI defaults ({temperature} K, {pressure} bar)."
            )

    # Compute mole fractions from nmol counts
    total_nmol = sum(c["nmol"] for c in components)
    for comp in components:
        comp["x"] = comp["nmol"] / total_nmol

    log.info(f"Benchmark run : {name}")
    log.info(f"SMILES        : {smiles}")
    for comp in components:
        comp["mapped_smiles"] = descent.utils.molecule.map_smiles(comp["smiles"])
        log.info(
            f"  component   : {comp['smiles']}  x{comp['nmol']}  (x={comp['x']:.4f})"
        )
        log.info(f"  mapped      : {comp['mapped_smiles']}")

    log.info(f"Loading force field '{args.forcefield}' ...")

    charge_callback = assign_mbis_charges if args.nagl_mbis else None
    if charge_callback:
        log.info("Using NAGL MBIS partial charges.")

    smiles_list = [c["smiles"] for c in components]
    nmol_list = [c["nmol"] for c in components]

    _tensor_system, tensor_ff, topologies = create_system_from_smiles(
        smiles_list=smiles_list,
        nmol_list=nmol_list,
        forcefield_name=args.forcefield,
        charge_assignment_callback=charge_callback,
    )
    tensor_ff = tensor_ff.to(args.device)

    log.info(f"Force field device: {args.device}")

    # Build topology dict keyed by mapped SMILES, as required by descent.
    topology_map = {
        c["mapped_smiles"]: t.to(args.device) for c, t in zip(components, topologies)
    }

    # ------------------------------------------------------------------ #
    # Config
    # ------------------------------------------------------------------ #
    custom_config = {
        "bulk": SimulationConfig(
            max_mols=1000,
            gen_coords=smee.mm.GenerateCoordsConfig(),
            equilibrate=[
                # smee.mm.MinimizationConfig(),
                # short NVT equilibration simulation
                smee.mm.SimulationConfig(
                    temperature=temperature * openmm.unit.kelvin,
                    pressure=None,
                    n_steps=50000,
                    timestep=1.0 * openmm.unit.femtosecond,
                ),
                smee.mm.SimulationConfig(
                    temperature=temperature * openmm.unit.kelvin,
                    pressure=pressure * openmm.unit.bar,
                    n_steps=100000,
                    timestep=1.0 * openmm.unit.femtosecond,
                ),
            ],
            production=smee.mm.SimulationConfig(
                temperature=temperature * openmm.unit.kelvin,
                pressure=pressure * openmm.unit.bar,
                n_steps=1000000,
                timestep=1.0 * openmm.unit.femtosecond,
            ),
            production_frequency=2000,
        )
    }

    # ------------------------------------------------------------------ #
    # Build descent.targets.thermo dataset entries
    # ------------------------------------------------------------------ #
    # descent.create_dataset calls map_smiles internally, so pass plain SMILES here.
    # topology_map uses mapped SMILES keys (already set above).
    smiles_a = components[0]["smiles"]
    x_a = components[0]["x"]
    smiles_b = components[1]["smiles"] if len(components) > 1 else None
    x_b = components[1]["x"] if len(components) > 1 else None

    entries = []

    if not args.no_density:
        density_entry = {
            "type": "density",
            "smiles_a": smiles_a,
            "x_a": x_a,
            "smiles_b": smiles_b,
            "x_b": x_b,
            "temperature": temperature,
            "pressure": pressure,
            "value": args.density_ref,
            "std": args.density_std,
            "units": "g/mL",
            "source": None,
        }
        entries.append(density_entry)
        log.info(
            f"Density target : {args.density_ref} ± {args.density_std} g/mL "
            f"@ {temperature} K / {pressure} bar"
        )

    if not args.no_hvap:
        # Use 'hmix' for mixtures (hvap requires a vacuum sim, only valid for pure systems)
        hvap_type = "hmix" if smiles_b is not None else "hvap"
        hvap_entry = {
            "type": hvap_type,
            "smiles_a": smiles_a,
            "x_a": x_a,
            "smiles_b": smiles_b,
            "x_b": x_b,
            "temperature": temperature,
            "pressure": pressure,
            "value": args.hvap_ref,
            "std": args.hvap_std,
            "units": "kcal/mol",
            "source": None,
        }
        entries.append(hvap_entry)
        log.info(
            f"{hvap_type.upper()} target   : {args.hvap_ref} ± {args.hvap_std} kcal/mol "
            f"@ {temperature} K / {pressure} bar"
        )

    if not entries:
        raise ValueError("All property types disabled – nothing to predict.")

    # ------------------------------------------------------------------ #
    # Run predictions
    # ------------------------------------------------------------------ #
    cache_dir = output_dir / "cache"
    pred_dir = output_dir / "predictions"

    log.info("Creating thermo dataset ...")
    dataset = descent.targets.thermo.create_dataset(*entries)

    log.info("Running predictions ...")
    # predict() returns (reference, reference_std, predicted, predicted_std)
    # each is a 1-D tensor with one entry per dataset row.
    ref_vals, ref_stds, pred_vals, pred_stds = descent.targets.thermo.predict(
        dataset,
        tensor_ff,
        topology_map,
        output_dir=pred_dir,
        cached_dir=cache_dir,
        verbose=True,
        simulation_config=custom_config,
    )

    log.info("Predictions complete.")

    # Pair predictions with their dataset entry metadata
    results_list = []
    for i, entry in enumerate(entries):
        results_list.append(
            {
                "type": entry["type"],
                "units": entry["units"],
                "reference": float(ref_vals[i]),
                "reference_std": float(ref_stds[i]),
                "predicted": float(pred_vals[i]),
                "predicted_std": float(pred_stds[i]),
            }
        )

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as fh:
        json.dump(
            {"name": name, "smiles": smiles, "results": results_list}, fh, indent=2
        )
    log.info(f"Saved predictions -> '{predictions_path}'")

    # Human-readable summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as fh:
        fh.write("=== SCALeJ Thermodynamic Benchmark Summary ===\n\n")
        fh.write(f"Run name      : {name}\n")
        fh.write(f"SMILES        : {smiles}\n")
        fh.write(f"Force field   : {args.forcefield}\n")
        fh.write(f"Temperature   : {temperature} K\n")
        fh.write(f"Pressure      : {pressure} bar\n\n")
        fh.write(
            f"{'Property':<24} {'Units':<10} {'Predicted':>12} {'Reference':>12} {'|Error|':>10}\n"
        )
        fh.write("-" * 72 + "\n")
        for r in results_list:
            pred = r["predicted"]
            ref = r["reference"]
            err = abs(pred - ref) if ref == ref else float("nan")  # nan-safe
            ref_str = f"{ref:.4f}" if ref == ref else "n/a"
            fh.write(
                f"{r['type']:<24} {r['units']:<10} {pred:>12.4f} {ref_str:>12} {err:>10.4f}\n"
            )
    log.info(f"Saved summary  -> '{summary_path}'")
    log.info("All done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run thermodynamic property predictions (density, Hvap) for a system "
            "described by a JSON config file using a trained LJ force field."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        metavar="JSON",
        help=(
            "Path to the JSON run config file with keys 'name', 'smiles', "
            "and 'components' (list of {smiles, nmol})."
        ),
    )
    p.add_argument(
        "--forcefield",
        required=True,
        metavar="OFFXML",
        help="OpenFF force field name or path (e.g. 'openff-2.0.0.offxml').",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for force field tensors (default: cpu).",
    )
    p.add_argument(
        "--conditions-csv",
        default=None,
        metavar="CSV",
        help=(
            "CSV file with per-run 'Temperature (K)' and 'Pressure (kPa)' columns "
            "(OpenFF Evaluator format). Row i corresponds to run_<i:04d>. "
            "When provided, overrides --temperature/--pressure for the run."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/benchmark"),
        metavar="DIR",
        help="Directory for predictions and summary (default: output/benchmark).",
    )

    p.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        metavar="K",
        help="Simulation temperature in Kelvin (default: 298.15).",
    )
    p.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        metavar="bar",
        help="Simulation pressure in bar (default: 1.0).",
    )

    p.add_argument(
        "--density-ref",
        type=float,
        default=0.000,
        metavar="G_ML",
        help="Experimental reference density in g/mL (used for error reporting).",
    )
    p.add_argument(
        "--density-std",
        type=float,
        default=0.000,
        metavar="G_ML",
        help="Uncertainty on the reference density in g/mL (default: 0.001).",
    )
    p.add_argument(
        "--hvap-ref",
        type=float,
        default=0.000,
        metavar="KCAL",
        help="Experimental reference Hvap in kcal/mol (used for error reporting).",
    )
    p.add_argument(
        "--hvap-std",
        type=float,
        default=0.000,
        metavar="KCAL",
        help="Uncertainty on the reference Hvap in kcal/mol (default: 0.001).",
    )

    p.add_argument(
        "--no-density",
        action="store_true",
        help="Skip density prediction.",
    )
    p.add_argument(
        "--no-hvap",
        action="store_true",
        help="Skip Hvap prediction.",
    )
    p.add_argument(
        "--nagl-mbis",
        action="store_true",
        help="Use NAGL MBIS partial charges instead of AM1-BCC",
    )

    return p.parse_args()


if __name__ == "__main__":
    benchmark(_parse_args())
