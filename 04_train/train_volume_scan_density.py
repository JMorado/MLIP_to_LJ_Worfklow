"""Train force field parameters against experimental densities via volume-scan configurations.

Instead of running NPT simulations, this script predicts density from
pre-computed volume-scan configurations using a differentiable soft-argmin.
The volume-scan density closure can be combined with the standard
volume-scaling energy/force closure via ``--vscaling-dataset``.

Experimental targets (density, temperature, pressure) are read from a CSV
file (e.g. ``sage-training-set.csv``) and matched to volume-scan runs by
SMILES.

Typical usage
-------------
    python train_volume_scan_density.py \
        --vscaling-dataset /path/to/aggregated_dataset \
        --csv /path/to/sage-training-set.csv \
        --forcefield de-force-1.0.3.offxml \
        --beta-eff 5e-3 \
        --n-epochs 200 \
        --output-dir output/vscan_density
"""

import argparse
import logging
from pathlib import Path

import datasets
import descent
import descent.targets.energy
import descent.train
import descent.utils.loss
import descent.utils.reporting
import pandas as pd
import scalej.targets.condensed_ddp
import smee
import torch
from openff.toolkit import ForceField

from scalej.analysis import evaluate_force_field, save_prediction_parquet
from scalej.data import (
    export_forcefield_to_offxml,
    load_json,
    save_object,
)
from scalej.simulation.systems import create_composite_system
from scalej.targets import normalize_closure_weights
from scalej.targets.volume_scan_density import (
    create_dataset as create_vscan_dataset,
    create_entries_from_csv,
    extract_smiles as extract_vscan_smiles,
    volume_scan_density_closure,
)
from scalej.targets.volume_scan_hvap import (
    create_dataset as create_hvap_dataset,
    create_entries_from_csv as create_hvap_entries_from_csv,
    extract_smiles as extract_hvap_smiles,
    volume_scan_hvap_closure,
)
from scalej.train import run_training_loop

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assign_mbis_charges(mol, alpha=0.5):
    """Assign NAGL-MBIS partial charges to a molecule (in-place)."""
    from naglmbis.models import load_charge_model
    from naglmbis.models.base_model import ComputePartialPolarised
    from openff.toolkit import Quantity, unit

    gas = load_charge_model("nagl-gas-charge-dipole-esp-wb-default")
    water = load_charge_model("nagl-water-charge-dipole-esp-wb-default")
    model = ComputePartialPolarised(model_gas=gas, model_water=water, alpha=alpha)
    charges = model.compute_polarised_charges(mol.to_rdkit())[:, 0].detach().cpu()
    mol.partial_charges = Quantity(charges, unit.elementary_charge)
    mol._normalize_partial_charges()
    return mol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vscaling_dataset_dir = Path(args.vscaling_dataset)
    charge_callback = assign_mbis_charges if args.nagl_mbis else None

    # ------------------------------------------------------------------
    # 1. Load volume-scaling dataset & metadata
    # ------------------------------------------------------------------
    vscaling_dataset = datasets.load_from_disk(
        vscaling_dataset_dir / "combined_data"
    )
    vscaling_dataset.set_format("torch")
    vscaling_metadata = load_json(
        vscaling_dataset_dir / "combined_dataset_meta.json"
    )
    vscaling_systems_config = [
        {"name": m["name"], "components": m["components"]}
        for m in vscaling_metadata["runs"]
    ]

    LOGGER.info(
        f"Volume-scaling dataset: {len(vscaling_dataset)} entry(ies), "
        f"{len(descent.targets.energy.extract_smiles(vscaling_dataset))} "
        "unique molecule(s)."
    )

    # ------------------------------------------------------------------
    # 2. Create volume-scan density entries from CSV
    # ------------------------------------------------------------------
    vscan_entries = create_entries_from_csv(
        csv_path=args.csv,
        vscaling_dataset=vscaling_dataset,
        vscaling_metadata=vscaling_metadata,
        stride=args.vscan_stride,
    )
    if not vscan_entries:
        # Log the SMILES on each side to help debug.
        from scalej.targets.volume_scan_density import extract_smiles as _extract
        run_smiles = sorted({r["smiles"] for r in vscaling_metadata["runs"]})
        LOGGER.error(f"Run SMILES ({len(run_smiles)}): {run_smiles[:10]} ...")
        import pandas as _pd
        csv_smiles = sorted(_pd.read_csv(args.csv)["Component 1"].dropna().unique()[:10])
        LOGGER.error(f"CSV SMILES (first 10): {csv_smiles}")
        raise RuntimeError(
            "No volume-scan density entries could be created. "
            "Check that CSV SMILES match the runs in the aggregated dataset."
        )
    vscan_dataset = create_vscan_dataset(vscan_entries)
    vscan_smiles = extract_vscan_smiles(vscan_dataset)
    LOGGER.info(
        f"Volume-scan density: {len(vscan_entries)} entry(ies), "
        f"{len(vscan_smiles)} unique molecule(s)."
    )

    # ------------------------------------------------------------------
    # 2b. (Optional) Create volume-scan ΔH_vap entries from CSV
    # ------------------------------------------------------------------
    hvap_entries = []
    hvap_dataset_obj = None

    if args.hvap_weight > 0.0:
        hvap_entries = create_hvap_entries_from_csv(
            csv_path=args.csv,
            vscaling_dataset=vscaling_dataset,
            vscaling_metadata=vscaling_metadata,
            stride=args.vscan_stride,
        )
        if hvap_entries:
            hvap_dataset_obj = create_hvap_dataset(hvap_entries)
            hvap_smiles = extract_hvap_smiles(hvap_dataset_obj)
            LOGGER.info(
                f"Volume-scan ΔH_vap: {len(hvap_entries)} entry(ies), "
                f"{len(hvap_smiles)} unique molecule(s)."
            )
        else:
            LOGGER.warning(
                "--hvap-weight > 0 but no ΔH_vap entries could be created "
                "from the CSV.  Skipping ΔH_vap closure."
            )

    # ------------------------------------------------------------------
    # 3. Build composite system (covers all molecules)
    # ------------------------------------------------------------------
    systems_config = vscaling_systems_config  # already contains all runs

    composite_tf, _, _, all_tensor_systems, _ = create_composite_system(
        systems_config, args.forcefield, charge_callback,
    )
    # Keep only vdW + Electrostatics potentials.
    composite_tf.potentials = [
        p for p in composite_tf.potentials
        if p.type in ("Electrostatics", "vdW")
    ]
    descent.utils.reporting.print_force_field_summary(composite_tf)

    composite_tf = composite_tf.to(args.device)
    all_tensor_systems = {
        k: v.to(args.device) for k, v in all_tensor_systems.items()
    }

    # ------------------------------------------------------------------
    # 4. Create trainable
    # ------------------------------------------------------------------
    vdw_parameters = {
        "vdW": descent.train.ParameterConfig(
            cols=args.parameters,
            scales={"epsilon": 10, "r_min": 1},
        )
    }
    trainable = descent.train.Trainable(
        force_field=composite_tf,
        parameters=vdw_parameters,
        attributes={},
    )
    params = trainable.to_values().to(args.device)

    # perturb parameters
    params.data += torch.randn_like(params.data).abs() * 0.2

    LOGGER.info(f"Trainable: {params.numel()} parameter(s).")
    descent.utils.reporting.print_force_field_summary(
        trainable.to_force_field(params.detach().abs())
    )

    # ------------------------------------------------------------------
    # 5. Build closures
    # ------------------------------------------------------------------
    closures: dict[str, descent.utils.loss.ClosureFn] = {}
    weights: dict[str, float] = {}

    beta_eff = None if args.physical_beta else args.beta_eff
    if args.physical_beta:
        LOGGER.info("Using physical β = 1/(R·T) per entry (ESS will be ≈ 1).")

    # Volume-scan density closure.
    closures["vscan_density"] = volume_scan_density_closure(
        trainable=trainable,
        topologies=all_tensor_systems,
        dataset=vscan_dataset,
        beta_eff=beta_eff,
        pressure_atm=1.0,
        batch_size=args.vscan_batch_size,
    )
    weights["vscan_density"] = args.density_weight

    # Optional: ΔH_vap closure.
    if hvap_dataset_obj is not None and len(hvap_entries) > 0:
        # Build gas-phase topology lookup: entry_id → single-molecule TensorTopology.
        gas_topologies: dict[str, object] = {}
        for entry in hvap_entries:
            eid = entry["id"]
            tensor_sys = all_tensor_systems[eid]
            gas_topologies[eid] = tensor_sys.topologies[0]

        closures["vscan_hvap"] = volume_scan_hvap_closure(
            trainable=trainable,
            topologies=all_tensor_systems,
            gas_topologies=gas_topologies,
            dataset=hvap_dataset_obj,
            beta_eff=beta_eff,
            pressure_atm=1.0,
        )
        weights["vscan_hvap"] = args.hvap_weight

    # Optional: energy/force closure on the same vscaling data.
    if not args.density_only:
        closures["condensed"] = scalej.targets.condensed_ddp.ddp_closure(
            trainable=trainable,
            topologies=all_tensor_systems,
            dataset=vscaling_dataset,
            reference=args.reference,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            batch_size=args.batch_size,
            energy_cutoff=args.energy_cutoff,
        )
        weights["condensed"] = 1.0

    # Auto-weight closures so each starts at ~1.
    if args.auto_weights:
        LOGGER.info("Auto-weighting closures so each starts at ~1 ...")
        weights = normalize_closure_weights(closures, params, weights)

    closure = descent.utils.loss.combine_closures(
        closures, weights=weights, verbose=True,
    )

    # ------------------------------------------------------------------
    # 6. Log initial metrics
    # ------------------------------------------------------------------
    # Initial density prediction.
    with torch.no_grad():
        with torch.enable_grad():
            vscan_loss, *_ = closures["vscan_density"](
                params, compute_gradient=False,
            )
    vscan_info = closures["vscan_density"].last_losses
    LOGGER.info(f"Initial vscan_density loss: {vscan_loss.item():.4e}")
    for key, val in vscan_info.items():
        if key.endswith("/density_pred") or key.endswith("/ess"):
            LOGGER.info(f"  {key}: {val:.4f}")

    # Initial ΔH_vap prediction.
    if "vscan_hvap" in closures:
        with torch.no_grad():
            with torch.enable_grad():
                hvap_loss, *_ = closures["vscan_hvap"](
                    params, compute_gradient=False,
                )
        hvap_info = closures["vscan_hvap"].last_losses
        LOGGER.info(f"Initial vscan_hvap loss: {hvap_loss.item():.4e}")
        for key, val in hvap_info.items():
            if key.endswith("/hvap_pred") or key.endswith("/ess"):
                LOGGER.info(f"  {key}: {val:.4f}")

    # Initial energy/force evaluation (if condensed closure is active).
    scale_factors_map = {}
    stride_meta = vscaling_metadata.get("stride", 1)
    scale_factors_map = {
        run["name"]: run["scale_factors"][::stride_meta]
        for run in vscaling_metadata["runs"]
        if "scale_factors" in run
    }

    LOGGER.info("Evaluating initial force field on vscaling set ...")
    initial_prediction, (
        init_e_mae, init_e_rmse, init_e_r2,
        init_f_mae, init_f_rmse, init_f_r2,
    ) = evaluate_force_field(
        force_field=trainable.to_force_field(params.detach().abs()),
        dataset=vscaling_dataset,
        tensor_systems=all_tensor_systems,
        reference=args.reference,
        energy_cutoff=args.energy_cutoff,
    )
    LOGGER.info(
        f"Initial energy -> MAE: {init_e_mae:.4e} | "
        f"RMSE: {init_e_rmse:.4e} | R2: {init_e_r2:.4f}"
    )
    LOGGER.info(
        f"Initial forces -> MAE: {init_f_mae:.4e} | "
        f"RMSE: {init_f_rmse:.4e} | R2: {init_f_r2:.4f}"
    )
    save_prediction_parquet(
        initial_prediction, output_dir, "initial", scale_factors_map,
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    LOGGER.info(
        f"Training: {args.n_epochs} epoch(s), lr={args.lr:.2e}, "
        f"beta_eff={beta_eff!r}, density_weight={args.density_weight}, "
        f"hvap_weight={args.hvap_weight}"
    )

    losses = run_training_loop(
        params=params,
        closure=closure,
        trainable=trainable,
        n_epochs=args.n_epochs,
        lr=args.lr,
        clamp=True,
    )
    LOGGER.info("Training complete.")

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    trained_ff = trainable.to_force_field(params.detach().abs())
    descent.utils.reporting.print_force_field_summary(trained_ff)
    save_object(trained_ff, output_dir / "trained_forcefield.pt")
    LOGGER.info(f"Saved trained force field -> '{output_dir / 'trained_forcefield.pt'}'")

    offxml_path = output_dir / "trained_forcefield.offxml"
    ff = ForceField(args.forcefield, load_plugins=True)
    export_forcefield_to_offxml(ff, trained_ff, offxml_path)
    LOGGER.info(f"Saved OFFXML -> '{offxml_path}'")

    pd.DataFrame({"loss": losses}).to_parquet(output_dir / "loss_history.parquet")

    # Final density prediction.
    with torch.no_grad():
        with torch.enable_grad():
            vscan_loss, *_ = closures["vscan_density"](
                params, compute_gradient=False,
            )
    vscan_info = closures["vscan_density"].last_losses
    LOGGER.info(f"Final vscan_density loss: {vscan_loss.item():.4e}")
    for key, val in vscan_info.items():
        if key.endswith("/density_pred") or key.endswith("/ess"):
            LOGGER.info(f"  {key}: {val:.4f}")

    # Save per-entry density results.
    density_rows = []
    for key, val in vscan_info.items():
        if key.endswith("/density_pred"):
            entry_id = key.replace("/density_pred", "")
            density_rows.append({
                "entry_id": entry_id,
                "density_pred": val,
                "density_target": vscan_info.get(f"{entry_id}/density_target", None),
                "ess": vscan_info.get(f"{entry_id}/ess", None),
            })
    pd.DataFrame(density_rows).to_parquet(output_dir / "density_results.parquet")

    # Final ΔH_vap prediction.
    if "vscan_hvap" in closures:
        with torch.no_grad():
            with torch.enable_grad():
                hvap_loss, *_ = closures["vscan_hvap"](
                    params, compute_gradient=False,
                )
        hvap_info = closures["vscan_hvap"].last_losses
        LOGGER.info(f"Final vscan_hvap loss: {hvap_loss.item():.4e}")
        for key, val in hvap_info.items():
            if key.endswith("/hvap_pred") or key.endswith("/ess"):
                LOGGER.info(f"  {key}: {val:.4f}")

        hvap_rows = []
        for key, val in hvap_info.items():
            if key.endswith("/hvap_pred"):
                entry_id = key.replace("/hvap_pred", "")
                hvap_rows.append({
                    "entry_id": entry_id,
                    "hvap_pred": val,
                    "hvap_target": hvap_info.get(f"{entry_id}/hvap_target", None),
                    "ess": hvap_info.get(f"{entry_id}/ess", None),
                })
        pd.DataFrame(hvap_rows).to_parquet(output_dir / "hvap_results.parquet")

    # ------------------------------------------------------------------
    # 9. (Optional) Evaluate final FF on vscaling dataset
    # ------------------------------------------------------------------
    LOGGER.info("Evaluating trained force field on vscaling set ...")
    final_prediction, (
        final_e_mae, final_e_rmse, final_e_r2,
        final_f_mae, final_f_rmse, final_f_r2,
    ) = evaluate_force_field(
        force_field=trained_ff.to(args.device),
        dataset=vscaling_dataset,
        tensor_systems=all_tensor_systems,
        reference=args.reference,
        energy_cutoff=args.energy_cutoff,
    )
    LOGGER.info(
        f"Final energy -> MAE: {final_e_mae:.4e} | "
        f"RMSE: {final_e_rmse:.4e} | R2: {final_e_r2:.4f}"
    )
    LOGGER.info(
        f"Final forces -> MAE: {final_f_mae:.4e} | "
        f"RMSE: {final_f_rmse:.4e} | R2: {final_f_r2:.4f}"
    )
    save_prediction_parquet(
        final_prediction, output_dir, "final", scale_factors_map,
    )

    pd.DataFrame([
        {
            "stage": "initial",
            "energy_mae": init_e_mae, "energy_rmse": init_e_rmse,
            "energy_r2": init_e_r2,
            "forces_mae": init_f_mae, "forces_rmse": init_f_rmse,
            "forces_r2": init_f_r2,
        },
        {
            "stage": "final",
            "energy_mae": final_e_mae, "energy_rmse": final_e_rmse,
            "energy_r2": final_e_r2,
            "forces_mae": final_f_mae, "forces_rmse": final_f_rmse,
            "forces_r2": final_f_r2,
        },
    ]).to_parquet(output_dir / "metrics.parquet")

    LOGGER.info("All done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train LJ parameters against experimental densities from "
            "volume-scan configurations (optionally combined with "
            "energy/force targets)."
        ),
    )

    # ---- Data sources ----
    p.add_argument(
        "--vscaling-dataset", type=Path, required=True, metavar="DIR",
        help="Directory with combined_data/ and combined_dataset_meta.json.",
    )
    p.add_argument(
        "--csv", required=True, metavar="CSV",
        help="CSV file with experimental densities (Component 1, "
             "Temperature (K), Pressure (kPa), Density Value (g / ml)).",
    )
    p.add_argument(
        "--vscan-stride", type=int, default=1,
        help="Keep every N-th configuration from each volume scan (default: 1).",
    )

    # ---- Force field ----
    p.add_argument(
        "--forcefield", default="de-force-1.0.3.offxml", metavar="FF",
        help="Force field to optimise (OFFXML).",
    )
    p.add_argument("--nagl-mbis", action="store_true")

    # ---- Density closure options ----
    p.add_argument(
        "--beta-eff", type=float, default=5e-3,
        help="Effective inverse temperature for soft-argmin [(kcal/mol)^-1]. "
             "Controls ESS; typical range 1e-3 to 1e-2 (default: 5e-3). "
             "Ignored when --physical-beta is set.",
    )
    p.add_argument(
        "--physical-beta", action="store_true",
        help="Use the physical inverse temperature β = 1/(R·T) per entry "
             "instead of a fixed β_eff.  WARNING: ESS will be ≈ 1 and "
             "gradients will be very noisy.",
    )
    p.add_argument(
        "--density-weight", type=float, default=1.0,
        help="Relative weight for the density closure (default: 1.0).",
    )
    p.add_argument(
        "--hvap-weight", type=float, default=0.0,
        help="Relative weight for the ΔH_vap closure (default: 0.0 = off). "
             "Set > 0 to train against enthalpies of vaporisation read from "
             "the 'EnthalpyOfMixing Value (kJ / mol)' column for pure-component "
             "rows in the CSV.",
    )
    p.add_argument(
        "--vscan-batch-size", type=int, default=32,
        help="Number of volume-scan configs to batch (default: 32).",
    )

    # ---- Energy/force closure options ----
    p.add_argument(
        "--density-only", action="store_true",
        help="Train only against density (skip energy/force closure).",
    )
    p.add_argument("--energy-weight", type=float, default=1.0)
    p.add_argument("--force-weight", type=float, default=1.0)
    p.add_argument(
        "--reference", default="infinite", choices=["mean", "min", "infinite"],
    )
    p.add_argument("--energy-cutoff", type=float, default=20.0, metavar="KCAL")
    p.add_argument("--batch-size", type=int, default=2)

    # ---- Training ----
    p.add_argument(
        "--parameters", nargs="+", default=["epsilon", "r_min"], metavar="COL",
    )
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--auto-weights", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--output-dir", type=Path, default=Path("train_vscan_density_output"))

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
