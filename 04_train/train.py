"""Train a force field using volume-scaling and dimer data."""

import argparse
import functools
import logging
from pathlib import Path

import datasets
import descent
import descent.targets
import descent.targets.dimers
import descent.targets.energy
import descent.train
import descent.utils.loss
import descent.utils.reporting
import pandas as pd
import scalej.targets.condensed
import scalej.targets.condensed_ddp
import scalej.targets.dimers
import scalej.targets.nagl_mbis
import torch
from openff.toolkit import ForceField, Molecule
from scalej.analysis import evaluate_force_field, save_prediction_parquet
from scalej.data import (
    export_forcefield_to_offxml,
    load_json,
    save_object,
)
from scalej.simulation.systems import (
    create_composite_system,
)
from scalej.targets import normalize_closure_weights
from scalej.train import run_training_loop

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


# TIP3P partial charges (O = -0.834 e, H = +0.417 e)
_TIP3P_CHARGES = {8: -0.834, 1: 0.417}
_WATER_MOL = Molecule.from_smiles("O")


def assign_mbis_charges(mol, alpha=0.5, library_charges=False):
    """
    Assign NAGL-MBIS charges to a molecule.

    Parameters
    ----------
    mol : openff.toolkit.topology.Molecule
        The molecule to which the charges will be assigned.
    alpha : float
        Polarisation mixing parameter for NAGL-MBIS.
    library_charges : bool
        If True, assign TIP3P library charges to water instead of NAGL-MBIS.

    Returns
    -------
    openff.toolkit.topology.Molecule
        The molecule with assigned charges.
    """
    from openff.toolkit import Quantity, unit

    if library_charges:
        if mol.is_isomorphic_with(_WATER_MOL):
            charges = [_TIP3P_CHARGES[a.atomic_number] for a in mol.atoms]
            mol.partial_charges = Quantity(charges, unit.elementary_charge)
            return mol

    from naglmbis.models import load_charge_model
    from naglmbis.models.base_model import ComputePartialPolarised

    gas = load_charge_model("nagl-gas-charge-dipole-esp-wb-default")
    water = load_charge_model("nagl-water-charge-dipole-esp-wb-default")
    model = ComputePartialPolarised(model_gas=gas, model_water=water, alpha=alpha)
    charges = model.compute_polarised_charges(mol.to_rdkit())[:, 0].detach().cpu()
    mol.partial_charges = Quantity(charges, unit.elementary_charge)
    mol._normalize_partial_charges()
    return mol


def train(args: argparse.Namespace) -> None:
    vscaling_dataset_dir = Path(args.vscaling_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dimers dataset.
    dimer_dataset = None
    dimer_smiles: list[str] = []
    dimer_systems_config: list[dict] = []
    if args.dimer_dataset is not None:
        LOGGER.info(f"Loading dimer dataset from '{args.dimer_dataset}' ...")
        dimer_dataset = datasets.load_from_disk(str(args.dimer_dataset))
        dimer_dataset.set_format("torch")
        dimer_smiles = descent.targets.dimers.extract_smiles(dimer_dataset)
        dimer_systems_config = [
            {"name": smi, "components": [{"smiles": smi, "nmol": 1}]}
            for smi in dimer_smiles
        ]
        LOGGER.info(
            f"Dimer dataset: {len(dimer_dataset)} entry(ies), "
            f"{len(dimer_smiles)} unique molecule(s)."
        )

    # Load the volume-scaling dataset.
    vscaling_dataset = datasets.load_from_disk(vscaling_dataset_dir / "combined_data")
    vscaling_dataset.set_format("torch")
    vscaling_smiles = descent.targets.energy.extract_smiles(vscaling_dataset)
    LOGGER.info(
        f"Volume-scaling dataset: {len(vscaling_dataset)} entry(ies), "
        f"{len(vscaling_smiles)} unique molecule(s)."
    )

    # Load metadata.
    LOGGER.info(f"Loading metadata from '{vscaling_dataset_dir}' ...")
    vscaling_metadata = load_json(vscaling_dataset_dir / "combined_dataset_meta.json")
    vscaling_systems_config = [
        {"name": m["name"], "components": m["components"]}
        for m in vscaling_metadata["runs"]
    ]

    # Build composite objects.
    systems_config = vscaling_systems_config + dimer_systems_config
    composite_tf, _, composite_topologies, all_tensor_systems, _ = (
        create_composite_system(
            systems_config,
            args.forcefield,
            functools.partial(
                assign_mbis_charges, library_charges=args.library_charges
            ) if args.nagl_mbis else None,
        )
    )

    # Keep only vdW + Electrostatics potentials.
    composite_tf.potentials = [
        p for p in composite_tf.potentials if p.type in ("Electrostatics", "vdW")
    ]
    descent.utils.reporting.print_force_field_summary(composite_tf)

    # Convert to device
    composite_tf = composite_tf.to(args.device)
    all_tensor_systems = {k: v.to(args.device) for k, v in all_tensor_systems.items()}

    # Extract dimer topologies.
    dimer_topologies = {
        smi: all_tensor_systems[smi].topologies[0] for smi in dimer_smiles
    }

    # Create the trainable object.
    vdw_attributes = {
        "vdW": descent.train.AttributeConfig(
            cols=["alpha", "beta"],
            scales={"alpha": 1, "beta": 10.0},
        )
    }
    vdw_parameters = {
        "vdW": descent.train.ParameterConfig(
            cols=args.parameters,
            scales={"epsilon": 10, "r_min": 1},
        )
    }
    trainable = descent.train.Trainable(
        force_field=composite_tf,
        parameters=vdw_parameters,
        attributes=vdw_attributes,
    )

    # Print initial parameters.
    params = trainable.to_values().to(args.device)
    descent.utils.reporting.print_force_field_summary(
        trainable.to_force_field(params.detach().abs())
    )
    LOGGER.info(f"Trainable: {params.numel()} parameter(s).")

    # Create closures.
    closures, weights = {}, {}
    if args.ddp:
        closures["condensed"] = scalej.targets.condensed_ddp.ddp_closure(
            trainable=trainable,
            topologies=all_tensor_systems,
            dataset=vscaling_dataset,
            reference=args.reference,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            batch_size=args.batch_size,
            energy_cutoff=args.energy_cutoff,
            n_gpus=args.n_gpus,
        )
    else:
        closures["condensed"] = scalej.targets.condensed.default_closure(
            trainable=trainable,
            topologies=all_tensor_systems,
            dataset=vscaling_dataset,
            reference=args.reference,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            batch_size=args.batch_size,
            energy_cutoff=args.energy_cutoff,
        )
    weights["condensed"] = 1

    if args.dimer_dataset is not None:
        closures["dimers"] = scalej.targets.dimers.default_closure(
            trainable=trainable,
            topologies=dimer_topologies,
            dataset=dimer_dataset,
            reference=args.reference,
        )
        weights["dimers"] = args.dimer_weight
        LOGGER.info(f"Dimer closure weight: {args.dimer_weight}")

    # Auto-weight closures so each starts at ~1.
    if args.auto_weights:
        LOGGER.info("Auto-weighting closures so each starts at ~1 ...")
        weights = normalize_closure_weights(closures, params, weights)

    # Combine closures.
    closure = descent.utils.loss.combine_closures(
        closures, weights=weights, verbose=True
    )

    # Build scale-factor map (one entry per conformer in the strided dataset).
    stride = vscaling_metadata.get("stride", 1)
    scale_factors_map = {
        run["name"]: run["scale_factors"][::stride]
        for run in vscaling_metadata["runs"]
        if "scale_factors" in run
    }

    # Evaluate initial force field.
    LOGGER.info("Evaluating initial force field...")
    (
        initial_prediction,
        (
            init_e_mae,
            init_e_rmse,
            init_e_r2,
            init_f_mae,
            init_f_rmse,
            init_f_r2,
        ),
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
        initial_prediction, output_dir, "initial", scale_factors_map
    )

    LOGGER.info(
        f"Training: {args.n_epochs} epoch(s), lr={args.lr:.2e}, "
        f"energy_weight={args.energy_weight}, force_weight={args.force_weight}, "
        f"reference='{args.reference}'"
    )

    # Perturb parameters.
    # params.data += torch.randn_like(params.data) * 0.001
    # Print perturbed parameters.
    descent.utils.reporting.print_force_field_summary(
        trainable.to_force_field(params.detach().abs())
    )

    # Run training loop.
    losses = run_training_loop(
        params=params,
        closure=closure,
        trainable=trainable,
        n_epochs=args.n_epochs,
        lr=args.lr,
        clamp=False,
    )
    LOGGER.info("Training complete.")

    # Save trained force field.
    trained_ff = trainable.to_force_field(params.detach().abs())
    descent.utils.reporting.print_force_field_summary(trained_ff)
    save_object(trained_ff, output_dir / "trained_forcefield.pt")
    LOGGER.info(
        f"Saved trained force field -> '{output_dir / 'trained_forcefield.pt'}'"
    )

    offxml_path = output_dir / "trained_forcefield.offxml"
    ff = ForceField(args.forcefield, load_plugins=True)
    export_forcefield_to_offxml(ff, trained_ff, offxml_path)
    LOGGER.info(f"Saved OFFXML -> '{offxml_path}'")

    # Save loss history.
    pd.DataFrame({"loss": losses}).to_parquet(output_dir / "loss_history.parquet")

    # Evaluate trained force field.
    LOGGER.info("Evaluating trained force field on training set ...")
    (
        final_prediction,
        (
            final_e_mae,
            final_e_rmse,
            final_e_r2,
            final_f_mae,
            final_f_rmse,
            final_f_r2,
        ),
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
    save_prediction_parquet(final_prediction, output_dir, "final", scale_factors_map)

    # Save metrics.
    metrics_path = output_dir / "metrics.parquet"
    pd.DataFrame(
        [
            {
                "stage": "pre_perturbation",
                "energy_mae": init_e_mae,
                "energy_rmse": init_e_rmse,
                "energy_r2": init_e_r2,
                "forces_mae": init_f_mae,
                "forces_rmse": init_f_rmse,
                "forces_r2": init_f_r2,
            },
            {
                "stage": "perturbed",
                "energy_mae": init_e_mae,
                "energy_rmse": init_e_rmse,
                "energy_r2": init_e_r2,
                "forces_mae": init_f_mae,
                "forces_rmse": init_f_rmse,
                "forces_r2": init_f_r2,
            },
            {
                "stage": "final",
                "energy_mae": final_e_mae,
                "energy_rmse": final_e_rmse,
                "energy_r2": final_e_r2,
                "forces_mae": final_f_mae,
                "forces_rmse": final_f_rmse,
                "forces_r2": final_f_r2,
            },
        ]
    ).to_parquet(metrics_path)
    LOGGER.info(f"Saved metrics -> '{metrics_path}'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lennard-Jones training.")
    p.add_argument(
        "--vscaling-dataset",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory with combined_data/ and combined_dataset_meta.json.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("train_output"))
    p.add_argument("--forcefield", default="de-force-1.0.3.offxml", metavar="FF")
    p.add_argument("--nagl-mbis", action="store_true")
    p.add_argument(
        "--library-charges",
        action="store_true",
        help="Assign TIP3P library charges to water when using NAGL-MBIS.",
    )
    p.add_argument(
        "--parameters", nargs="+", default=["epsilon", "r_min"], metavar="COL"
    )
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--energy-weight", type=float, default=1.0)
    p.add_argument("--force-weight", type=float, default=1.0)
    p.add_argument(
        "--reference", default="infinite", choices=["mean", "min", "infinite"]
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--energy-cutoff", type=float, default=20.0, metavar="KCAL")
    p.add_argument("--dimer-dataset", type=Path, default=None, metavar="DIR")
    p.add_argument(
        "--dimer-weight",
        type=float,
        default=1.0,
        metavar="W",
    )
    p.add_argument("--auto-weights", action="store_true")
    p.add_argument(
        "--ddp",
        action="store_true",
        default=False,
        help="Use multi-GPU DDP closure for condensed phase (default: single GPU).",
    )
    p.add_argument(
        "--n-gpus",
        type=int,
        default=4,
        metavar="N",
        help="Number of GPUs to use when --ddp is set (default: 4).",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
