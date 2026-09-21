"""Train a force field using volume-scaling and dimer data."""

import argparse
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
from scalej.targets.nagl_mbis import (
    _interpolate_charges,
    _inject_charges,
    _precompute_molecule_charges,
    _build_charge_tensors,
)
import torch
from openff.toolkit import ForceField
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


def assign_mbis_charges(mol, alpha=0.5):
    """
    Assign NAGL-MBIS charges to a molecule.

    Parameters
    ----------
    mol : openff.toolkit.topology.Molecule
        The molecule to which the charges will be assigned.

    Returns
    -------
    openff.toolkit.topology.Molecule
        The molecule with assigned NAGL-MBIS charges.
    """
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


def _inject_nagl_mbis_charges(
    force_field,
    alpha,
    all_tensor_systems,
    smiles_per_topology,
    molecule_charges,
):
    """Inject interpolated NAGL-MBIS charges into a TensorForceField.

    Computes q = (1 - alpha) * q_gas + alpha * q_water for every unique
    molecule and writes the result into the Electrostatics potential
    parameters of *force_field* in-place.
    """
    e_pot = force_field.potentials_by_type["Electrostatics"]
    n_global = e_pot.parameters.shape[0]
    dtype = e_pot.parameters.dtype
    device = e_pot.parameters.device

    # Build a single pair of (q_gas, q_water) tensors covering all topologies.
    all_topos, all_smiles = [], []
    seen_smiles: set[str] = set()
    for entry_id, smi_list in smiles_per_topology.items():
        topo = all_tensor_systems[entry_id]
        topos = topo.topologies if hasattr(topo, "topologies") else [topo]
        for t, s in zip(topos, smi_list):
            if s not in seen_smiles:
                all_topos.append(t)
                all_smiles.append(s)
                seen_smiles.add(s)

    q_gas, q_water = _build_charge_tensors(
        all_topos, all_smiles, molecule_charges, n_global, dtype, device,
    )
    charges = _interpolate_charges(alpha, q_gas, q_water)
    _inject_charges(force_field, charges)


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
            assign_mbis_charges if args.nagl_mbis else None,
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
            scales={"alpha": 0.25, "beta": 1.0},
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
    alpha = None

    if args.nagl_mbis:
        from naglmbis.models import load_charge_model

        LOGGER.info(
            f"Creating NAGL-MBIS closure with initial alpha={args.alpha_init:.4f} ..."
        )
        gas_model = load_charge_model("nagl-gas-charge-dipole-esp-wb-default")
        water_model = load_charge_model("nagl-water-charge-dipole-esp-wb-default")

        alpha = torch.tensor(
            args.alpha_init, dtype=params.dtype, device=params.device, requires_grad=True
        )

        # Build smiles_per_topology from the metadata.
        smiles_per_topology = {
            m["name"]: [c["smiles"] for c in m["components"]]
            for m in vscaling_metadata["runs"]
        }

        # Pre-compute gas/water charges for charge injection at evaluation time.
        all_unique_smiles: set[str] = set()
        for smi_list in smiles_per_topology.values():
            all_unique_smiles.update(smi_list)
        molecule_charges = _precompute_molecule_charges(
            list(all_unique_smiles), gas_model, water_model
        )

        closures["condensed"] = scalej.targets.nagl_mbis.default_closure(
            trainable=trainable,
            topologies=all_tensor_systems,
            dataset=vscaling_dataset,
            gas_model=gas_model,
            water_model=water_model,
            smiles_per_topology=smiles_per_topology,
            alpha=alpha,
            reference=args.reference,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            batch_size=args.batch_size,
            normalize=True,
            energy_cutoff=args.energy_cutoff,
        )
    else:
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
    initial_ff = trainable.to_force_field(params.detach().abs())
    if alpha is not None:
        _inject_nagl_mbis_charges(initial_ff, alpha.detach(), all_tensor_systems,
                                  smiles_per_topology, molecule_charges)
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
        force_field=initial_ff,
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
    params.data += torch.randn_like(params.data) * 0.001
    # Print perturbed parameters.
    descent.utils.reporting.print_force_field_summary(
        trainable.to_force_field(params.detach().abs())
    )

    # Run training loop.
    extra_params = None
    post_step_fn = None
    if alpha is not None:
        extra_params = [{"params": [alpha], "lr": args.alpha_lr}]
        #post_step_fn = lambda: alpha.data.clamp_(0.0, 1.0)

    losses = run_training_loop(
        params=params,
        closure=closure,
        trainable=trainable,
        n_epochs=args.n_epochs,
        lr=args.lr,
        clamp=True,
        extra_params=extra_params,
        post_step_fn=post_step_fn,
    )
    LOGGER.info("Training complete.")
    if alpha is not None:
        LOGGER.info(f"Final alpha: {alpha.item():.6f}")

    # Save trained force field (inject trained alpha charges if applicable).
    trained_ff = trainable.to_force_field(params.detach().abs())
    if alpha is not None:
        _inject_nagl_mbis_charges(trained_ff, alpha.detach(), all_tensor_systems,
                                  smiles_per_topology, molecule_charges)
        LOGGER.info(
            f"Injected NAGL-MBIS charges with trained alpha={alpha.item():.6f} "
            "into the final force field."
        )
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

    # Save alpha if it was optimised.
    if alpha is not None:
        alpha_path = output_dir / "alpha.pt"
        torch.save(alpha.detach().cpu(), alpha_path)
        LOGGER.info(f"Saved optimal alpha ({alpha.item():.6f}) -> '{alpha_path}'")

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
    p.add_argument("--alpha-init", type=float, default=0.5, metavar="A",
                   help="Initial value of the NAGL-MBIS alpha mixing parameter.")
    p.add_argument("--alpha-lr", type=float, default=1e-3, metavar="LR",
                   help="Learning rate for the NAGL-MBIS alpha parameter.")
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
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
