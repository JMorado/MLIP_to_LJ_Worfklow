"""calculate_mlp_data.py

Compute MLP energies and forces for the current run folder and save the
results as:

* ``energies_forces.npz``       – compressed numpy arrays (coords, box vectors,
  scale factors, energies, forces).
* ``energies_forces_meta.json`` – lightweight metadata (run name, SMILES,
  component counts).

Place this script (or a symlink to it) directly inside a ``run_*`` directory
and execute it there:

    cd /path/to/mixtures/run_0001
    python calculate_mlp_data.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
from MDAnalysis.coordinates.DCD import DCDReader
from scalej.scaling import create_scaled_configurations
from scalej.simulation import (
    compute_mlp_energies_forces,
    create_system_from_smiles,
    setup_mlp_simulation,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _parse_run_script(run_dir: Path) -> dict | None:
    """
    Parses the ``run_*.py`` script in the given directory and extracts component information.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory containing the ``run_*.py`` script to parse.

    Returns
    -------
    dict or None
        If successful, returns a dictionary with:
            components : list of dict
                List of components, each as a dict with keys 'smiles' and 'nmol'.
            smiles : str
                Dot-joined SMILES string representing the mixture.
        Returns None if parsing fails.
    """
    scripts = sorted(run_dir.glob("run_*.py"))
    if not scripts:
        log.error(f"No run_*.py file found in '{run_dir}'.")
        return None

    source = scripts[0].read_text()

    # Define functions to extract string, float, and int variables from the script source
    def _str(name: str) -> str | None:
        m = re.search(rf'^{name}\s*=\s*["\'](.*?)["\']', source, re.MULTILINE)
        return m.group(1) if m else None

    def _float(name: str) -> float | None:
        m = re.search(rf"^{name}\s*=\s*([\d.eE+\-]+)", source, re.MULTILINE)
        return float(m.group(1)) if m else None

    def _int(name: str) -> int | None:
        m = re.search(rf"^{name}\s*=\s*(\d+)", source, re.MULTILINE)
        return int(m.group(1)) if m else None

    smiles1 = _str("smiles1")
    smiles2 = _str("smiles2")
    x1 = _float("x1")
    x2 = _float("x2")
    n_mol = _int("N_MOL")

    if n_mol is None:
        log.error(f"Could not find N_MOL in '{scripts[0].name}'.")
        return None

    components: list[dict] = []
    if smiles1 and (nmol1 := int((x1 or 0.0) * n_mol)) > 0:
        components.append({"smiles": smiles1, "nmol": nmol1})
    if smiles2 and (nmol2 := int((x2 or 0.0) * n_mol)) > 0:
        components.append({"smiles": smiles2, "nmol": nmol2})

    if not components:
        log.error(f"No valid components parsed from '{scripts[0].name}'.")
        return None

    smiles = ".".join(c["smiles"] for c in components)
    log.info(f"Parsed components: {smiles}")
    return {"components": components, "smiles": smiles}


def _load_last_frames_dcd(
    filename: str | Path,
    n_frames: int = 1,
    from_end: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load coordinates and box vectors from a DCD trajectory file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the DCD file.
    n_frames : int, optional
        Number of frames to load (default: 1).
    from_end : bool, optional
        If True, load the last `n_frames` from the trajectory; otherwise, load the first `n_frames` (default: True).

    Returns
    -------
    coords_arr : numpy.ndarray
        Array of shape (n_frames, n_atoms, 3) containing atomic coordinates in Ångströms.
    box_vecs : numpy.ndarray
        Array of shape (n_frames, 3, 3) containing box vectors for each frame.

    Notes
    -----
    The box vectors are constructed as diagonal matrices from the first three box dimensions, assuming orthorhombic boxes.
    If the DCD file does not contain box information, zero vectors are returned.
    """
    dcd = DCDReader(str(filename))
    all_coords, all_boxes = [], []
    for ts in dcd:
        all_coords.append(ts.positions.copy())
        dims = ts.dimensions if ts.dimensions is not None else np.zeros(6)
        all_boxes.append(dims)

    coords_arr = np.array(all_coords)
    boxes_arr = np.array(all_boxes)

    if from_end:
        coords_arr = coords_arr[-n_frames:]
        boxes_arr = boxes_arr[-n_frames:]
    else:
        coords_arr = coords_arr[:n_frames]
        boxes_arr = boxes_arr[:n_frames]

    # Convert (N, 6) to (N, 3, 3) diagonal box matrices (assume orthorhombic boxes)
    box_vecs = boxes_arr[:, :3, None] * np.eye(3)
    return coords_arr, box_vecs


def run(args: argparse.Namespace) -> None:
    run_dir = Path(".").resolve()
    log.info(f"Run directory: {run_dir}")

    # Parse the run_*.py script to get SMILES and counts
    parsed = _parse_run_script(run_dir)
    if parsed is None:
        raise SystemExit(1)

    components = parsed["components"]
    smiles = parsed["smiles"]

    # Create the system from SMILES
    log.info(f"Building system for: {smiles}")
    smiles_list = [c["smiles"] for c in components]
    nmol_list = [c["nmol"] for c in components]

    tensor_system, _, _ = create_system_from_smiles(
        smiles_list, nmol_list, args.forcefield
    )

    # Load the last n_frames from the DCD trajectory
    traj_path = run_dir / "trajectory.dcd"
    if not traj_path.exists():
        log.error(f"trajectory.dcd not found in '{run_dir}'.")
        raise SystemExit(1)

    log.info(f"Loading last {args.n_frames} frame(s) from trajectory...")
    coords, box_vecs = _load_last_frames_dcd(
        traj_path, n_frames=args.n_frames, from_end=True
    )
    log.info(f"Loaded {len(coords)} frame(s), shape {coords.shape}")

    # Generate scale factors for the three regions
    scale_factors = np.concatenate([
        np.linspace(*args.close_range),
        np.linspace(*args.equilibrium_range)[1:],
        np.linspace(*args.long_range)[1:],
    ])
    n_scales = len(scale_factors)
    log.info(
        f"Scale factors: {n_scales} points ({scale_factors[0]:.3f} – {scale_factors[-1]:.3f})"
    )

    # Per-frame scaled configurations
    # Each of the n_frames seeds its own independent set of n_scales
    # total = n_frames x n_scales configs.
    log.info(
        f"Generating scaled configurations: {args.n_frames} frame(s) x {n_scales} scales = {args.n_frames * n_scales} total..."
    )

    all_coords: list[np.ndarray] = []
    all_box_vectors: list[np.ndarray] = []
    all_scale_factors: list[np.ndarray] = []

    for i in range(args.n_frames):
        # Single-frame arrays: shape (1, n_atoms, 3) and (1, 3, 3)
        frame_coords = coords[i : i + 1]
        frame_box = box_vecs[i : i + 1]

        result = create_scaled_configurations(
            tensor_system, frame_coords, frame_box, scale_factors
        )
        result_coords = result.coords if hasattr(result, "coords") else result[0]
        result_box_vecs = result.box_vectors if hasattr(result, "box_vectors") else result[1]
        result_sf = result.scale_factors if hasattr(result, "scale_factors") else result[2]
        all_coords.extend(result_coords)
        all_box_vectors.extend(result_box_vecs)
        all_scale_factors.append(np.asarray(result_sf))
        log.info(
            f"  Frame {i + 1}/{args.n_frames} -> {len(result_coords)} scaled config(s)"
        )

    # Concatenate all frames' results
    combined_coords = np.array(all_coords)
    combined_box_vectors = np.array(all_box_vectors)
    combined_scale_factors = np.concatenate(all_scale_factors, axis=0)

    log.info(f"Total scaled configurations: {len(combined_coords)}")

    # MLP simulation setup and energy/force computation
    log.info(f"Setting up MLP simulation ('{args.mlp_name}')...")
    mlp_simulation = setup_mlp_simulation(
        tensor_system,
        args.mlp_name,
        mlp_device=args.mlp_device,
        platform=args.mlp_platform,
    )

    log.info("Computing MLP energies and forces...")
    ef_result = compute_mlp_energies_forces(
        mlp_simulation,
        all_coords,
        all_box_vectors,
        show_progress=True,
    )
    energies = ef_result.energies if hasattr(ef_result, "energies") else ef_result[0]
    forces = ef_result.forces if hasattr(ef_result, "forces") else ef_result[1]

    # Arrays -> compressed npz
    npz_path = run_dir / "energies_forces.npz"
    np.savez_compressed(
        npz_path,
        coords=combined_coords,
        box_vectors=combined_box_vectors,
        scale_factors=combined_scale_factors,
        energies=energies,
        forces=forces,
    )
    log.info(f"Saved arrays -> '{npz_path}'")

    # Metadata (strings / component dicts) -> JSON
    meta_path = run_dir / "energies_forces_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(
            {"name": run_dir.name, "smiles": smiles, "components": components},
            fh,
            indent=2,
        )
    log.info(f"Saved metadata -> '{meta_path}'")
    log.info("Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute MLP energies and forces for the run in the current "
            "directory and save them to energies_forces.npz + energies_forces_meta.json."
        )
    )
    p.add_argument(
        "--forcefield",
        default="openff-2.0.0.offxml",
        metavar="FF",
        help="OpenFF force field name (default: openff-2.0.0.offxml).",
    )
    p.add_argument(
        "--mlp-name",
        default="mace-off24-medium",
        help="ML potential name (default: mace-off24-medium).",
    )
    p.add_argument(
        "--mlp-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for the ML potential (default: cuda).",
    )
    p.add_argument(
        "--mlp-platform",
        default="CPU",
        choices=["CPU", "CUDA", "OpenCL"],
        help="OpenMM platform (default: CPU).",
    )
    p.add_argument(
        "--n-frames",
        type=int,
        default=1,
        help="Number of last trajectory frames to use (default: 1).",
    )
    p.add_argument(
        "--close-range",
        nargs=3,
        type=float,
        metavar=("START", "END", "N"),
        default=[0.75, 0.9, 5],
        help="Close-range scale-factor range: start end n_points (default: 0.75 0.9 5).",
    )
    p.add_argument(
        "--equilibrium-range",
        nargs=3,
        type=float,
        metavar=("START", "END", "N"),
        default=[0.9, 1.1, 15],
        help="Equilibrium scale-factor range (default: 0.9 1.1 15).",
    )
    p.add_argument(
        "--long-range",
        nargs=3,
        type=float,
        metavar=("START", "END", "N"),
        default=[1.1, 2.0, 12],
        help="Long-range scale-factor range (default: 1.1 2.0 12).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
