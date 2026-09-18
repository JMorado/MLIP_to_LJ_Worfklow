"""Convert legacy ``energies_forces.npz`` files to Arrow IPC format for easier downstream processing."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from descent.targets.energy import DATA_SCHEMA
from scalej.data import load_json, save_json

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def convert_npz_to_parquet(run_dir: Path, delete_npz: bool = False, force: bool = False) -> bool:
    """
    Convert a single ``energies_forces.npz`` to ``energies_forces.arrow``.

    Parameters
    ----------
    run_dir : Path
        Directory containing the ``energies_forces.npz`` file.
    delete_npz : bool
        If True, remove the ``.npz`` file after successful conversion.

    Returns
    -------
    bool
        True if conversion succeeded, False if skipped.
    """
    npz_path = run_dir / "energies_forces.npz"
    arrow_path = run_dir / "energies_forces.arrow"

    if not npz_path.exists():
        LOGGER.warning(f"[{run_dir.name}] No energies_forces.npz found — skipping.")
        return False

    if arrow_path.exists() and not force:
        LOGGER.info(
            f"[{run_dir.name}] energies_forces.arrow already exists — skipping."
        )
        return False

    # Load data.
    data = np.load(npz_path)
    old_meta = {}
    meta_path = run_dir / "energies_forces_meta.json"
    if meta_path.exists():
        old_meta = load_json(meta_path)

    coords_flat = np.array(data["coords"], dtype=np.float64).flatten()
    forces_flat = np.array(data["forces"], dtype=np.float64).flatten()
    box_flat = np.array(data["box_vectors"], dtype=np.float64).flatten()
    energies = np.array(data["energies"], dtype=np.float64).ravel()
    n_configs = len(energies)
    scale_factors = data["scale_factors"].tolist() if "scale_factors" in data else []
    data.close()

    batch = pa.record_batch(
        [
            pa.array([old_meta.get("name") or run_dir.name], type=pa.string()),
            pa.array([old_meta.get("smiles", "")]),
            pa.array([coords_flat.tolist()], type=pa.list_(pa.float64())),
            pa.array([box_flat.tolist()], type=pa.list_(pa.float64())),
            pa.array([energies.tolist()], type=pa.list_(pa.float64())),
            pa.array([forces_flat.tolist()], type=pa.list_(pa.float64())),
        ],
        schema=DATA_SCHEMA,
    )
    # Serialize to an in-memory buffer first, then flush to disk in one go.
    # Writing directly to the filesystem can trigger EFAULT on some HPC
    # filesystems when pyarrow attempts a single large write syscall.
    arrow_path.parent.mkdir(parents=True, exist_ok=True)
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, DATA_SCHEMA) as writer:
        writer.write_batch(batch)
    arrow_path.write_bytes(sink.getvalue().to_pybytes())
    LOGGER.info(f"[{run_dir.name}] Converted {n_configs} frames -> {arrow_path.name}")

    # Update metadata.
    meta_data = {
        "name": old_meta.get("name") or run_dir.name,
        "smiles": old_meta.get("smiles", ""),
        "components": old_meta.get("components", []),
        "n_total_configs": n_configs,
        "scale_factors": scale_factors,
    }
    save_json(meta_data, meta_path)
    arrow_meta_path = run_dir / "energies_forces_arrow_meta.json"
    save_json(meta_data, arrow_meta_path)

    if delete_npz:
        npz_path.unlink()
        LOGGER.info(f"[{run_dir.name}] Deleted {npz_path.name}")

    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert legacy energies_forces.npz files to Arrow IPC format."
    )
    p.add_argument(
        "path",
        type=Path,
        help="Single run_* directory or (with --glob) parent directory.",
    )
    p.add_argument(
        "--glob", action="store_true", help="Convert all run_* subdirectories."
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing energies_forces.arrow files.",
    )
    p.add_argument(
        "--delete-npz",
        action="store_true",
        help="Delete the original .npz file after conversion.",
    )
    args = p.parse_args()

    if args.glob:
        dirs = sorted(args.path.glob("run_*"))
        if not dirs:
            LOGGER.error(f"No run_* directories found under '{args.path}'.")
            raise SystemExit(1)
        converted = sum(
            convert_npz_to_parquet(d, delete_npz=args.delete_npz, force=args.force) for d in dirs
        )
        LOGGER.info(f"Converted {converted} / {len(dirs)} run(s).")
    else:
        convert_npz_to_parquet(args.path, delete_npz=args.delete_npz, force=args.force)


if __name__ == "__main__":
    main()
