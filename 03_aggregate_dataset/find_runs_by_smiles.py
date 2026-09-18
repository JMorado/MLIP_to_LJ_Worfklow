"""
Search through run_* directories and find those that match specific SMILES
strings in their energies_forces_meta.json files.
"""

import argparse
import logging
from pathlib import Path

import scalej

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def split_smiles_entries(smiles_values: list[str]) -> set[str]:
    """Return unique SMILES tokens, splitting mixture entries on '+'."""
    tokens = set()
    for smiles_value in smiles_values:
        tokens.update(part.strip() for part in smiles_value.split("+") if part.strip())
    return tokens


def find_runs(runs_dir: Path, target_smiles: list[str]) -> list[str]:
    """Find run directories that match any of the target SMILES."""
    matching_runs = []

    target_smiles_set = split_smiles_entries(target_smiles)

    log.info(f"Searching for runs matching any of: {sorted(target_smiles_set)}")

    # Use glob to find all run_* directories
    run_dirs = sorted(runs_dir.glob("run_*"))

    if not run_dirs:
        log.warning(f"No run_* directories found in {runs_dir}")
        return []

    for d in run_dirs:
        if not d.is_dir():
            continue

        meta_path = d / "energies_forces_meta.json"
        if not meta_path.exists():
            log.debug(f"[{d.name}] metadata not found - skipping.")
            continue

        try:
            meta = scalej.load_json(meta_path)
        except (ValueError, OSError) as e:
            log.error(f"[{d.name}] Failed to read metadata: {e}")
            continue

        run_smiles = meta.get("smiles")
        if not run_smiles:
            log.debug(f"[{d.name}] No 'smiles' field in metadata.")
            continue

        run_smiles_set = split_smiles_entries([run_smiles])
        components = {
            component["smiles"]
            for component in meta.get("components", [])
            if component.get("smiles")
        }
        candidate_smiles = run_smiles_set | components

        if target_smiles_set & candidate_smiles:
            matching_runs.append(d.name)
            log.info(f"[{d.name}] Matches: {run_smiles}")

    return matching_runs


def main():
    parser = argparse.ArgumentParser(
        description="Find runs matching specific SMILES strings."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Base directory containing run_* subdirectories.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--smiles",
        nargs="+",
        help="SMILES string(s) to search for.",
    )
    group.add_argument(
        "--smiles-file",
        type=Path,
        help="Path to a file containing SMILES strings (one per line).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="If provided, write the matching run names to this file.",
    )

    args = parser.parse_args()

    target_smiles = []
    if args.smiles:
        target_smiles = args.smiles
    elif args.smiles_file:
        if not args.smiles_file.exists():
            log.error(f"SMILES file not found: {args.smiles_file}")
            return
        with open(args.smiles_file, "r") as f:
            target_smiles = [line.strip() for line in f if line.strip()]

    if not target_smiles:
        log.error("No target SMILES provided.")
        return

    matching_runs = find_runs(args.runs_dir, target_smiles)

    log.info(f"Found {len(matching_runs)} matching run(s).")

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            for run_name in matching_runs:
                f.write(f"{run_name}\n")
        log.info(f"Saved matching runs to '{args.output_file}'")
    else:
        print("\n".join(matching_runs))


if __name__ == "__main__":
    main()
