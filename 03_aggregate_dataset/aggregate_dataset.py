"""Aggregate ``energies_forces.arrow`` / ``energies_forces_meta.json`` files from
multiple ``run_*`` directories into a single combined training dataset.
"""

import argparse
import logging
from pathlib import Path

import descent.targets.energy
import pyarrow.ipc as ipc
from scalej.data import create_from_scalej, load_json, save_dataset, save_json

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _discover_runs(
    runs_dir: Path,
    run_names: list[str] | None = None,
    max_run: str | None = None,
) -> list[Path]:
    """
    Return a sorted list of run directories that have energies_forces.arrow + .json.

    Parameters
    ----------
    max_run:
        If given, only include runs whose trailing numeric suffix is <= *max_run*
        For example, ``'0220'`` keeps ``run_0000`` through ``run_0220``.
    """
    if run_names:
        candidates = [runs_dir / n for n in run_names]
    else:
        candidates = sorted(runs_dir.glob("run_*"))

    valid: list[Path] = []
    seen: set[Path] = set()
    for d in candidates:
        resolved = d.resolve()
        if resolved in seen:
            log.warning(
                f"'{d.name}' resolved to a path already seen - skipping duplicate."
            )
            continue
        seen.add(resolved)
        if not d.is_dir():
            log.warning(f"'{d}' is not a directory - skipping.")
            continue
        # Apply --max-run filter
        if max_run is not None:
            suffix = d.name.split("_", 1)[-1]
            if suffix.isdigit() and int(suffix) > int(max_run):
                log.debug(
                    f"[{d.name}] {int(suffix)} > max_run {int(max_run)} - skipping."
                )
                continue
        arrow = d / "energies_forces.arrow"
        meta_primary = d / "energies_forces_arrow_meta.json"
        meta_fallback = d / "energies_forces_meta.json"
        meta = meta_primary if meta_primary.exists() else meta_fallback
        if not arrow.exists() or not meta.exists():
            log.warning(
                f"[{d.name}] energies_forces.arrow / .json not found - skipping (run calculate_mlp_data.py first)."
            )
            continue
        valid.append(d)

    log.info(
        f"Found {len(valid)} / {len(candidates)} run(s) with energies_forces data."
    )
    return valid


def aggregate(
    runs_dir: Path,
    output_dir: Path,
    run_names: list[str] | None,
    max_run: str | None = None,
    stride: int = 1,
) -> None:
    """Aggregate energies/forces data from multiple runs into a combined dataset."""
    run_dirs = _discover_runs(runs_dir, run_names, max_run=max_run)
    if stride < 1:
        raise ValueError(f"--stride must be >= 1, got {stride}.")
    if not run_dirs:
        log.error("No valid run directories found. Exiting.")
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[descent.targets.energy.Entry] = []
    seen_names: set[str] = set()
    frame_counts: dict[str, tuple[int, int]] = {}  # name -> (n_total, n_kept)

    for d in run_dirs:
        meta_primary = d / "energies_forces_arrow_meta.json"
        meta_fallback = d / "energies_forces_meta.json"
        meta_path = meta_primary if meta_primary.exists() else meta_fallback
        meta = load_json(meta_path)
        name = meta["name"]

        if name in seen_names:
            log.warning(
                f"[{d.name}] metadata name '{name}' already seen - skipping duplicate."
            )
            continue
        seen_names.add(name)

        log.info(f"[{name}] Loading pre-computed MLP data ...")
        with ipc.open_file(str(d / "energies_forces.arrow")) as _r:
            n_total = len(_r.read_all().column("energy")[0].as_py())
        entry = create_from_scalej(d / "energies_forces.arrow", stride=stride)
        if not entry.get("id"):
            entry["id"] = name
        n_kept = int(entry["energy"].shape[0])

        frame_counts[name] = (n_total, n_kept)
        entries.append(entry)

        log.info(
            f"[{name}] {n_kept} / {n_total} configuration(s) kept"
            + (f" (stride={stride})" if stride > 1 else ".")
        )

    log.info(f"Building dataset from {len(entries)} entry(ies) ...")
    dataset = descent.targets.energy.create_dataset(entries)

    data_path = output_dir / "combined_data"
    meta_path = output_dir / "combined_dataset_meta.json"

    save_dataset(dataset, data_path)

    # Build and save aggregation metadata so the provenance of the combined
    # dataset is always traceable.
    run_meta: list[dict] = []
    for d in run_dirs:
        m = load_json(d / "energies_forces_meta.json")
        n_total, n_kept = frame_counts.get(m["name"], (0, 0))
        run_meta.append(
            {
                "run": d.name,
                "name": m["name"],
                "smiles": m["smiles"],
                "components": m["components"],
                "scale_factors": m["scale_factors"],
                "n_frames_total": n_total,
                "n_frames_kept": n_kept,
            }
        )

    n_total_frames = sum(n for _, n in frame_counts.values())
    agg_meta = {
        "stride": stride,
        "max_run": max_run,
        "n_runs": len(run_dirs),
        "n_configurations_total": n_total_frames,
        "runs": run_meta,
    }
    save_json(agg_meta, meta_path)

    log.info(f"Saved combined data        -> '{data_path}'")
    log.info(f"Saved aggregation metadata -> '{meta_path}'")
    log.info(f"Done. {n_total_frames} total frames across {len(run_dirs)} run(s).")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate energies_forces.arrow files from run_* directories into "
            "a combined training dataset."
        )
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Base directory containing run_* subdirectories.",
    )
    p.add_argument(
        "--runs",
        nargs="*",
        default=None,
        metavar="RUN",
        help=(
            "Specific run folder names to aggregate (e.g. run_0001 run_0003). "
            "If omitted, all run_* folders under --runs-dir are used."
        ),
    )
    p.add_argument(
        "--runs-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="A text file containing run names to aggregate, one per line.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/dataset"),
        help="Directory where the combined dataset is saved (default: output/dataset).",
    )
    p.add_argument(
        "--max-run",
        default=None,
        metavar="SUFFIX",
        help=(
            "Only process runs whose numeric suffix is <= SUFFIX "
            "(e.g. '0220' keeps run_0001 .. run_0220)."
        ),
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="Keep every Nth frame from each run (default: 1 = all frames).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    run_names = args.runs or []
    if args.runs_file:
        with open(args.runs_file, "r") as f:
            file_runs = [line.strip() for line in f if line.strip()]
        run_names.extend(file_runs)

    # If both were None/empty, pass None to _discover_runs to trigger globbing.
    run_names = run_names if run_names else None

    aggregate(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        run_names=run_names,
        max_run=args.max_run,
        stride=args.stride,
    )
