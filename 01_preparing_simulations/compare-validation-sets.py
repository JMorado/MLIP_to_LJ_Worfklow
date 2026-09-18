import pathlib
import sys

import click
import tqdm

from loguru import logger
from openff.evaluator.datasets import PhysicalPropertyDataSet

logger.remove()
logger.add(sys.stdout)

SHORT_NAMES = {
    "Density": "dens",
    "EnthalpyOfMixing": "dhmix",
}

def main(
    reference_paths: list[str] = ["output/rc1-training-set.json", "output/rc1-validation-set.json"],
    target_paths: list[str] = ["output/training-set.json", "output/validation-set.json"],
    reference_output_path: str = "output/rc1-combined-set.json",
    target_output_path: str = "output/rc2-combined-set.json",
    reference_minus_target_output_path: str = "output/rc1-minus-rc2-set.json",
    target_minus_reference_output_path: str = "output/rc2-minus-rc1-set.json",
):
    reference_dataset = PhysicalPropertyDataSet()
    for path in tqdm.tqdm(reference_paths, desc="Loading reference datasets"):
        dataset = PhysicalPropertyDataSet.from_json(path)
        reference_dataset.add_properties(*dataset.properties)
    logger.info(f"Loaded {len(reference_dataset.properties)} reference properties")

    target_dataset = PhysicalPropertyDataSet()
    for path in tqdm.tqdm(target_paths, desc="Loading target datasets"):
        dataset = PhysicalPropertyDataSet.from_json(path)
        target_dataset.add_properties(*dataset.properties)
    logger.info(f"Loaded {len(target_dataset.properties)} target properties")

    # update ids
    for physprop in tqdm.tqdm(
        reference_dataset.properties + target_dataset.properties,
        desc="Updating property IDs",
    ):
        hash_value = physprop.get_property_hash()
        physprop.metadata = {"previous_id": physprop.id}
        clsname = SHORT_NAMES[type(physprop).__name__]
        prop_id = f"{clsname}_{hash_value}"
        physprop.id = prop_id

    # save renamed datasets
    reference_output_path = pathlib.Path(reference_output_path)
    reference_output_path.parent.mkdir(exist_ok=True, parents=True)
    reference_dataset.json(reference_output_path, format=True)
    logger.info(
        f"Saved reference dataset with {len(reference_dataset.properties)} "
        f"properties to {reference_output_path}"
    )
    target_output_path = pathlib.Path(target_output_path)
    target_output_path.parent.mkdir(exist_ok=True, parents=True)
    target_dataset.json(target_output_path, format=True)
    logger.info(
        f"Saved target dataset with {len(target_dataset.properties)} "
        f"properties to {target_output_path}"
    )


    reference_minus_target = PhysicalPropertyDataSet()
    target_ids = {prop.id for prop in target_dataset.properties}
    for prop in reference_dataset.properties:
        if prop.id not in target_ids:
            reference_minus_target.add_properties(prop)
    logger.info(
        f"Reference dataset minus target dataset has "
        f"{len(reference_minus_target.properties)} properties"
    )
    reference_minus_target_output_path = pathlib.Path(reference_minus_target_output_path)
    reference_minus_target_output_path.parent.mkdir(exist_ok=True, parents=True)
    reference_minus_target.json(reference_minus_target_output_path, format=True)

    target_minus_reference_output_path = pathlib.Path(target_minus_reference_output_path)
    target_minus_reference_output_path.parent.mkdir(exist_ok=True, parents=True)
    target_minus_reference = PhysicalPropertyDataSet()
    reference_ids = {prop.id for prop in reference_dataset.properties}
    for prop in target_dataset.properties:
        if prop.id not in reference_ids:
            target_minus_reference.add_properties(prop)
    logger.info(
        f"Target dataset minus reference dataset has "
        f"{len(target_minus_reference.properties)} properties"
    )
    target_minus_reference.json(target_minus_reference_output_path, format=True)


if __name__ == "__main__":
    main()