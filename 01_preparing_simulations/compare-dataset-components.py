"""
This script compares two datasets of physical properties, identifying unique components in each dataset.
It reads two JSON files containing datasets, extracts unique components from each,
and saves the results to a JSON file.
It also generates high-resolution images of the added and removed components.
The images are saved in a specified directory.
"""
import pathlib
import json
import logging

import click

from rdkit import Chem
from rdkit.Chem import Draw
from openff.evaluator.datasets import PhysicalPropertyDataSet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def save_highres_image(smiles: list[str], filename: str):
    """
    Save a high-resolution SVG of the RDKit molecules from the given SMILES strings.
    """
    rdmols = [Chem.MolFromSmiles(smi) for smi in smiles if smi]
    img = Draw.MolsToGridImage(
        rdmols,
        molsPerRow=5,
        subImgSize=(300, 300),
        useSVG=True,
    )
    with open(filename, "w") as f:
        f.write(img)#.data)
    logger.info(f"Saved high-resolution image to {filename}")

@click.command()
@click.option(
    "--new-dataset-file",
    "-n",
    default="new-dataset.json",
    help=(
        "The JSON file containing the new dataset. "
        "This should be a valid PhysicalPropertyDataSet JSON file."
    ),
)
@click.option(
    "--old-dataset-file",
    "-o",
    default="sage-training-set.json",
    help=(
        "The JSON file containing the old dataset. "
        "This should be a valid PhysicalPropertyDataSet JSON file."
    ),
)
@click.option(
    "--output-file",
    "-f",
    default="functional-group-profiles/unique-components.json",
    help="The output file to save the unique components to",
)
@click.option(
    "--image-directory",
    "-i",
    default="images",
    help=(
        "Directory to save the high-resolution images of added/removed components. "
        "Added components are found in new_dataset but not in old_dataset, "
        "and removed components are found in old_dataset but not in new_dataset. "
        "Added components are saved as 'highres_added.svg', "
        "and removed components as 'highres_removed.svg'."
    ),
)
def main(
    new_dataset_file: str = "new-dataset.json",
    old_dataset_file: str = "sage-training-set.json",
    output_file: str = "functional-group-profiles/unique-components.json",
    image_directory: str = "images",
):
    new_dataset = PhysicalPropertyDataSet.from_json(new_dataset_file)
    old_dataset = PhysicalPropertyDataSet.from_json(old_dataset_file)

    new_unique_components = set()
    old_unique_components = set()

    for physprop in old_dataset.properties:
        for component in physprop.substance.components:
            old_unique_components.add(component.smiles)

    for physprop in new_dataset.properties:
        for component in physprop.substance.components:
            new_unique_components.add(component.smiles)

    logger.info(f"Old unique components: {len(old_unique_components)}")
    logger.info(f"New unique components: {len(new_unique_components)}")


    added_components = new_unique_components - old_unique_components
    removed_components = old_unique_components - new_unique_components

    logger.info(f"Added components: {len(added_components)}")
    logger.info(f"Removed components: {len(removed_components)}")


    data = {
        "old_unique_components": sorted(old_unique_components),
        "new_unique_components": sorted(new_unique_components),
        "added_components": sorted(added_components),
        "removed_components": sorted(removed_components),
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Saved unique components to {output_file}")

    # plot high-resolution RDKit image
    image_directory = pathlib.Path(image_directory)
    image_directory.mkdir(parents=True, exist_ok=True)
    save_highres_image(added_components, str(image_directory / "highres_added.svg"))
    save_highres_image(removed_components, str(image_directory / "highres_removed.svg"))


if __name__ == "__main__":
    main()
