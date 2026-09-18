"""
This applies filter to create a final training set for physical property data.

This builds off https://github.com/openforcefield/openff-sage/blob/main/data-set-curation/physical-property/optimizations/curate-training-set.py
"""
import collections
import logging
import time
import pathlib
import click
import typing

import pandas as pd
import numpy as np

from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet
from openff.evaluator.datasets.curation.components import filtering, selection, thermoml
from openff.evaluator.datasets.curation.components.selection import State, TargetState
from openff.evaluator.datasets.curation.workflow import (
    CurationWorkflow,
    CurationWorkflowSchema,
)

from openff.evaluator.utils.checkmol import ChemicalEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHEMICAL_ENVIRONMENTS = [
    # amines
    ChemicalEnvironment.PrimaryAliphAmine,
    ChemicalEnvironment.PrimaryAromAmine,
    ChemicalEnvironment.PrimaryAmine,
    ChemicalEnvironment.SecondaryAliphAmine,
    ChemicalEnvironment.SecondaryAromAmine,
    ChemicalEnvironment.SecondaryAmine,
    ChemicalEnvironment.TertiaryAliphAmine,
    ChemicalEnvironment.TertiaryAromAmine,
    ChemicalEnvironment.TertiaryAmine,
    # halogens
    ChemicalEnvironment.AlkylChloride,
    ChemicalEnvironment.ArylChloride,
    ChemicalEnvironment.AlkylBromide,
    ChemicalEnvironment.ArylBromide,
    # amides
    ChemicalEnvironment.CarboxylicAcidPrimaryAmide,
    ChemicalEnvironment.CarboxylicAcidSecondaryAmide,
    ChemicalEnvironment.CarboxylicAcidTertiaryAmide,
    ChemicalEnvironment.CarboxylicAcidAmide,

    # not found but keep it in anyway?
    ChemicalEnvironment.Cyanate,
    ChemicalEnvironment.Isocyanate,
    ChemicalEnvironment.Thioacetal,

    # common
    ChemicalEnvironment.Alkane,
    ChemicalEnvironment.Alkene,
    ChemicalEnvironment.Alcohol,
    ChemicalEnvironment.Ketone,
    ChemicalEnvironment.Ether,
    ChemicalEnvironment.CarbonylHydrate,
    ChemicalEnvironment.CarboxylicAcidEster,
    ChemicalEnvironment.Aromatic,
    ChemicalEnvironment.CarboxylicAcid,  # acetic acid
    ChemicalEnvironment.HalogenDeriv,  # chloroform
    ChemicalEnvironment.Nitrile,
    ChemicalEnvironment.Acetal,  # C1COCO1
    ChemicalEnvironment.Hemiacetal,
    ChemicalEnvironment.Hemiaminal,
    ChemicalEnvironment.Aminal,
    ChemicalEnvironment.Aldehyde,
    ChemicalEnvironment.Heterocycle,
    ChemicalEnvironment.Aqueous,  # water
]

TARGET_STATES = [
    TargetState(
        property_types=[
            ("Density", 1),
        ],
        states=[
            State(
                temperature=298.15,
                pressure=101.325,
                mole_fractions=(1.0,),
            ),
        ],
    ),
    TargetState(
        property_types=[
            ("Density", 2),
            ("EnthalpyOfMixing", 2),
        ],
        states=[
            State(
                temperature=298.15,
                pressure=101.325,
                mole_fractions=(0.25, 0.75),
            ),
            State(
                temperature=298.15,
                pressure=101.325,
                mole_fractions=(0.5, 0.5),
            ),
            State(
                temperature=298.15,
                pressure=101.325,
                mole_fractions=(0.75, 0.25),
            ),
        ],
    ),
]


def curate_data_set(
    input_data_frame,
    smiles_to_exclude,
    property_type_filter: filtering.FilterByPropertyTypesSchema,
    n_processes,
) -> pd.DataFrame:
    """Curate the input data frame to select a training set based on the defined target states and chemical environments."""
    allowed_elements = [
        "C",
        "O",
        "N",
        "Cl",
        "Br",
        "H",  # "F", "S"
    ]
    # filter out properties that include components
    # with less than 4 representations across entire dataset
    select_num_component = selection.SelectNumRepresentationSchema(
        minimum_representation=4, per_component=True
    )
    # filter out substances (i.e. full mixtures)
    # with less than 3 representations across entire dataset
    select_num_substance = selection.SelectNumRepresentationSchema(
        minimum_representation=3, per_component=False
    )

    component_schemas=[
        # Remove any molecules containing elements that aren't currently of interest
        filtering.FilterByElementsSchema(allowed_elements=allowed_elements),
        # property_type_filter,
    ]
    if smiles_to_exclude:
        logger.info(f"Excluding {len(smiles_to_exclude)} SMILES from the dataset")
        component_schemas.append(
            filtering.FilterBySmilesSchema(
                smiles_to_exclude=smiles_to_exclude,
            )
        )
    component_schemas.extend([
        # Retain only measurements made for substances which contain environments
        # of interest.
        filtering.FilterByEnvironmentsSchema(environments=CHEMICAL_ENVIRONMENTS),
        # select data points at particular concentrations
        selection.SelectDataPointsSchema(target_states=TARGET_STATES),
        select_num_component,
        select_num_substance,
        # select diverse mixtures
        selection.SelectSubstancesSchema(
            target_environments=CHEMICAL_ENVIRONMENTS,
            n_per_environment=5,
            per_property=False,
        ),
        # select_num,
        # Filter out the density of water.
        filtering.FilterBySubstancesSchema(substances_to_exclude=[("O",)]),
    ])

    curation_schema = CurationWorkflowSchema(
        component_schemas=component_schemas,
    )

    return CurationWorkflow.apply(input_data_frame, curation_schema, n_processes)


def save_dataset(dataset, output_file: pathlib.Path):
    """
    Save the dataset to a CSV and JSON file.
    The CSV file will be a valid PhysicalPropertyDataSet CSV file.
    The JSON file will be a valid PhysicalPropertyDataSet JSON file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_pandas().to_csv(output_file)
    dataset.json(output_file.with_suffix(".json"), format=True)

    logger.info(f"Saved to {output_file}")
    logger.info(f"Saved to {output_file.with_suffix('.json')}")


@click.command()
@click.option(
    "--output-file",
    "-o",
    default="output/training-set.csv",
    help=(
        "The output CSV file to save the filtered data to. "
        "Note, a JSON file with the same name but with a .json extension will also be created. "
        "Both encode PhysicalPropertyDataSet objects."
    ),
)
@click.option(
    "--input-file",
    "-i",
    default="../intermediate/output/renamed-filtered.csv",
    help="The CSV file containing existing parsed ThermoML data",
)
@click.option(
    "--exclude-file",
    "-x",
    default=None,
    help="The file containing SMILES to exclude",
)
@click.option(
    "--n-processes",
    "-np",
    default=1,
    help="The number of processes to use for filtering the data",
)
def main(
    input_file: str = "../intermediate/output/renamed-filtered.csv",
    exclude_file: str = None,
    output_file: str = "output/training-set.csv",
    n_processes: int = 1,
):
    now = time.time()
    logger.info(f"Starting at {time.ctime(now)}")

    df = pd.read_csv(input_file)
    df["Id"] = df["Id"].astype(str)
    df["N Components"] = df["N Components"].astype(int)
    ds = PhysicalPropertyDataSet.from_pandas(df)

    thermoml_data_frame = ds.to_pandas()
    logger.info(f"Loading {len(thermoml_data_frame)} data")

    # load smiles to exclude
    if exclude_file:
        with open(exclude_file, "r") as f:
            contents = f.readlines()
        smiles_to_exclude = [x.strip().split()[0] for x in contents]
    else:
        smiles_to_exclude = []

    property_type_filter = filtering.FilterByPropertyTypesSchema(
        property_types=[
            "Density",
            "EnthalpyOfMixing",
        ],
        n_components={
            "Density": [1, 2],
            "EnthalpyOfMixing": [2],
        },
        strict=True,
    )

    training_set_frame = curate_data_set(
        thermoml_data_frame,
        smiles_to_exclude,
        property_type_filter,
        n_processes,
    )
    logger.info(f"Filtered to {len(training_set_frame)} data points")

    assert len(training_set_frame) > 0, "No data points left after filtering"
    # make sure we wind up with a reasonable number of data points
    assert len(training_set_frame) > 1000, "Not enough data points left after filtering"
    assert len(training_set_frame) < 2000, "Too many data points left after filtering"

    ds = PhysicalPropertyDataSet.from_pandas(training_set_frame)

    # count and log properties
    counter = collections.Counter()
    for prop in ds.properties:
        counter[type(prop).__name__] += 1
    count_str = ", ".join(
        f"{clsname}: {count}" for clsname, count in sorted(counter.items())
    )
    logger.info(f"Property count: {count_str}")

    save_dataset(ds, pathlib.Path(output_file))

    logger.info(f"Finished at {time.ctime(time.time())}")
    logger.info(f"Elapsed time: {time.time() - now} seconds")


if __name__ == "__main__":
    main()
