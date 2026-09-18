"""
This applies filters to generate an extended validation set for physical property data.

This differs from `filter-data-validation.py` in that it is intended to be used
on the intermediate filtered data, and to produce a larger validation set.

This builds off https://github.com/openforcefield/openff-sage/blob/main/data-set-curation/physical-property/optimizations/curate-training-set.py
"""

import json
import time
import collections
import pathlib
import click
import logging

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
    # not found but keep it in anyway?
    ChemicalEnvironment.Cyanate,
    ChemicalEnvironment.Isocyanate,
    # these are distinct enough to try to grab both
    ChemicalEnvironment.PrimaryAliphAmine,
    ChemicalEnvironment.PrimaryAromAmine,
    # amines
    ChemicalEnvironment.SecondaryAmine,
    ChemicalEnvironment.TertiaryAmine,
    # halogens
    ChemicalEnvironment.AlkylChloride,
    ChemicalEnvironment.ArylChloride,
    ChemicalEnvironment.AlkylBromide,
    ChemicalEnvironment.ArylBromide,
    ChemicalEnvironment.Alkane,
    ChemicalEnvironment.Alkene,
    ChemicalEnvironment.Alcohol,
    ChemicalEnvironment.Ketone,
    ChemicalEnvironment.CarboxylicAcidEster,
    ChemicalEnvironment.Ether,
    ChemicalEnvironment.Aromatic,
    # amides
    ChemicalEnvironment.CarboxylicAcidPrimaryAmide,
    ChemicalEnvironment.CarboxylicAcidSecondaryAmide,
    ChemicalEnvironment.CarboxylicAcidTertiaryAmide,
    ChemicalEnvironment.Heterocycle,
    # "rare" groups -- OCCO, CC(C)O
    ChemicalEnvironment.CarboxylicAcid,  # acetic acid
    ChemicalEnvironment.HalogenDeriv,  # chloroform
    ChemicalEnvironment.Aqueous,  # water
    ChemicalEnvironment.Nitrile,
    ChemicalEnvironment.Acetal,  # C1COCO1
    ChemicalEnvironment.Aldehyde,
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
    n_processes,
) -> pd.DataFrame:
    """Curate the input data frame to select a training set based on the defined target states and chemical environments"""
    allowed_elements = [
        "C", "O", "N", "Cl", "Br", "H",
        "F", "S", "P", "I"
    ]

    component_schemas=[
        # Filter out any measurements made for systems with more than
        # two components
        filtering.FilterByNComponentsSchema(n_components=[1, 2]),
        # Filter out data points measured away from ambient conditions.
        filtering.FilterByTemperatureSchema(
            minimum_temperature=288.15, maximum_temperature=318.15
        ),
        filtering.FilterByPressureSchema(
            minimum_pressure=99.9, maximum_pressure=101.4
        ),
        # Retain only density and enthalpy of mixing data points which
        # have been measured for the same systems.
        # property_type_filter,
        # Filter out long chain molecules (slower to simulate / converge) and 1, 3
        # carbonyl compounds where one of the carbonyls is a ketone (cases where
        # the enol form may be present in non-negligible amounts).
        filtering.FilterBySmirksSchema(
            smirks_to_exclude=[
                # Long chain alkane /ether
                "-".join(["[#6X4,#8X2]"] * 10),
                # 1, 3 carbonyls with at least one ketone carbonyl.
                "[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](=[#8])-[#6]",
            ],
        ),
        # Filter out problematic molecules
        filtering.FilterBySmilesSchema(
            smiles_to_exclude=[
                # Heavy water.
                "[2H]O[2H]",
                # Molecules which OpenMM misinterprets
                "N[C@@H](CS)C(=O)O",
                "CSCC[C@H](N)C(=O)O",
                # Molecules which cause NaNs during simulations
                "O=S(=O)(O)CCCN1CCOCC1",
            ]
        ),
        # Filter out systems where one component is in a significant excess.
        filtering.FilterByMoleFractionSchema(
            mole_fraction_ranges={2: [[(0.05, 0.95)]]}
        ),
        # Filter out any racemic mixtures
        filtering.FilterByRacemicSchema(),
        # Remove any substances measured for systems with undefined
        # stereochemistry
        filtering.FilterByStereochemistrySchema(),
        # Remove any measurements made for systems where any of the components
        # are charged.
        filtering.FilterByChargedSchema(),
        # Remove measurements made for ionic liquids
        filtering.FilterByIonicLiquidSchema(),
        # Remove any molecules containing elements that aren't currently of interest
        filtering.FilterByElementsSchema(allowed_elements=allowed_elements),
        # Remove any molecules containing elements that aren't currently of interest
        selection.SelectDataPointsSchema(target_states=TARGET_STATES),
    ]
    if smiles_to_exclude:
        component_schemas.append(
            filtering.FilterBySmilesSchema(
                smiles_to_exclude=smiles_to_exclude,
            )
        )
    component_schemas.extend(
        [
        selection.SelectSubstancesSchema(
            target_environments=CHEMICAL_ENVIRONMENTS,
            n_per_environment=5,
            per_property=False,
        ),
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
    default="output/validation-set.csv",
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
    help="The JSON file containing existing parsed ThermoML data",
)
@click.option(
    "--exclude-file",
    "-x",
    default="",
    help="The file containing SMILES to exclude",
)
@click.option(
    "--n-processes",
    "-np",
    default=1,
    help="The number of processes to use for filtering the data",
)
@click.option(
    "--training-file",
    "-t",
    "training_files",
    multiple=True,
    default=["output/training-set.json"],
    help=(
        "The JSON file/s containing the training set. "
        "This is used to filter out properties that are already in the training set."
    ),
)
def main(
    input_file: str = "../intermediate/output/renamed-filtered.csv",
    training_files: list[str] = ["output/training-set.json"],
    exclude_file: str = None,
    output_file: str = "output/validation-set.csv",
    n_processes: int = 1,
    # renumber: bool = True,
):
    now = time.time()
    logger.info(f"Starting at {time.ctime(now)}")

    if input_file.endswith("json"):
        ds = PhysicalPropertyDataSet.from_json(pathlib.Path(input_file))
        df = ds.to_pandas()
    else:
        df = pd.read_csv(input_file)
        df["Id"] = df["Id"].astype(str)
        # ds = PhysicalPropertyDataSet.from_pandas(df)

    logger.info(f"Loaded {len(df)} properties from {input_file}")

    training_set = PhysicalPropertyDataSet()
    for training_file in training_files:
        tset = PhysicalPropertyDataSet.from_json(pathlib.Path(training_file))
        training_set.add_properties(*tset.properties)
        logger.info(f"Loaded {len(tset)} properties from {training_file}")
    logger.info(f"Loaded a total of {len(training_set)} training properties")

    # filter out training properties
    training_ids = [x.id for x in training_set.properties]
    thermoml_data_frame = df[~df["Id"].isin(training_ids)]
    # ds2 = PhysicalPropertyDataSet()
    # for prop in ds.properties:
    #     if prop.id not in training_ids:
    #         ds2.add_properties(prop)

    # thermoml_data_frame = ds2.to_pandas()
    logger.info(f"Filtered to {len(thermoml_data_frame)} data")

    # load smiles to exclude
    if exclude_file:
        with open(exclude_file, "r") as f:
            contents = f.readlines()
        smiles_to_exclude = [x.strip().split()[0] for x in contents]
    else:
        smiles_to_exclude = []

    training_set_frame = curate_data_set(
        thermoml_data_frame,
        smiles_to_exclude,
        n_processes,
    )
    logger.info(f"Filtered to {len(training_set_frame)} data points")

    assert len(training_set_frame) > 0, "No data points left after filtering"
    # make sure we wind up with a reasonable number of data points
    assert len(training_set_frame) > 500, "Not enough data points left after filtering"
    # assert len(training_set_frame) < 2000, "Too many data points left after filtering"

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
