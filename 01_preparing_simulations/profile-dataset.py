"""
This module contains functions for profiling a dataset of physical properties.
It reads a PhysicalPropertyDataSet from a CSV file,
uses checkmol to analyze the functional groups present in the dataset,
and generates a summary of the functional groups.
The functional groups are categorized by their type (e.g., Density, EnthalpyOfMixing),
and the dataset name (which is provided as an argument).

The files saved are:
- `functional-groups.json`: Contains the counts of functional groups by dataset.
The file is structured as a JSON object of shape:
{
    "counts": {
        "dataset_name": { # input argument
            "Density": {
                "functional_group_name": int, # count of Density properties with this group
                ...
            },
            "EnthalpyOfMixing": {
                "functional_group_name": int, # count of EnthalpyOfMixing properties with this group
                ...
            },
        },
        ...,
    },
    "groups": {
        "dataset_name": { # input argument
            "Density": {
                "functional_group_name": [list of property IDs], # list of property IDs with this group
                ...
            },
            "EnthalpyOfMixing": {
                "functional_group_name": [list of property IDs], # list of property IDs with this group
                ...
            },
        },
        ...,
    }
}

- `property-counts.json`: Contains counts of different property types across datasets.
The file is structured as a JSON object of shape:
{
    "counts": {
        "dataset_name": { # input argument
            "Density": int, # count of Density properties
            "EnthalpyOfMixing": int, # count of EnthalpyOfMixing properties
        },
        ...
    }
}
- `density-functional-groups.csv`: A CSV file summarizing functional group counts for Density properties.
- `dhmix-functional-groups.csv`: A CSV file summarizing functional group counts for EnthalpyOfMixing properties.
"""

from collections import defaultdict, Counter
import itertools
import json
import typing
import logging
import pathlib

import click
import pandas as pd

from openff.evaluator.datasets import PhysicalPropertyDataSet
from yammbs.checkmol import analyze_functional_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def label_dataset(dataset: PhysicalPropertyDataSet) -> dict[str, dict[str, list[str]]]:
    """
    Label the dataset with functional groups based on the SMILES of the components.

    Parameters
    ----------
    dataset : PhysicalPropertyDataSet
        The dataset to label.
    
    Returns
    -------
    dict[str, dict[str, list[str]]]
        A dictionary where the keys are property types (e.g., "Density", "EnthalpyOfMixing"),
        and the values are dictionaries mapping functional group names to lists of property IDs.
    """
    groups_by_type = defaultdict(lambda: defaultdict(list))
    for prop in dataset.properties:
        data_type = type(prop).__name__
        components = [comp.smiles for comp in prop.substance.components]
        all_groups = []
        for component in components:
            groups = [x.value for x in analyze_functional_groups(component)]
            all_groups.append(groups)
        group_combinations = set()
        for comb in itertools.product(*all_groups):
            comb_str = " | ".join(sorted(comb))
            group_combinations.add(comb_str)
        for comb in group_combinations:
            groups_by_type[data_type][comb].append(prop.id)

    return groups_by_type


def combine_datasets_to_dataframe(
    data_groups: dict[str, dict],
    data_type: typing.Literal["Density", "EnthalpyOfMixing"],
) -> pd.DataFrame:
    """
    Combine the counts of functional groups across multiple datasets into a single DataFrame.
    Parameters
    ----------
    data_groups : dict[str, dict]
        A dictionary where keys are dataset names and values are dictionaries of functional group counts.
        The structure is:
        {
            "dataset_name": {
                "Density": {
                    "functional_group_name": [list of property IDs],
                    ...
                },
                "EnthalpyOfMixing": {
                    "functional_group_name": [list of property IDs],
                    ...
                },
            },
            ...
        }
    data_type : typing.Literal["Density", "EnthalpyOfMixing"]
        The type of property to filter by (e.g., "Density" or "EnthalpyOfMixing").

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the counts of functional groups for the specified property type.
        The DataFrame will have columns "Group", "Count", and "Dataset".
    """
    all_dfs = []
    for name, groups in data_groups.items():
        df_ = pd.DataFrame.from_dict(groups[data_type], orient="index")
        df_ = df_.reset_index().rename(columns={0: "Count", "index": "Group"})
        df_["Dataset"] = name
        all_dfs.append(df_)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df



@click.command()
@click.option(
    "--name-and-dataset-files",
    "-nd",
    multiple=True,
    type=(str, str),
    help="Pairs of dataset names and their corresponding JSON files",
)
@click.option(
    "--output-directory",
    "-o",
    default="functional-groups",
    help="Directory to save the functional groups analysis",
)
def main(
    name_and_dataset_files: list[tuple[str, str]] = [],
    output_directory: str = "functional-groups",
):
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    counts_by_name = {}
    groups_by_name = {}
    property_type_counts = defaultdict(Counter)
    for name, dataset_file in name_and_dataset_files:
        dataset = PhysicalPropertyDataSet.from_json(dataset_file)
        groups_by_type = label_dataset(dataset)
        for prop in dataset.properties:
            data_type = type(prop).__name__
            property_type_counts[name][data_type] += 1
        counts_by_type: dict[str, dict[str, int]] = {}
        for data_type, groups in groups_by_type.items():
            counts_by_type[data_type] = {
                group: len(ids)
                for group, ids in groups.items()
            }
        counts_by_name[name] = counts_by_type
        groups_by_name[name] = groups_by_type

        # combine unique mixtures for output
        unique_mixtures = set(counts_by_type["Density"].keys()).union(
            counts_by_type["EnthalpyOfMixing"].keys()
        )
        logger.info(f"Unique mixtures for {name}: {len(unique_mixtures)}")

    logger.info(f"Counts by name: {counts_by_name}")

    output_file = output_directory / "functional-groups.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "counts": counts_by_name,
                "groups": groups_by_name,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved functional groups to {output_file}")

    count_file = output_directory / "property-counts.json"
    with open(count_file, "w") as f:
        json.dump(
            {
                "counts": property_type_counts,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved property type counts to {count_file}")


    density_df = combine_datasets_to_dataframe(
        counts_by_name,
        "Density",
    )
    density_csv = output_directory / "density-functional-groups.csv"
    density_df.to_csv(density_csv, index=False)
    logger.info(f"Saved density functional groups to {density_csv}")

    dhmix_df = combine_datasets_to_dataframe(
        counts_by_name,
        "EnthalpyOfMixing",
    )
    dhmix_csv = output_directory / "dhmix-functional-groups.csv"
    dhmix_df.to_csv(dhmix_csv, index=False)
    logger.info(f"Saved enthalpy of mixing functional groups to {dhmix_csv}")


if __name__ == "__main__":
    main()
