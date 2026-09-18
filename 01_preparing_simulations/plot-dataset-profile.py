"""
This script generates plots for the functional group profiles in the dataset.
It reads a CSV file containing functional group data,
filters it to find the top groups with the greatest differences in counts across datasets,
and then creates a bar plot of these groups.
The plot is saved to a specified output file.
"""

import logging

import click
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def get_subframe_with_greatest_differences(dataframe, n_groups: int = 25):
    grouped = dataframe.groupby("Group")
    max_diff = grouped["Count"].max() - grouped["Count"].min()
    top_groups = max_diff.nlargest(n_groups).index
    subframe = dataframe[dataframe["Group"].isin(top_groups)]
    return subframe

@click.command()
@click.option(
    "--input-file",
    "-i",
    required=True,
    help="The input CSV file containing functional group data",
)
@click.option(
    "--output-file",
    "-o",
    required=True,
    help="The output file to save the plot",
)
@click.option(
    "--name",
    "-s",
    required=True,
    help="Name to be used in the plot title",
)
@click.option(
    "--n-groups",
    "-n",
    default=25,
    help="Number of top functional groups to display",
)
@click.option(
    "--figwidth",
    "-w",
    default=8,
    help="Width of the figure",
)
@click.option(
    "--figheight",
    "-h",
    default=7,
    help="Height of the figure",
)
def main(
    input_file: str,
    output_file: str,
    name: str,
    n_groups: int = 25,
    figwidth: int = 8,
    figheight: int = 7,
):
    df = pd.read_csv(input_file)
    subframe = get_subframe_with_greatest_differences(df, n_groups)

    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    ax = sns.barplot(
        ax=ax,
        data=subframe,
        x="Count",
        y="Group",
        hue="Dataset",
    )
    ax.set_ylabel("")
    ax.set_title(f"Top {n_groups} Functional Groups with Greatest Differences\n({name})")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    logger.info(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()
