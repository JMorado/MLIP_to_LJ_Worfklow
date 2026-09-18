"""
This script compares two datasets of physical properties.
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
