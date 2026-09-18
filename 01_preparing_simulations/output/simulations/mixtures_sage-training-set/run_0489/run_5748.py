import time
import os

import numpy as np
import openmm
from openff.interchange import Interchange
from openff.interchange.components._packmol import UNIT_CUBE, pack_box
from openff.toolkit import ForceField, Molecule, unit

N_MOL = 500

def create_simulation(
    interchange: Interchange,
    dcd_stride: int = 1000,
    trajectory_name: str = "trajectory.dcd",
) -> openmm.app.Simulation:
    integrator = openmm.LangevinMiddleIntegrator(
        300 * openmm.unit.kelvin,
        1 / openmm.unit.picosecond,
        1 * openmm.unit.femtoseconds,
    )

    barostat = openmm.MonteCarloBarostat(
        1.0 * openmm.unit.bar,
        300 * openmm.unit.kelvin,
        25,
    )

    simulation = interchange.to_openmm_simulation(
        combine_nonbonded_forces=True,
        integrator=integrator,
        additional_forces=[barostat],
    )

    simulation.minimizeEnergy()

    simulation.context.setVelocitiesToTemperature(300 * openmm.unit.kelvin)
    simulation.context.computeVirtualSites()

    pdb_reporter = openmm.app.DCDReporter(trajectory_name, dcd_stride)
    state_data_reporter = openmm.app.StateDataReporter(
        "data.csv",
        100,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True,
        volume=True,
        density=True,
    )
    simulation.reporters.append(pdb_reporter)
    simulation.reporters.append(state_data_reporter)

    return simulation


def run_simulation(simulation: openmm.app.Simulation):
    print("Starting simulation")
    start_time = time.process_time()
    simulation.step(10000000)
    end_time = time.process_time()
    print(f"Elapsed time: {(end_time - start_time):.2f} seconds")

smiles1 = "CC(=O)N(C)C"
smiles2 = "ClCCCl"
x1 = 0.202
x2 = 0.798
density = nan

if smiles1 != "":
    mol1 = Molecule.from_smiles(smiles1)
    nmols1 = int(x1 * N_MOL)
else:
    mol1 = np.nan

if smiles2 != "":
    mol2 = Molecule.from_smiles(smiles2)
    nmols2 = int(x2 * N_MOL)
else:
    mol2 = np.nan

molecules = [mol1] if mol1 is not np.nan else []
molecules += [mol2] if mol2 is not np.nan else []
number_of_copies = [nmols1] if mol1 is not np.nan else []
number_of_copies += [nmols2] if mol2 is not np.nan else []

# Create the topology
topology = pack_box(
    molecules=molecules,
    number_of_copies=number_of_copies,
    box_shape=UNIT_CUBE,
    target_density=density * unit.grams / unit.milliliters,
)

# Create the ForceField and interchange
sage = ForceField("openff-2.3.0.offxml")
interchange = Interchange.from_smirnoff(force_field=sage, topology=topology)

# Create and run the simulation
simulation = create_simulation(interchange)
run_simulation(simulation)