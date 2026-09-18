"""extended_lagrangian — OpenMM extended Lagrangian MD for fluctuating multipoles.

Package layout
--------------
integrators.py
    OpenMM CustomIntegrator factories.

    * create_integrator(dt, charge_mass, dipole_mass, ...)
        Symmetric VV integrator for monopoles (q) and dipoles (mu).
    * create_higher_multipole_integrator(dt, masses, frictions, max_rank)
        Generalises to arbitrary multipole rank.

simulation.py
    Python-level simulation wrapper that manages charge/dipole forces.

    * ExtendedLagrangianSimulation(simulation, integrator, charge_force_fn)
        .step(n)           — simple loop (leapfrog-like O(dt²) for ext. DOF)
        .step_strict_vv(n) — exact VV at cost of 2× force evals per step

multipoles.py
    Packing utilities for spherical multipoles ↔ OpenMM per-dof 3-vectors.

    * pack_multipoles(Q_lm, l)   → list[np.ndarray]
    * unpack_multipoles(packed, l) → np.ndarray

forces.py
    Charge/dipole force functions  (fq, fmu) = -∂E/∂(q, mu).

    * coulomb_site_potentials(positions, charges) → phi
    * CoulombForceFn(chi, eta, total_charge)      — classical QEq model
    * MLPotentialForceFn(model, nonbonded, ctx)   — ML potential via autograd

Quick-start example
-------------------
::

    import numpy as np
    import openmm as mm
    import openmm.app as app

    from extended_lagrangian import (
        create_integrator,
        ExtendedLagrangianSimulation,
        CoulombForceFn,
    )

    # 1. Build your OpenMM system / topology as usual.
    # 2. Add a NonbondedForce with initial (guess) charges.
    # 3. Create integrator and simulation.

    integrator = create_integrator(dt=0.001, charge_mass=0.1, dipole_mass=0.1,
                                   charge_friction=10.0, dipole_friction=10.0)
    sim = app.Simulation(topology, system, integrator)

    # 4. Initialise per-dof charge variables.
    n = topology.getNumAtoms()
    integrator.setPerDofVariableByName("q",   [[0.0, 0.0, 0.0]] * n)
    integrator.setPerDofVariableByName("vq",  [[0.0, 0.0, 0.0]] * n)
    integrator.setPerDofVariableByName("mu",  [[0.0, 0.0, 0.0]] * n)
    integrator.setPerDofVariableByName("vmu", [[0.0, 0.0, 0.0]] * n)

    # 5. Create charge force function.
    chi = np.zeros(n)   # electronegativities (kJ/mol/e)
    eta = np.ones(n)    # hardnesses (kJ/mol/e²)
    charge_fn = CoulombForceFn(chi, eta, total_charge=0.0)

    # 6. Wrap and run.
    esim = ExtendedLagrangianSimulation(sim, integrator, charge_fn)
    esim.step(1000)

    print("charges:", esim.get_charges())
"""

from .forces import CoulombForceFn, MLPotentialForceFn, coulomb_site_potentials
from .integrators import create_higher_multipole_integrator, create_integrator
from .multipoles import pack_multipoles, unpack_multipoles
from .simulation import ExtendedLagrangianSimulation

__all__ = [
    "create_integrator",
    "create_higher_multipole_integrator",
    "ExtendedLagrangianSimulation",
    "pack_multipoles",
    "unpack_multipoles",
    "coulomb_site_potentials",
    "CoulombForceFn",
    "MLPotentialForceFn",
]
