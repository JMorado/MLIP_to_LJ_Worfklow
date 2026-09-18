"""integrators.py

OpenMM CustomIntegrator factories for extended Lagrangian MD.

Two integrators are provided:

* :func:`create_integrator`
    Propagates atomic monopoles (q) and dipoles (mu) as extended dynamical
    variables alongside nuclear positions using symmetric velocity Verlet.

* :func:`create_higher_multipole_integrator`
    Generalises to arbitrary multipole rank l (monopole, dipole, quadrupole,
    octupole, …).

Theory
------
The extended Lagrangian has the form

    L = (1/2) Σ_i m_i ṙ_i²
      + (1/2) m_q  Σ_i q̇_i²
      + (1/2) m_mu Σ_i |μ̇_i|²
      - V(r, q, μ)

Equations of motion:

    m_i  r̈_i = -∂V/∂r_i         (nuclear forces — computed by OpenMM)
    m_q  q̈_i = -∂V/∂q_i         (electrochemical potential gradient)
    m_mu μ̈_i = -∂V/∂μ_i         (local electric field)

Integration (symmetric velocity Verlet):

    1.  v(n+½)   = v(n)   + ½ dt f_r(n) / m
        vq(n+½)  = vq(n)  + ½ dt fq(n)  / m_q
        vmu(n+½) = vmu(n) + ½ dt fmu(n) / m_mu

    2.  r(n+1)  = r(n)  + dt v(n+½)
        q(n+1)  = q(n)  + dt vq(n+½)
        mu(n+1) = mu(n) + dt vmu(n+½)

    3.  Evaluate forces at (r(n+1), q(n+1), mu(n+1)).
        OpenMM refreshes f_r automatically via addUpdateContextState().
        The caller must refresh fq and fmu (see simulation.py).

    4.  v(n+1)   = v(n+½)   + ½ dt f_r(n+1) / m
        vq(n+1)  = vq(n+½)  + ½ dt fq(n+1)  / m_q
        vmu(n+1) = vmu(n+½) + ½ dt fmu(n+1) / m_mu

Note on fq timing
-----------------
addUpdateContextState() cannot refresh per-dof variables such as fq/fmu —
only nuclear forces f_r are updated.  The second half-kick therefore uses
fq from time n, giving a leapfrog-like O(dt²) truncation error for the
extended variables.  This is acceptable for dt ≤ 1 fs with thermostatted
charges.  Use ExtendedLagrangianSimulation.step_strict_vv() for exact VV.
"""

from __future__ import annotations

import math

from openmm import CustomIntegrator


def create_integrator(
    dt: float,
    charge_mass: float,
    dipole_mass: float,
    charge_friction: float = 0.0,
    dipole_friction: float = 0.0,
) -> CustomIntegrator:
    """
    Build an OpenMM CustomIntegrator that propagates atomic monopoles (q) and
    atomic dipoles (mu) as extended dynamical variables alongside nuclear
    positions, using symmetric velocity Verlet.

    Per-dof variables registered
    ----------------------------
    q, vq, fq   – charge value, velocity, and force (-∂E/∂q)
    mu, vmu, fmu – dipole vector, velocity, and force (-∂E/∂mu)

    Because OpenMM per-dof variables are 3-vectors, the scalar charge q uses
    only the x-component (y=z=0).  The dipole mu maps naturally to (x,y,z).

    Parameters
    ----------
    dt : float
        Timestep in picoseconds.
    charge_mass : float
        Fictitious mass for the charge DOF in amu·e².  Tune to
        ~0.01–1.0 amu·e² so that the charge oscillation period is much
        shorter than the nuclear period (adiabatic separation).
    dipole_mass : float
        Fictitious mass for the dipole DOF in amu·e²·Å².
    charge_friction : float, optional
        Langevin friction for charge DOF in ps⁻¹.  Set > 0 to thermostat
        charges independently.  Default 0.0 (NVE for charge DOF).
    dipole_friction : float, optional
        Langevin friction for dipole DOF in ps⁻¹.  Default 0.0.

    Returns
    -------
    CustomIntegrator
        Before use, initialise 'q', 'vq', 'mu', 'vmu' via
        integrator.setPerDofVariableByName() and arrange for fq/fmu to be
        populated each step (see simulation.py).
    """
    integrator = CustomIntegrator(dt)

    # Extended DOF: charges (scalar — x-component only)
    integrator.addPerDofVariable("q",   0.0)
    integrator.addPerDofVariable("vq",  0.0)
    integrator.addPerDofVariable("fq",  0.0)

    # Extended DOF: dipoles (full 3-vector)
    integrator.addPerDofVariable("mu",  0.0)
    integrator.addPerDofVariable("vmu", 0.0)
    integrator.addPerDofVariable("fmu", 0.0)

    # Fictitious masses and friction coefficients
    integrator.addGlobalVariable("mq",           charge_mass)
    integrator.addGlobalVariable("mmu",          dipole_mass)
    integrator.addGlobalVariable("gamma_q",      charge_friction)
    integrator.addGlobalVariable("gamma_mu",     dipole_friction)

    # kT for Langevin noise (kJ/mol); update after creation if needed
    integrator.addGlobalVariable("kT",           2.479)   # ~300 K
    integrator.addGlobalVariable("noisescale_q", 0.0)
    integrator.addGlobalVariable("noisescale_mu", 0.0)

    # Precompute noise scales  σ = sqrt(2 kT γ dt / m)
    integrator.addComputeGlobal("noisescale_q",  "sqrt(2*kT*gamma_q*dt/mq)")
    integrator.addComputeGlobal("noisescale_mu", "sqrt(2*kT*gamma_mu*dt/mmu)")

    # Step 1: half-kick
    integrator.addComputePerDof("v",   "v   + 0.5*dt*f/m")
    integrator.addComputePerDof("vq",  "vq*(1 - 0.5*dt*gamma_q)  + 0.5*dt*fq/mq")
    integrator.addComputePerDof("vmu", "vmu*(1 - 0.5*dt*gamma_mu) + 0.5*dt*fmu/mmu")

    # Step 2: full drift
    integrator.addComputePerDof("x",  "x  + dt*v")
    integrator.addComputePerDof("q",  "q  + dt*vq")
    integrator.addComputePerDof("mu", "mu + dt*vmu")

    # Step 3: recompute nuclear forces (fq/fmu must be refreshed externally)
    integrator.addUpdateContextState()
    integrator.addConstrainPositions()

    # Step 4: second half-kick
    integrator.addComputePerDof("v",   "v   + 0.5*dt*f/m")
    integrator.addConstrainVelocities()
    integrator.addComputePerDof(
        "vq",
        "vq*(1 - 0.5*dt*gamma_q)  + 0.5*dt*fq/mq  + noisescale_q*gaussian",
    )
    integrator.addComputePerDof(
        "vmu",
        "vmu*(1 - 0.5*dt*gamma_mu) + 0.5*dt*fmu/mmu + noisescale_mu*gaussian",
    )

    return integrator


def create_higher_multipole_integrator(
    dt: float,
    masses: dict[int, float],
    frictions: dict[int, float] | None = None,
    max_rank: int = 2,
) -> CustomIntegrator:
    """
    Extended Lagrangian integrator for spherical multipoles up to rank
    ``max_rank`` (0 = monopole, 1 = dipole, 2 = quadrupole, 3 = octupole, …).

    Because OpenMM per-dof variables are 3-vectors, rank-l multipoles
    (2l+1 components) are packed into ceil((2l+1)/3) variables.
    Use pack_multipoles / unpack_multipoles from multipoles.py to convert.

    Per-dof variables registered (for each rank l, each slot k)
    -----------------------------------------------------------
    Q{l}_{k}, vQ{l}_{k}, fQ{l}_{k}

    Parameters
    ----------
    dt : float
        Timestep in picoseconds.
    masses : dict[int, float]
        Fictitious mass per rank, e.g. {0: 0.1, 1: 0.1, 2: 0.05}.
    frictions : dict[int, float] or None
        Langevin friction (ps⁻¹) per rank.  None → NVE for all ranks.
    max_rank : int
        Highest multipole rank to include (default 2 = quadrupole).

    Returns
    -------
    CustomIntegrator
    """
    if frictions is None:
        frictions = {}

    integrator = CustomIntegrator(dt)
    integrator.addGlobalVariable("kT", 2.479)

    for l in range(max_rank + 1):
        n_comp = 2 * l + 1
        n_vars = math.ceil(n_comp / 3)
        ml      = masses.get(l, 1.0)
        gamma_l = frictions.get(l, 0.0)
        integrator.addGlobalVariable(f"mQ{l}",      ml)
        integrator.addGlobalVariable(f"gamma_Q{l}", gamma_l)
        integrator.addGlobalVariable(f"ns_Q{l}",    0.0)
        for k in range(n_vars):
            integrator.addPerDofVariable(f"Q{l}_{k}",  0.0)
            integrator.addPerDofVariable(f"vQ{l}_{k}", 0.0)
            integrator.addPerDofVariable(f"fQ{l}_{k}", 0.0)

    # Nuclear half-kick
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")

    # Noise scales and extended half-kick
    for l in range(max_rank + 1):
        n_vars = math.ceil((2 * l + 1) / 3)
        integrator.addComputeGlobal(
            f"ns_Q{l}", f"sqrt(2*kT*gamma_Q{l}*dt/mQ{l})"
        )
        for k in range(n_vars):
            integrator.addComputePerDof(
                f"vQ{l}_{k}",
                f"vQ{l}_{k}*(1 - 0.5*dt*gamma_Q{l}) + 0.5*dt*fQ{l}_{k}/mQ{l}",
            )

    # Full drift
    integrator.addComputePerDof("x", "x + dt*v")
    for l in range(max_rank + 1):
        n_vars = math.ceil((2 * l + 1) / 3)
        for k in range(n_vars):
            integrator.addComputePerDof(f"Q{l}_{k}", f"Q{l}_{k} + dt*vQ{l}_{k}")

    # Recompute nuclear forces
    integrator.addUpdateContextState()
    integrator.addConstrainPositions()

    # Second half-kick
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    integrator.addConstrainVelocities()
    for l in range(max_rank + 1):
        n_vars = math.ceil((2 * l + 1) / 3)
        for k in range(n_vars):
            integrator.addComputePerDof(
                f"vQ{l}_{k}",
                f"vQ{l}_{k}*(1 - 0.5*dt*gamma_Q{l}) + 0.5*dt*fQ{l}_{k}/mQ{l}"
                f" + ns_Q{l}*gaussian",
            )

    return integrator
