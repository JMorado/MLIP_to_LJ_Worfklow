"""forces.py

Force function implementations for the extended Lagrangian charge/dipole DOF.

Each class / function here computes  (fq, fmu) = (-∂E/∂q, -∂E/∂mu)  given
the current positions, charges, and dipoles.  The returned arrays are passed
to :class:`simulation.ExtendedLagrangianSimulation` as the ``charge_force_fn``.

Two contributions involve the charges at every step:

    A.  q  →  f_nuclear   (charges exert Coulomb forces on nuclear positions)
    B.  r  →  fq          (nuclear positions determine the potential at each
                            site, which drives the charge DOF)

These flow in opposite directions and must both be evaluated every step.

────────────────────────────────────────────────────────────────────────────
A. q → f_nuclear
────────────────────────────────────────────────────────────────────────────

The cleanest route is a NonbondedForce (or CustomNonbondedForce) whose
per-particle charges are updated from the current q_i each step via
``updateParametersInContext()``.  OpenMM then handles PBC, PME, and
long-range corrections automatically.

Setup (once, before the simulation)::

    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.PME)
    nonbonded.setCutoffDistance(1.2 * nanometer)
    for i in range(n_atoms):
        nonbonded.addParticle(q0[i], sigma[i], epsilon[i])
    system.addForce(nonbonded)

Per-step update (inside charge_force_fn)::

    for i, q in enumerate(charges):
        _, sig, eps = nonbonded.getParticleParameters(i)
        nonbonded.setParticleParameters(i, float(q), sig, eps)
    nonbonded.updateParametersInContext(context)

NOTE: ``updateParametersInContext`` only works if you do NOT change which
particles exist or their exclusions — only charge/sigma/eps values.

────────────────────────────────────────────────────────────────────────────
B. r → fq  (electrostatic potential → force on charges)
────────────────────────────────────────────────────────────────────────────

The force on charge q_i is the (negative) electrochemical potential::

    fq_i = -∂V/∂q_i = -(χ_i + η_i·q_i  +  Σ_{j≠i} J_ij(r_ij)·q_j)

The Coulomb-kernel term is the electrostatic potential::

    φ_i = Σ_{j≠i} q_j / (4πε₀ r_ij)   (kJ/mol/e if r in nm, q in e)

Classes
-------
:func:`coulomb_site_potentials`
    Direct O(N²) summation of  φ_i.

:class:`CoulombForceFn`
    Classical QEq model combining χ, η, and Coulomb potential.

:class:`MLPotentialForceFn`
    ML potential (e.g. MACE) where ∂E/∂q is obtained via autograd.
"""

from __future__ import annotations

import numpy as np


def coulomb_site_potentials(
    positions: np.ndarray,
    charges: np.ndarray,
    coulomb_constant: float = 138.935,
) -> np.ndarray:
    """
    Compute the electrostatic potential  φ_i = Σ_{j≠i} k·q_j/r_ij  at each
    site by direct O(N²) summation.

    Suitable for small systems (N < 2000) or when periodic boundary conditions
    are not needed.  For production use with PBC, extract φ_i from OpenMM's
    ``CustomNonbondedForce`` or use :class:`MLPotentialForceFn` with a model
    that handles long-range electrostatics internally.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Atomic positions in **nanometres**.
    charges : np.ndarray, shape (N,)
        Atomic charges in elementary charge units (e).
    coulomb_constant : float
        k = 1/(4πε₀) in kJ/mol·nm/e².  Default is the OpenMM value
        138.935 kJ/mol·nm/e².

    Returns
    -------
    phi : np.ndarray, shape (N,)
        Electrostatic potential at each site in kJ/mol/e.
        The force on charge q_i is  fq_i = -(chi_i + eta_i·q_i + phi_i).
    """
    r_ij = positions[:, None, :] - positions[None, :, :]   # (N, N, 3)
    dist  = np.linalg.norm(r_ij, axis=-1)                   # (N, N)
    np.fill_diagonal(dist, np.inf)                          # exclude self
    phi = coulomb_constant * (charges[None, :] / dist).sum(axis=1)  # (N,)
    return phi


class CoulombForceFn:
    """
    Classical QEq charge-force function.

    Energy decomposition::

        V = Σ_i χ_i·q_i
          + (1/2) Σ_i η_i·q_i²
          + (1/2) Σ_{i≠j} k·q_i·q_j/r_ij

    Force on each charge DOF::

        fq_i = -(χ_i + η_i·q_i + φ_i)   where φ_i = Σ_{j≠i} k·q_j/r_ij

    Nuclear Coulomb forces are NOT returned here; they are computed by
    OpenMM's NonbondedForce after ``updateParametersInContext()`` is called
    externally (see module docstring and :class:`MLPotentialForceFn`).

    Total-charge constraint
    -----------------------
    The extended Lagrangian has no built-in charge conservation.  We maintain
    Σ_i q_i = Q_tot by projecting the charge force onto the constraint
    surface via a Lagrange multiplier::

        fq_i  →  fq_i - λ,    λ = (1/N) Σ_i fq_i

    This removes the component of fq that would otherwise shift total charge.

    Parameters
    ----------
    electronegativities : np.ndarray, shape (N,)
        χ_i values in kJ/mol/e.
    hardnesses : np.ndarray, shape (N,)
        η_i values in kJ/mol/e².
    total_charge : float
        Target total charge  Σ_i q_i.  Used only as documentation here;
        enforcement is via the fq projection above.
    coulomb_constant : float
        k = 1/(4πε₀) in kJ/mol·nm/e².  Default: 138.935.
    """

    def __init__(
        self,
        electronegativities: np.ndarray,
        hardnesses: np.ndarray,
        total_charge: float = 0.0,
        coulomb_constant: float = 138.935,
    ):
        self.chi   = np.asarray(electronegativities)
        self.eta   = np.asarray(hardnesses)
        self.Q_tot = total_charge
        self.k     = coulomb_constant

    def __call__(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        dipoles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)   — in nm
        charges   : np.ndarray, shape (N,)     — in e
        dipoles   : np.ndarray, shape (N, 3)   — in e·nm (unused here)

        Returns
        -------
        fq  : np.ndarray, shape (N,)    — kJ/mol/e
        fmu : np.ndarray, shape (N, 3)  — zeros (no dipole coupling)
        """
        phi = coulomb_site_potentials(positions, charges, self.k)
        fq  = -(self.chi + self.eta * charges + phi)
        fq -= fq.mean()   # Lagrange-multiplier projection for charge conservation
        fmu = np.zeros((len(charges), 3))
        return fq, fmu


class MLPotentialForceFn:
    """
    Charge-force function for an ML potential (e.g. MACE) that supports
    per-atom charges as differentiable inputs.

    The ML model must accept charges as a PyTorch tensor with
    ``requires_grad=True`` so that  ∂E/∂q  can be computed via a single
    backward pass through the model::

        pos_t = torch.tensor(positions, dtype=torch.float64)
        q_t   = torch.tensor(charges,   dtype=torch.float64, requires_grad=True)
        E     = model.energy(pos_t, q_t)
        dEdq, = torch.autograd.grad(E, q_t)
        fq    = -dEdq.numpy()

    This is equivalent to one forward + one backward pass; no finite-
    difference approximation is needed.  If the model uses a QEq layer that
    *internally* solves for charges, you need to restructure it so that
    charges are accepted as external inputs (bypass the internal QEq solve
    during dynamics, use the extended Lagrangian to propagate them instead).

    After computing fq, the current charges are pushed into OpenMM's
    NonbondedForce via ``updateParametersInContext()`` so that the nuclear
    forces at the next OpenMM force evaluation correctly include Coulomb
    contributions from the dynamic charges.

    Parameters
    ----------
    model : callable
        PyTorch model.  Called as  ``E = model.energy(pos_tensor, q_tensor)``.
    nonbonded_force : openmm.NonbondedForce
        The OpenMM NonbondedForce to update with current charges each step.
    context : openmm.Context
        The simulation context (needed for updateParametersInContext).
    sigma : np.ndarray, shape (N,)
        LJ sigma parameters (fixed) in nm.
    epsilon : np.ndarray, shape (N,)
        LJ epsilon parameters (fixed) in kJ/mol.
    """

    def __init__(self, model, nonbonded_force, context, sigma, epsilon):
        self.model      = model
        self.nonbonded  = nonbonded_force
        self.context    = context
        self.sigma      = np.asarray(sigma)
        self.epsilon    = np.asarray(epsilon)

    def __call__(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        dipoles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        1. Run the ML model to get fq = -∂E_ML/∂q via autograd.
        2. Update OpenMM NonbondedForce charges so that f_r includes Coulomb.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)   — in nm
        charges   : np.ndarray, shape (N,)     — in e
        dipoles   : np.ndarray, shape (N, 3)   — in e·nm (unused here)

        Returns
        -------
        fq  : np.ndarray, shape (N,)    — kJ/mol/e
        fmu : np.ndarray, shape (N, 3)  — zeros (no dipole coupling)
        """
        import torch

        pos_t = torch.tensor(positions, dtype=torch.float64, requires_grad=False)
        q_t   = torch.tensor(charges,   dtype=torch.float64, requires_grad=True)

        E = self.model.energy(pos_t, q_t)
        (dEdq,) = torch.autograd.grad(E, q_t)
        fq = -dEdq.detach().numpy()   # (N,)

        fq -= fq.mean()   # Lagrange-multiplier projection for charge conservation

        # Push current charges into OpenMM so nuclear forces use them
        for i, q in enumerate(charges):
            self.nonbonded.setParticleParameters(
                i, float(q), float(self.sigma[i]), float(self.epsilon[i])
            )
        self.nonbonded.updateParametersInContext(self.context)

        fmu = np.zeros((len(charges), 3))
        return fq, fmu
