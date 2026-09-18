"""simulation.py

High-level wrapper that connects the extended Lagrangian integrator to a
user-supplied charge/dipole force function.

Usage
-----
::

    from extended_lagrangian.integrators import create_integrator
    from extended_lagrangian.simulation import ExtendedLagrangianSimulation
    from extended_lagrangian.forces import CoulombForceFn

    integrator = create_integrator(dt=0.001, charge_mass=0.1, dipole_mass=0.1)
    charge_fn  = CoulombForceFn(chi, eta, total_charge=0.0)
    sim        = ExtendedLagrangianSimulation(openmm_sim, integrator, charge_fn)

    sim.step(1000)

    charges = sim.get_charges()
    dipoles = sim.get_dipoles()
"""

from __future__ import annotations

import numpy as np


class ExtendedLagrangianSimulation:
    """
    Thin wrapper around an OpenMM Simulation that manages the charge/dipole
    forces for the extended Lagrangian integrator.

    The user supplies a ``charge_force_fn`` callable with signature::

        fq, fmu = charge_force_fn(positions, charges, dipoles)
            positions : np.ndarray  (N, 3) nm
            charges   : np.ndarray  (N,)   e
            dipoles   : np.ndarray  (N, 3) e·nm
            fq        : np.ndarray  (N,)   kJ/mol/e   (= -∂E/∂q_i)
            fmu       : np.ndarray  (N, 3) kJ/mol/e/nm (= -∂E/∂mu_i)

    For an ML potential that outputs per-atom charges through a charge-
    equilibration layer (e.g. MACE with QEq), ``charge_force_fn`` can be::

        def charge_force_fn(pos, q, mu):
            pos_t = torch.tensor(pos, requires_grad=False)
            q_t   = torch.tensor(q,   requires_grad=True)
            mu_t  = torch.tensor(mu,  requires_grad=True)
            E = mlp.energy(pos_t, q_t, mu_t)
            dEdq,  = torch.autograd.grad(E, q_t,  create_graph=False)
            dEdmu, = torch.autograd.grad(E, mu_t, create_graph=False)
            return -dEdq.numpy(), -dEdmu.numpy()

    Parameters
    ----------
    simulation : openmm.app.Simulation
        A fully set-up OpenMM Simulation whose integrator is the extended
        Lagrangian integrator from :func:`integrators.create_integrator`.
    integrator : openmm.CustomIntegrator
        The same integrator object attached to *simulation*.
    charge_force_fn : callable
        See signature above.
    """

    def __init__(self, simulation, integrator, charge_force_fn):
        self.sim = simulation
        self.integrator = integrator
        self.charge_force_fn = charge_force_fn
        self._n_atoms = simulation.system.getNumParticles()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions [nm], charges [e], dipoles [e·nm])."""
        import openmm.unit as unit
        state = self.sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        q_raw  = self.integrator.getPerDofVariableByName("q")
        mu_raw = self.integrator.getPerDofVariableByName("mu")
        q  = np.array([v[0] for v in q_raw])   # scalar uses x-component
        mu = np.array(mu_raw)                    # full 3-vector
        return pos, q, mu

    def _push_forces(self, pos: np.ndarray, q: np.ndarray, mu: np.ndarray) -> None:
        """Evaluate charge/dipole forces and write them to the integrator."""
        fq_arr, fmu_arr = self.charge_force_fn(pos, q, mu)
        n = self._n_atoms
        fq_3d = np.zeros((n, 3))
        fq_3d[:, 0] = fq_arr           # scalar charge force → x-component
        self.integrator.setPerDofVariableByName("fq",  fq_3d.tolist())
        self.integrator.setPerDofVariableByName("fmu", np.asarray(fmu_arr).tolist())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_charges(self) -> np.ndarray:
        """Return current atomic charges (N,) in elementary charge units."""
        _, q, _ = self._get_state()
        return q

    def get_dipoles(self) -> np.ndarray:
        """Return current atomic dipoles (N, 3) in e·nm."""
        _, _, mu = self._get_state()
        return mu

    def step(self, n_steps: int = 1) -> None:
        """
        Advance the simulation by ``n_steps`` steps.

        Timing note
        -----------
        ``integrator.step(1)`` uses fq/fmu from time *n* for **both**
        half-kicks.  The second half-kick cannot see fq at *n+1* because
        per-dof variables are not refreshed by ``addUpdateContextState``
        inside a single OpenMM step.  This gives a leapfrog-like O(dt²)
        truncation error for the extended variables — acceptable for
        dt ≤ 1 fs with thermostatted charges.

        For strict velocity-Verlet correctness use :meth:`step_strict_vv`.
        """
        for _ in range(n_steps):
            pos, q, mu = self._get_state()
            self._push_forces(pos, q, mu)
            self.integrator.step(1)

    def step_strict_vv(self, n_steps: int = 1) -> None:
        """
        Strict velocity-Verlet for the extended DOF, at the cost of one
        extra force evaluation per step.

        Scheme
        ------
        Within each step the Python layer explicitly performs the two
        half-kicks for the extended variables, bracketing a force refresh::

            1. half-kick extended:  vq += ½ dt fq(n) / mq
            2. drift extended:      q, mu advance by dt
               drift nuclear:       r advances by dt (inside integrator.step)
            3. recompute fq(n+1), fmu(n+1) externally
            4. second half-kick:    vq += ½ dt fq(n+1) / mq

        The nuclear DOF are handled correctly because
        ``addUpdateContextState`` refreshes f_r between the two halves.

        In practice (thermostatted charges, dt ≤ 1 fs) the difference
        between this and the simpler :meth:`step` method is negligible.
        """
        dt  = self.integrator.getStepSize()   # ps
        mq  = self.integrator.getGlobalVariableByName("mq")
        mmu = self.integrator.getGlobalVariableByName("mmu")
        n   = self._n_atoms

        for _ in range(n_steps):
            pos, q, mu = self._get_state()
            vq_raw  = self.integrator.getPerDofVariableByName("vq")
            vmu_raw = self.integrator.getPerDofVariableByName("vmu")
            vq  = np.array([v[0] for v in vq_raw])
            vmu = np.array(vmu_raw)

            fq_arr, fmu_arr = self.charge_force_fn(pos, q, mu)

            # ── first half-kick extended DOF ───────────────────────────
            vq  += 0.5 * dt * fq_arr  / mq
            vmu += 0.5 * dt * fmu_arr / mmu

            # ── drift extended DOF ─────────────────────────────────────
            q   += dt * vq
            mu  += dt * vmu

            # write updated state back so integrator drift is skipped
            q_3d  = np.zeros((n, 3)); q_3d[:, 0]  = q
            vq_3d = np.zeros((n, 3)); vq_3d[:, 0] = vq
            for name, arr in [("q", q_3d), ("mu", mu), ("vq", vq_3d), ("vmu", vmu)]:
                self.integrator.setPerDofVariableByName(name, arr.tolist())

            # zero fq/fmu so the integrator's built-in half-kicks are no-ops
            zeros3 = np.zeros((n, 3))
            self.integrator.setPerDofVariableByName("fq",  zeros3.tolist())
            self.integrator.setPerDofVariableByName("fmu", zeros3.tolist())

            # ── nuclear step + f_r refresh ─────────────────────────────
            self.integrator.step(1)

            # ── recompute extended forces at new (r(n+1), q(n+1)) ──────
            pos_new, _, _ = self._get_state()
            fq_new, fmu_new = self.charge_force_fn(pos_new, q, mu)

            # ── second half-kick extended DOF ───────────────────────────
            vq  += 0.5 * dt * fq_new  / mq
            vmu += 0.5 * dt * fmu_new / mmu

            vq_3d = np.zeros((n, 3)); vq_3d[:, 0] = vq
            fq_3d = np.zeros((n, 3)); fq_3d[:, 0] = fq_new
            for name, arr in [
                ("vq", vq_3d), ("vmu", vmu),
                ("fq", fq_3d), ("fmu", fmu_new),
            ]:
                self.integrator.setPerDofVariableByName(name, arr.tolist())
