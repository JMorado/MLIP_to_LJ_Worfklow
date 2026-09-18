"""multipoles.py

Utilities for packing and unpacking spherical multipole components into the
3-vector per-dof variables used by OpenMM's CustomIntegrator.

Background
----------
OpenMM per-dof variables are 3-vectors (x, y, z).  A rank-l spherical
multipole has 2l+1 real components  Q_{l,m}  for  m = -l, …, +l.
We pack them densely into ``ceil((2l+1) / 3)`` per-dof variables:

    rank  components  per-dof vars  layout (0-indexed within each var)
    ────  ──────────  ────────────  ────────────────────────────────────
     0     Q_{00}          1        var0: (Q00,  0,   0 )
     1     Q_{1-1..1}      1        var0: (Q1-1, Q10, Q11)
     2     Q_{2-2..2}      2        var0: (Q2-2, Q2-1, Q20)
                                    var1: (Q21,  Q22,  0 )
     3     Q_{3-3..3}      3        var0: (Q3-3, Q3-2, Q3-1)
                                    var1: (Q30,  Q31,  Q32)
                                    var2: (Q33,  0,    0 )
     4     Q_{4-4..4}      3        var0–var2, 9 slots for 9 components

General formula: n_vars(l) = ceil((2l+1) / 3)

Relationship to Cartesian multipoles
--------------------------------------
Cartesian quadrupole Q_αβ (symmetric, traceless → 5 independent components)
maps to spherical Q_{2m} via::

    Q_{2,0}  = sqrt(3/4) * Qzz
    Q_{2,±1} = sqrt(3)   * Qxz, Qyz
    Q_{2,±2} = (sqrt(3)/2) * (Qxx - Qyy),  sqrt(3) * Qxy

AMOEBA uses Cartesian multipoles through quadrupoles as *fixed* (not extended)
DOF.  For dynamic (extended Lagrangian) higher multipoles the spherical form
is preferred because the 2l+1 components are orthogonal and the Wigner
D-matrix cleanly handles frame rotations.

Force on multipoles from an ML potential (autograd sketch)
----------------------------------------------------------
If your ML model outputs energy as a differentiable function of spherical
multipole moments, the force on Q_{i,lm} is the autograd gradient::

    import torch
    Q = torch.tensor(Q_lm, requires_grad=True)   # (N, 2l+1)
    E = mlp.energy(positions, Q)
    dEdQ = torch.autograd.grad(E, Q)[0]           # (N, 2l+1)
    fQ_lm = -dEdQ.numpy()

For a purely electrostatic multipole interaction::

    V = Σ_{i<j} Σ_{l,l'} Σ_{m,m'}
          T^{l,m}_{l',m'}(r_ij) * Q_{i,lm} * Q_{j,l'm'}

where  T^{l,m}_{l',m'}(r)  is the interaction tensor (real-valued regular/
irregular solid harmonic product, tabulated in Stone's "Theory of
Intermolecular Forces").  The force is then::

    -∂V/∂Q_{i,lm} = -Σ_{j≠i} Σ_{l',m'} T^{l,m}_{l',m'}(r_ij) * Q_{j,l'm'}

which is a linear map (O(N²) or O(N log N) with fast multipole methods).

References
----------
- AMOEBA (OpenMM AmoebaMultipoleForce) — fixed multipoles l ≤ 2
- Tinker-HP — dynamic multipoles, PIMD extensions
- LICHEM — QM/MM with fluctuating multipoles
"""

from __future__ import annotations

import math

import numpy as np


def pack_multipoles(Q_lm: np.ndarray, l: int) -> list[np.ndarray]:
    """
    Pack a (N, 2l+1) array of spherical multipole components into
    ``ceil((2l+1)/3)`` arrays of shape (N, 3), ready for
    ``integrator.setPerDofVariableByName``.

    Parameters
    ----------
    Q_lm : np.ndarray, shape (N, 2l+1)
        Multipole components for N atoms, ordered m = -l … +l.
    l : int
        Multipole rank.

    Returns
    -------
    list of np.ndarray, each shape (N, 3)

    Examples
    --------
    >>> q = np.random.randn(100, 1)   # monopoles
    >>> packed = pack_multipoles(q, l=0)    # 1 array of shape (100, 3)
    >>> mu = np.random.randn(100, 3)   # dipoles
    >>> packed = pack_multipoles(mu, l=1)   # 1 array of shape (100, 3)
    """
    n_atoms, n_comp = Q_lm.shape
    expected = 2 * l + 1
    if n_comp != expected:
        raise ValueError(f"Expected {expected} components for l={l}, got {n_comp}")
    n_vars = math.ceil(n_comp / 3)
    padded = np.zeros((n_atoms, n_vars * 3))
    padded[:, :n_comp] = Q_lm
    return [padded[:, 3 * k : 3 * k + 3] for k in range(n_vars)]


def unpack_multipoles(packed: list[np.ndarray], l: int) -> np.ndarray:
    """
    Inverse of :func:`pack_multipoles`.

    Parameters
    ----------
    packed : list of np.ndarray, each shape (N, 3)
        Per-dof variable values retrieved via
        ``integrator.getPerDofVariableByName``.
    l : int
        Multipole rank.

    Returns
    -------
    Q_lm : np.ndarray, shape (N, 2l+1)
        Components ordered m = -l … +l.
    """
    full = np.concatenate(packed, axis=1)   # (N, n_vars*3)
    return full[:, : 2 * l + 1]             # trim padding
