"""Advected particle cloud inside the head, driven by the 16-anchor BFI signal.

A flock of ``n_particles`` particles is spawned in small Gaussian puffs around
each of the 16 sensor anchor points (pushed ``depth_mm`` inside along the
surface normal).  Each tick:

  1. Ages each particle by ``dt``.
  2. Advects it by its own baseline velocity plus a per-particle turbulence
     term that drifts with absolute time — gives the cloud an organic,
     swirling look rather than a uniform flow.
  3. Respawns dead particles (age > lifetime) around an anchor chosen by
     BFI-weighted probability, so more-active sensors "emit" more particles.
  4. Updates a per-particle scalar = ``bfi_anchor * fade(age)`` which drives
     the point colour (magma colormap) in the renderer.

Cost is fully vectorised numpy; ~2 ms for 2500 particles on a mid-tier CPU.

Public API:
    ParticleCloud.build(anchors, anchor_normals, n_particles=2500, depth_mm=8.0,
        sigma_mm=15.0, lifetime_range=(2.0, 4.0), speed_mm_s=4.0,
        turbulence_mm_s=10.0)
    ParticleCloud.update(dt, bfi_vec, t_abs)
    ParticleCloud.positions -> np.ndarray[n, 3]
    ParticleCloud.bfi       -> np.ndarray[n]      (particle scalar for colour)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ParticleCloud:
    """2500-ish particles that drift inside the head and carry the BFI signal."""

    def __init__(self,
                 anchors: np.ndarray,
                 anchor_normals: np.ndarray,
                 n_particles: int = 2500,
                 depth_mm: float = 8.0,
                 sigma_mm: float = 15.0,
                 lifetime_range: tuple[float, float] = (2.0, 4.0),
                 speed_mm_s: float = 4.0,
                 turbulence_mm_s: float = 10.0,
                 baseline_glow: float = 0.15,
                 seed: int | None = 42) -> None:
        if anchors.shape != (16, 3) or anchor_normals.shape != (16, 3):
            raise ValueError("anchors / anchor_normals must be (16, 3)")
        self.anchor_centers = (anchors - depth_mm * anchor_normals).astype(np.float32)
        self.sigma_mm = float(sigma_mm)
        self.lifetime_range = lifetime_range
        self.speed_mm_s = float(speed_mm_s)
        self.turbulence_mm_s = float(turbulence_mm_s)
        # Minimum per-particle scalar (before fade).  Guarantees particles
        # are visibly warm-red even at BFI=0 (e.g. during baseline warm-up).
        self.baseline_glow = float(baseline_glow)
        self.n = int(n_particles)

        self._rng = np.random.default_rng(seed)

        # Stratified assignment: every anchor gets exactly n/16 particles
        # (plus remainder distributed to the first few).  This guarantees
        # left/right symmetry regardless of BFI amplitude differences — we
        # do NOT reassign on respawn, so density is anchor-invariant and
        # only particle BRIGHTNESS tracks BFI.
        per_anchor = np.full(16, self.n // 16, dtype=np.int64)
        per_anchor[: self.n % 16] += 1
        self.anchor_idx = np.repeat(np.arange(16), per_anchor)
        self._rng.shuffle(self.anchor_idx)  # randomize draw order, not counts
        self.positions = self._spawn_positions(np.arange(self.n)).astype(np.float32)
        # Staggered starting ages so we don't get mass die-off at t=lifetime_min.
        self.age = self._rng.uniform(0.0, lifetime_range[1], size=self.n).astype(np.float32)
        self.lifetime = self._rng.uniform(
            lifetime_range[0], lifetime_range[1], size=self.n
        ).astype(np.float32)
        self.velocities = (self._rng.normal(0.0, speed_mm_s, size=(self.n, 3))
                           ).astype(np.float32)
        # Per-particle turbulence phase offsets (rad).
        self.phase = (self._rng.uniform(0.0, 2.0 * np.pi, size=(self.n, 3))
                      ).astype(np.float32)
        # Per-frame output scalar (particle colour).
        self.bfi = np.zeros(self.n, dtype=np.float32)

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, anchors: np.ndarray, anchor_normals: np.ndarray,
              **kwargs) -> "ParticleCloud":
        obj = cls(anchors, anchor_normals, **kwargs)
        logger.info("Particles: %d spawned around 16 anchors", obj.n)
        return obj

    # ------------------------------------------------------------------
    def _spawn_positions(self, idx: np.ndarray) -> np.ndarray:
        """Sample positions around the particles' chosen anchor centres."""
        centers = self.anchor_centers[self.anchor_idx[idx]]
        noise = self._rng.normal(0.0, self.sigma_mm, size=(len(idx), 3))
        return centers + noise

    def _respawn(self, idx: np.ndarray) -> None:
        """Re-seed position/age/velocity for dead particles.

        ``anchor_idx`` is NOT changed — each particle stays with its
        originally-assigned anchor for the entire run so L/R density is
        exactly symmetric regardless of per-anchor BFI amplitudes.
        """
        if idx.size == 0:
            return
        self.positions[idx] = self._spawn_positions(idx)
        self.age[idx] = 0.0
        self.lifetime[idx] = self._rng.uniform(
            self.lifetime_range[0], self.lifetime_range[1], size=idx.size
        )
        self.velocities[idx] = self._rng.normal(
            0.0, self.speed_mm_s, size=(idx.size, 3)
        )
        self.phase[idx] = self._rng.uniform(0.0, 2.0 * np.pi, size=(idx.size, 3))

    # ------------------------------------------------------------------
    def update(self, dt: float, bfi_vec: np.ndarray, t_abs: float) -> None:
        """Advance the simulation by ``dt`` seconds.

        ``bfi_vec`` is a (16,) array of current per-anchor BFI values, used
        both to weight respawn probability and to set each particle's colour.
        ``t_abs`` is absolute time (seconds) and drives the turbulence drift.
        """
        dt = float(dt)
        if dt <= 0.0:
            return

        # --- 1. Turbulence velocity (per-particle 3-axis drifting sine) ---
        tb = self.turbulence_mm_s
        turb = np.column_stack([
            tb * np.sin(1.1 * t_abs + self.phase[:, 0]),
            tb * np.cos(0.9 * t_abs + self.phase[:, 1]),
            tb * np.sin(1.3 * t_abs + self.phase[:, 2]),
        ]).astype(np.float32)

        # --- 2. Weak restoring force toward the owning anchor centre ---
        # Ornstein-Uhlenbeck-like: each particle is pulled back at rate
        # ``restore_rate`` (1/s) so it never drifts more than ~sigma_mm
        # away from its anchor, no matter how long it lives.
        restore_rate = 0.5  # 1/s — stronger = tighter plume
        restore = restore_rate * (
            self.anchor_centers[self.anchor_idx] - self.positions
        )

        # --- 3. Advect ---
        self.positions += (self.velocities + turb + restore) * dt

        # --- 4. Age, kill, respawn ---
        self.age += dt
        dead_mask = self.age > self.lifetime
        if dead_mask.any():
            self._respawn(np.flatnonzero(dead_mask))

        # --- 5. Per-particle scalar = (baseline_glow + bfi_anchor) * fade ---
        # Bell-curve fade that peaks at age = lifetime/2 (value 1.0) and is
        # 0 at birth and death.  baseline_glow keeps particles visible on
        # the warm end of the colormap even when BFI is zero, so the cloud
        # never disappears into the dark background.
        age_norm = self.age / np.maximum(self.lifetime, 1e-6)
        fade = 4.0 * age_norm * (1.0 - age_norm)
        np.clip(fade, 0.0, 1.0, out=fade)
        anchor_vals = np.asarray(bfi_vec, dtype=np.float32)[self.anchor_idx]
        self.bfi = (self.baseline_glow + anchor_vals) * fade.astype(np.float32)
