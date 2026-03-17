"""
Dimension 2 — Cross-Column Logic & Dependency Evaluator.

Theoretical framing
-------------------
Distributional fidelity (Dimension 1) measures whether the marginal and joint
*statistics* are preserved.  Dimension 2 asks a harder, qualitatively different
question: does the synthetic data obey the *deterministic logical constraints*
that govern the real world?

A model that memorises aggregate statistics can still violate:
    - Functional dependencies ("zip code X always implies city Y")
    - Ontological hierarchies ("this city cannot be in that state")
    - Arithmetic identities ("unit_price × quantity = total_price")

These violations cannot be detected by KS tests or correlation matrices; they
require evaluating the data against an explicit semantic knowledge base.

Metrics implemented
-------------------
DSI  — Distributional Similarity Index (universal)
    Fits a Gaussian Mixture Model to the *real* continuous-feature manifold.
    Scores synthetic samples against this model.  Reports both the raw
    mean log-likelihood and the gap from the real-data baseline.

ICVR — Inter-Column Violation Rate (conditional on LogicSpec.known_fds)
    For each registered A→B functional dependency, measures the fraction of
    synthetic rows whose B-value contradicts the A→B mapping from real data.

HCS  — Hierarchical Consistency Score (conditional on LogicSpec.hierarchies)
    For each registered coarse-to-fine chain (e.g. country→state→city),
    measures the fraction of synthetic rows with co-occurrence combinations
    absent from real data.

MDI  — Multivariate Dependency Index (conditional on LogicSpec.math_equations)
    For each registered arithmetic identity (A op B = C), measures the
    fraction of synthetic rows where the relative error exceeds ε.

Engineering guarantees
----------------------
- DSI operates exclusively on continuous features (GMM is undefined for
  categorical inputs); PCA is applied when n_continuous > 20 to avoid
  the curse of dimensionality.
- BIC-based automatic GMM component selection avoids over-fitting on
  small datasets while allowing sufficient expressiveness on large ones.
- ICVR validates that the declared FD actually holds in *real_df* before
  computing violations in synthetic data.  A "soft FD" (violated in real
  data beyond the tolerance) triggers a warning but continues.
- MDI uses relative error with a small denominator-regulariser (1e-9) and
  configurable epsilon (default 1e-3) to avoid penalising float-precision
  truncation from LLM tokenisation.
- All conditional metrics return NaN when LogicSpec prerequisites are absent;
  callers never receive exceptions for missing metadata.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from llm_gtd_benchmark.core.logic_spec import LogicSpec, MathEquation
from llm_gtd_benchmark.core.schema import DataSchema

logger = logging.getLogger(__name__)

_NAN = float("nan")
_MDI_EPSILON = 1e-3        # relative-error tolerance for arithmetic identities
_PCA_MAX_FEATURES = 20     # apply PCA when n_continuous exceeds this threshold
_GMM_CANDIDATES = [1, 2, 3, 5, 8]  # BIC search grid for n_components


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Dim2Result:
    """Output of :class:`LogicEvaluator`.

    Attributes
    ----------
    dsi_synth_ll:
        Mean log-likelihood of synthetic samples under the GMM fitted on real
        continuous features.  Higher = better (synthetic is more likely under
        the real distribution).  Comparable only within the same dataset.
    dsi_real_ll:
        Mean log-likelihood of real held-out samples under the same GMM.
        Serves as the upper-bound reference for dsi_synth_ll.
    dsi_gap:
        ``dsi_real_ll - dsi_synth_ll``.  Lower = better (0 = perfect match).
        This is the cross-dataset-comparable form of DSI.
    icvr:
        Inter-Column Violation Rate — fraction of synthetic rows violating at
        least one registered functional dependency.  Range [0, 1]; lower = better.
        NaN when LogicSpec.known_fds is empty.
    icvr_per_fd:
        Per-FD violation rates keyed as ``"A→B"``.
    hcs_violation_rate:
        Hierarchical Consistency violation rate — fraction of synthetic rows with
        illegal (ancestor, descendant) co-occurrences.  Range [0, 1]; lower = better.
        NaN when LogicSpec.hierarchies is empty.
    hcs_per_chain:
        Per-chain violation rates keyed as ``"Col1→Col2→..."``.
    mdi_per_equation:
        Per-equation arithmetic violation rates keyed as ``"A op B = C"``.
        NaN when LogicSpec.math_equations is empty.
    mdi_mean:
        Mean across all registered equations.
    gmm_n_components:
        Number of GMM components selected by BIC (informational).
    """

    # DSI
    dsi_synth_ll: float = _NAN
    dsi_real_ll: float = _NAN
    dsi_gap: float = _NAN
    gmm_n_components: int = 0

    # ICVR
    icvr: float = _NAN
    icvr_per_fd: Dict[str, float] = field(default_factory=dict)

    # HCS
    hcs_violation_rate: float = _NAN
    hcs_per_chain: Dict[str, float] = field(default_factory=dict)

    # MDI
    mdi_per_equation: Dict[str, float] = field(default_factory=dict)
    mdi_mean: float = _NAN

    @property
    def summary(self) -> str:
        def _f(v: float) -> str:
            return f"{v:.4f}" if not np.isnan(v) else "  N/A "

        lines = [
            "── Dimension 2: Cross-Column Logic & Dependencies ────────",
            "  DSI (Distributional Similarity Index):",
            f"    Synth log-likelihood        : {_f(self.dsi_synth_ll)}",
            f"    Real  log-likelihood (ref)  : {_f(self.dsi_real_ll)}",
            f"    Gap   (↓ better, 0=perfect) : {_f(self.dsi_gap)}",
            f"    GMM components selected     : {self.gmm_n_components}",
            "  ICVR (Functional Dependency Violation Rate, ↓ better):",
        ]
        if self.icvr_per_fd:
            for key, val in self.icvr_per_fd.items():
                lines.append(f"    {key:<35}: {_f(val)}")
            lines.append(f"    Overall ICVR               : {_f(self.icvr)}")
        else:
            lines.append("    N/A (no FDs registered)")

        lines.append("  HCS (Hierarchy Violation Rate, ↓ better):")
        if self.hcs_per_chain:
            for key, val in self.hcs_per_chain.items():
                lines.append(f"    {key:<35}: {_f(val)}")
            lines.append(f"    Overall HCS violation rate : {_f(self.hcs_violation_rate)}")
        else:
            lines.append("    N/A (no hierarchies registered)")

        lines.append("  MDI (Arithmetic Violation Rate, ↓ better):")
        if self.mdi_per_equation:
            for key, val in self.mdi_per_equation.items():
                lines.append(f"    {key:<35}: {_f(val)}")
            lines.append(f"    Mean MDI violation rate    : {_f(self.mdi_mean)}")
        else:
            lines.append("    N/A (no equations registered)")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class LogicEvaluator:
    """Dimension 2 cross-column logic and dependency evaluator.

    Parameters
    ----------
    schema:
        :class:`~llm_gtd_benchmark.core.schema.DataSchema` from real data.
    real_df:
        The real training DataFrame.
    logic_spec:
        Optional :class:`~llm_gtd_benchmark.core.logic_spec.LogicSpec`
        carrying semantic constraint metadata.  When ``None``, only the
        universal DSI metric is computed.
    gmm_n_components:
        Number of GMM components for DSI.  ``"auto"`` (default) selects the
        best value from ``[1, 2, 3, 5, 8]`` via BIC on the real data.
    mdi_epsilon:
        Relative-error tolerance for arithmetic-identity violations.
        Default: 1e-3 (tolerates floating-point truncation from LLM output).
    random_state:
        Reproducibility seed.

    Examples
    --------
    >>> spec = LogicSpec(name="adult", known_fds=[("education_num", "education")])
    >>> evaluator = LogicEvaluator(schema, real_df, logic_spec=spec)
    >>> result = evaluator.evaluate(clean_df)
    >>> print(result.summary)
    """

    def __init__(
        self,
        schema: DataSchema,
        real_df: pd.DataFrame,
        logic_spec: Optional[LogicSpec] = None,
        gmm_n_components: Union[int, str] = "auto",
        mdi_epsilon: float = _MDI_EPSILON,
        random_state: int = 42,
    ) -> None:
        self.schema = schema
        self.real_df = real_df.reset_index(drop=True)
        self.logic_spec = logic_spec or LogicSpec(name="(unnamed)")
        self.gmm_n_components = gmm_n_components
        self.mdi_epsilon = mdi_epsilon
        self.random_state = random_state

        self._cont_cols: List[str] = [c.name for c in schema.continuous_columns]

        # Fit shared preprocessing (scaler + optional PCA) on real data once.
        self._scaler = StandardScaler()
        self._pca: Optional[PCA] = None
        self._gmm: Optional[GaussianMixture] = None
        self._gmm_k: int = 0

        if self._cont_cols:
            self._fit_dsi_pipeline()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, clean_synth_df: pd.DataFrame) -> Dim2Result:
        """Compute all applicable Dimension 2 logic metrics.

        Parameters
        ----------
        clean_synth_df:
            Output of :meth:`StructuralInterceptor.evaluate` — structurally
            valid and dtype-correct.

        Returns
        -------
        :class:`Dim2Result`
        """
        synth = clean_synth_df.reset_index(drop=True)
        result = Dim2Result(gmm_n_components=self._gmm_k)

        # ── DSI (universal) ───────────────────────────────────────────────────
        if self._gmm is not None and self._cont_cols:
            result.dsi_synth_ll, result.dsi_real_ll, result.dsi_gap = (
                self._calc_dsi(synth)
            )
        else:
            logger.warning("DSI skipped: no continuous columns in schema.")

        # ── ICVR (conditional) ────────────────────────────────────────────────
        if self.logic_spec.is_applicable("ICVR"):
            result.icvr_per_fd, result.icvr = self._calc_icvr(synth)

        # ── HCS (conditional) ─────────────────────────────────────────────────
        if self.logic_spec.is_applicable("HCS"):
            result.hcs_per_chain, result.hcs_violation_rate = self._calc_hcs(synth)

        # ── MDI (conditional) ─────────────────────────────────────────────────
        if self.logic_spec.is_applicable("MDI"):
            result.mdi_per_equation, result.mdi_mean = self._calc_mdi(synth)

        return result

    # ── DSI pipeline ─────────────────────────────────────────────────────────

    def _fit_dsi_pipeline(self) -> None:
        """Fit StandardScaler + optional PCA + GMM on real continuous features."""
        X_real = self.real_df[self._cont_cols].astype(float).values

        # Remove rows with NaN (edge case: real data may have missing values)
        X_real = X_real[~np.isnan(X_real).any(axis=1)]
        if len(X_real) == 0:
            logger.warning("DSI: all real rows have NaN in continuous columns; skipping.")
            return

        # StandardScale
        X_scaled = self._scaler.fit_transform(X_real)

        # PCA when dimensionality is high
        n_features = X_scaled.shape[1]
        if n_features > _PCA_MAX_FEATURES:
            n_components = min(_PCA_MAX_FEATURES, X_scaled.shape[0] - 1)
            self._pca = PCA(n_components=n_components, random_state=self.random_state)
            X_scaled = self._pca.fit_transform(X_scaled)
            logger.debug(
                "DSI: applied PCA %d → %d features (variance retained: %.1f%%).",
                n_features,
                n_components,
                self._pca.explained_variance_ratio_.sum() * 100,
            )

        # BIC-based GMM component selection
        self._gmm, self._gmm_k = self._select_gmm(X_scaled)

    def _select_gmm(self, X: np.ndarray) -> Tuple[GaussianMixture, int]:
        """Select GMM via BIC on the real data manifold."""
        if self.gmm_n_components != "auto":
            k = int(self.gmm_n_components)
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=self.random_state,
                max_iter=200,
            )
            gmm.fit(X)
            logger.debug("DSI: GMM fitted with n_components=%d (manual).", k)
            return gmm, k

        best_bic = np.inf
        best_gmm = None
        best_k = 1
        n_max = min(max(_GMM_CANDIDATES), X.shape[0] // 5)  # guard: ≥ 5 rows/component

        for k in _GMM_CANDIDATES:
            if k > n_max or k >= X.shape[0]:
                break
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=self.random_state,
                    max_iter=200,
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                logger.debug("DSI: GMM k=%d  BIC=%.2f", k, bic)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    best_k = k
            except Exception as exc:  # noqa: BLE001
                logger.debug("DSI: GMM k=%d failed: %s", k, exc)

        if best_gmm is None:
            # Fallback: single Gaussian
            best_gmm = GaussianMixture(
                n_components=1, random_state=self.random_state
            ).fit(X)
            best_k = 1

        logger.info("DSI: selected GMM n_components=%d (BIC=%.2f).", best_k, best_bic)
        return best_gmm, best_k

    def _encode_continuous(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Apply the fitted scaler (and PCA if applicable) to a DataFrame."""
        missing = [c for c in self._cont_cols if c not in df.columns]
        if missing:
            logger.error("DSI encoding: columns missing from synth_df: %s", missing)
            return None

        X = df[self._cont_cols].astype(float).values
        valid_rows = ~np.isnan(X).any(axis=1)
        if not valid_rows.any():
            return None

        X = X[valid_rows]
        X = self._scaler.transform(X)
        if self._pca is not None:
            X = self._pca.transform(X)
        return X

    def _calc_dsi(self, synth: pd.DataFrame) -> Tuple[float, float, float]:
        """Return (synth_ll, real_ll, gap)."""
        X_real = self._encode_continuous(self.real_df)
        X_synth = self._encode_continuous(synth)

        if X_real is None or X_synth is None:
            return _NAN, _NAN, _NAN

        real_ll = float(np.mean(self._gmm.score_samples(X_real)))
        synth_ll = float(np.mean(self._gmm.score_samples(X_synth)))
        gap = real_ll - synth_ll

        logger.info(
            "DSI: real_ll=%.4f  synth_ll=%.4f  gap=%.4f.", real_ll, synth_ll, gap
        )
        return synth_ll, real_ll, gap

    # ── ICVR ─────────────────────────────────────────────────────────────────

    def _calc_icvr(
        self, synth: pd.DataFrame
    ) -> Tuple[Dict[str, float], float]:
        """Compute per-FD and overall Inter-Column Violation Rate."""
        per_fd: Dict[str, float] = {}
        all_violation_masks: List[pd.Series] = []

        for det, dep in self.logic_spec.known_fds:
            key = f"{det}→{dep}"

            # Validate columns exist
            if det not in self.real_df.columns or dep not in self.real_df.columns:
                logger.warning("ICVR: columns '%s' or '%s' not in real_df; skipping FD.", det, dep)
                per_fd[key] = _NAN
                continue
            if det not in synth.columns or dep not in synth.columns:
                logger.warning("ICVR: columns '%s' or '%s' not in synth_df; skipping FD.", det, dep)
                per_fd[key] = _NAN
                continue

            # Validate that the FD holds in real_df
            fd_violations_in_real = (
                self.real_df.groupby(det, observed=True)[dep].nunique() > 1
            ).sum()
            if fd_violations_in_real > 0:
                real_violation_rate = fd_violations_in_real / self.real_df[det].nunique()
                if real_violation_rate > self.logic_spec.fd_violation_tolerance:
                    logger.warning(
                        "ICVR: FD '%s' is violated in real_df for %.1f%% of '%s' values "
                        "(tolerance=%.1f%%).  This is a 'soft' FD; results may be unreliable.",
                        key,
                        real_violation_rate * 100,
                        det,
                        self.logic_spec.fd_violation_tolerance * 100,
                    )

            # Build deterministic mapping from real data (use majority vote for soft FDs)
            mapping: dict = (
                self.real_df.groupby(det, observed=True)[dep]
                .agg(lambda s: s.mode().iloc[0])  # majority vote handles soft FDs
                .to_dict()
            )

            # Compute violations in synth: rows where det is known but dep differs
            det_col = synth[det].astype(str)
            dep_col = synth[dep].astype(str)
            known_mask = det_col.isin({str(k) for k in mapping})

            violation_mask = pd.Series(False, index=synth.index)
            if known_mask.any():
                expected = det_col[known_mask].map(
                    {str(k): str(v) for k, v in mapping.items()}
                )
                violation_mask[known_mask] = dep_col[known_mask] != expected

            rate = float(violation_mask.sum()) / len(synth) if len(synth) > 0 else _NAN
            per_fd[key] = rate
            all_violation_masks.append(violation_mask)
            logger.info("ICVR: FD '%s' violation rate = %.4f.", key, rate)

        if not all_violation_masks:
            return per_fd, _NAN

        # Overall ICVR: fraction of rows violating ANY registered FD
        combined = pd.concat(all_violation_masks, axis=1).any(axis=1)
        overall = float(combined.sum()) / len(synth)
        return per_fd, overall

    # ── HCS ──────────────────────────────────────────────────────────────────

    def _calc_hcs(
        self, synth: pd.DataFrame
    ) -> Tuple[Dict[str, float], float]:
        """Compute per-chain and overall Hierarchical Consistency violation rate."""
        per_chain: Dict[str, float] = {}
        all_violation_masks: List[pd.Series] = []

        for chain in self.logic_spec.hierarchies:
            key = "→".join(chain)

            # Validate all columns exist
            missing_cols = [c for c in chain if c not in self.real_df.columns]
            if missing_cols:
                logger.warning("HCS: chain '%s' — columns %s not in real_df; skipping.", key, missing_cols)
                per_chain[key] = _NAN
                continue
            if any(c not in synth.columns for c in chain):
                logger.warning("HCS: chain '%s' — columns missing in synth_df; skipping.", key)
                per_chain[key] = _NAN
                continue

            # Build valid co-occurrence sets for each consecutive pair in the chain
            # (ancestor, descendant) tuples observed in real data
            chain_violation_mask = pd.Series(False, index=synth.index)

            for i in range(len(chain) - 1):
                ancestor, descendant = chain[i], chain[i + 1]
                pair_key = f"{ancestor}→{descendant}"

                valid_pairs: set = set(
                    zip(
                        self.real_df[ancestor].astype(str),
                        self.real_df[descendant].astype(str),
                    )
                )

                synth_pairs = list(
                    zip(
                        synth[ancestor].astype(str),
                        synth[descendant].astype(str),
                    )
                )
                pair_violations = pd.Series(
                    [p not in valid_pairs for p in synth_pairs],
                    index=synth.index,
                )
                chain_violation_mask |= pair_violations
                logger.debug(
                    "HCS: pair '%s' — %d violations out of %d rows.",
                    pair_key,
                    pair_violations.sum(),
                    len(synth),
                )

            rate = float(chain_violation_mask.sum()) / len(synth) if len(synth) > 0 else _NAN
            per_chain[key] = rate
            all_violation_masks.append(chain_violation_mask)
            logger.info("HCS: chain '%s' violation rate = %.4f.", key, rate)

        if not all_violation_masks:
            return per_chain, _NAN

        combined = pd.concat(all_violation_masks, axis=1).any(axis=1)
        overall = float(combined.sum()) / len(synth)
        return per_chain, overall

    # ── MDI ──────────────────────────────────────────────────────────────────

    def _calc_mdi(
        self, synth: pd.DataFrame
    ) -> Tuple[Dict[str, float], float]:
        """Compute per-equation arithmetic violation rate with relative-error tolerance."""
        per_eq: Dict[str, float] = {}

        for equation in self.logic_spec.math_equations:
            col_a, op, col_b, col_c = equation
            key = f"{col_a} {op} {col_b} = {col_c}"

            # Validate columns
            missing = [c for c in (col_a, col_b, col_c) if c not in synth.columns]
            if missing:
                logger.warning("MDI: equation '%s' — columns %s not in synth_df; skipping.", key, missing)
                per_eq[key] = _NAN
                continue

            try:
                a = pd.to_numeric(synth[col_a], errors="coerce").values.astype(float)
                b = pd.to_numeric(synth[col_b], errors="coerce").values.astype(float)
                c = pd.to_numeric(synth[col_c], errors="coerce").values.astype(float)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MDI: equation '%s' — numeric conversion failed: %s", key, exc)
                per_eq[key] = _NAN
                continue

            lhs = self._apply_operator(a, op, b)
            if lhs is None:
                per_eq[key] = _NAN
                continue

            # Relative error: |lhs - c| / (|c| + 1e-9)
            rel_error = np.abs(lhs - c) / (np.abs(c) + 1e-9)

            # Exclude rows where any operand is NaN
            valid = ~(np.isnan(a) | np.isnan(b) | np.isnan(c) | np.isnan(lhs))
            if not valid.any():
                logger.warning("MDI: equation '%s' — all rows are NaN after coercion.", key)
                per_eq[key] = _NAN
                continue

            violation_rate = float((rel_error[valid] > self.mdi_epsilon).sum()) / valid.sum()
            per_eq[key] = violation_rate
            logger.info("MDI: equation '%s' violation rate = %.4f.", key, violation_rate)

        valid_rates = [v for v in per_eq.values() if not np.isnan(v)]
        mean_mdi = float(np.mean(valid_rates)) if valid_rates else _NAN
        return per_eq, mean_mdi

    @staticmethod
    def _apply_operator(a: np.ndarray, op: str, b: np.ndarray) -> Optional[np.ndarray]:
        """Apply arithmetic operator; returns None on unsupported op or division-by-zero."""
        if op == "+":
            return a + b
        elif op == "-":
            return a - b
        elif op == "*":
            return a * b
        elif op == "/":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = np.where(np.abs(b) < 1e-9, np.nan, a / b)
            return result
        else:
            logger.error("MDI: unsupported operator '%s'. Use '+', '-', '*', or '/'.", op)
            return None
