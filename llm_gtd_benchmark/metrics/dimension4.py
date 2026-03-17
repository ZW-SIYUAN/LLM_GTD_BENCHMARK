"""
Dimension 4 — Privacy & Anti-Memorization Evaluator.

Theoretical framing
-------------------
Privacy evaluation operates on two complementary axes:

    Black-box geometric (PrivacyEvaluator)
        Measures Distance to Closest Record (DCR) in a mixed feature space
        without any access to the generator model.  The heterogeneous L1
        metric is formally:

            dist(x, y) = Σ_k |x_k^cont − y_k^cont|   (MinMax-normalized)
                       + Σ_k 1(x_k^cat ≠ y_k^cat)     (0/1 indicator)

        which is equivalent to:
            Σ |x^cont − y^cont| + 0.5 · ||OneHot(x^cat) − OneHot(y^cat)||₁

        aligning with the GReaT and GraDe paper definitions.  The 5th
        percentile of the DCR distribution provides a conservative measure of
        the worst-case leakage; the exact-match rate flags near-verbatim copies.

    White-box parametric (MemorizationProbe)
        Probes the generator model's parameters for evidence of memorization
        via Masked Conditional Perplexity (Masked PPL), inspired by the
        HARMONIC framework.

        The key insight is that naively computing PPL on serialized table rows
        dilutes the signal: the model predicts template tokens (column names,
        "is", ",") with near-zero loss, masking its uncertainty about actual
        data values.  By setting template token labels to −100 (PyTorch's
        ignore index), only value tokens contribute to the cross-entropy loss,
        yielding a precise memorization probe.

        DLT gap = Masked_PPL(D_test) − Masked_PPL(D_train)

        A large positive gap indicates the generator predicts training values
        significantly better than held-out test values — a hallmark of
        parameter-level over-fitting and memorization.

Engineering guarantees
----------------------
- Strategy routing is automatic: FAISS IndexFlat(METRIC_L1) (fast) is used when
  all categorical cardinalities ≤ threshold AND faiss is installed; otherwise
  double-chunked numpy broadcasting (memory-safe, no extra dependencies).
- Both strategies implement the same mixed L1 metric — results are equivalent.
- The chunked numpy strategy never materialises an N×M distance matrix; peak
  memory is O(chunk_size² × n_features).
- MinMaxScaler is fitted on real training data only (the reference domain).
- Token offset mapping (return_offsets_mapping=True) ensures precise character-
  to-token span alignment for the masking mechanism.
- DataCopyingWarning is emitted when exact_match_rate exceeds the configured
  threshold, providing a hard privacy guardrail.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from llm_gtd_benchmark.core.schema import DataSchema

logger = logging.getLogger(__name__)

_NAN = float("nan")
_EXACT_MATCH_EPS: float = 1e-5
_HIGH_CARDINALITY_THRESHOLD: int = 200


# ---------------------------------------------------------------------------
# Custom warning
# ---------------------------------------------------------------------------


class DataCopyingWarning(UserWarning):
    """Emitted when the synthetic data exact-match rate exceeds the configured
    threshold, indicating potential near-verbatim copying of training records."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Dim4Result:
    """Output of :class:`PrivacyEvaluator` and optionally :class:`MemorizationProbe`.

    Attributes
    ----------
    dcr_5th_percentile:
        5th percentile of the Distance to Closest Record (DCR) distribution.
        Higher values mean synthetic points are farther from training records
        (safer).  Computed under the mixed L1 heterogeneous metric.
    exact_match_rate:
        Fraction of synthetic rows with DCR < ``exact_match_threshold``.
        Values exceeding 1 % suggest potential verbatim copying of training data.
    distance_strategy:
        Internal strategy used: ``"chunked_numpy"`` or ``"faiss_l1"`` (exact L1).
    dlt_masked_ppl_train:
        Masked Conditional PPL of the generator model evaluated on the real
        training set.  Lower PPL on training vs. test signals memorization.
    dlt_masked_ppl_test:
        Masked Conditional PPL evaluated on the real held-out test set.
    dlt_gap:
        ``PPL(test) − PPL(train)``.  A large positive value indicates the
        generator predicts training data values far better than unseen test
        values — evidence of parameter-level memorization.
    """

    # Black-box
    dcr_5th_percentile: float = _NAN
    exact_match_rate: float = _NAN
    distance_strategy: str = ""

    # White-box (optional)
    dlt_masked_ppl_train: float = _NAN
    dlt_masked_ppl_test: float = _NAN
    dlt_gap: float = _NAN

    # Bootstrap CIs (populated only when n_boot > 0 is passed to evaluate())
    dcr_5th_ci: Optional[Tuple[float, float]] = None
    exact_match_rate_ci: Optional[Tuple[float, float]] = None

    @property
    def summary(self) -> str:
        def _fmt(v: float) -> str:
            return f"{v:.4f}" if not np.isnan(v) else "  N/A "

        lines = [
            "── Dimension 4: Privacy & Anti-Memorization ──────────────────",
            "  Black-box distance metrics (DCR):",
            f"    DCR 5th percentile  (↑ safer) : {_fmt(self.dcr_5th_percentile)}",
            f"    Exact match rate    (↓ safer) : {_fmt(self.exact_match_rate)}",
            f"    Distance strategy             : {self.distance_strategy or 'N/A'}",
        ]
        if not np.isnan(self.dlt_gap):
            lines += [
                "  White-box memorization (Masked DLT):",
                f"    Masked PPL — train  (↑ safer) : {_fmt(self.dlt_masked_ppl_train)}",
                f"    Masked PPL — test             : {_fmt(self.dlt_masked_ppl_test)}",
                f"    DLT gap PPL(test)−PPL(train)  : {_fmt(self.dlt_gap)}",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Privacy Evaluator — black-box DCR
# ---------------------------------------------------------------------------


class PrivacyEvaluator:
    """Dimension 4 black-box privacy evaluator via Distance to Closest Record (DCR).

    Computes the mixed L1 distance between each synthetic row and its nearest
    real training record:

        dist(x, y) = Σ |x_cont − y_cont|  (MinMax-normalized, [0, 1] per feature)
                   + Σ 1(x_cat ≠ y_cat)   (0/1 Hamming on categoricals)

    **Strategy routing (automatic):**

    - *FAISS IndexFlat(METRIC_L1)* — used when all categorical columns have
      cardinality ≤ ``high_cardinality_threshold`` **and** ``faiss`` is installed.
      Categorical features are one-hot encoded and scaled by 0.5 so that each
      mismatch contributes exactly 1.0 to the L1 distance (preserving the
      Hamming equivalence). Fastest path, recommended for typical tabular data.

    - *Chunked numpy broadcasting* — fallback for high-cardinality categoricals
      (e.g., ZIP codes, free-text IDs) or when faiss is unavailable. Computes
      distances over double-chunked tiles; peak memory is
      O(chunk_size² × n_features), never materialising the full N×M matrix.

    Both strategies implement the same mixed L1 metric; results are equivalent.

    Parameters
    ----------
    schema:
        :class:`~llm_gtd_benchmark.core.schema.DataSchema` built from real data.
    real_train_df:
        Reference real training DataFrame (the DCR reference corpus).
    exact_match_threshold:
        DCR below which a synthetic row is flagged as a near-exact copy.
        Default: 1e-5 (effectively zero distance in normalized space).
    exact_match_warn_rate:
        If ``exact_match_rate`` exceeds this fraction, emit
        :class:`DataCopyingWarning`.  Default: 0.01 (1 %).
    chunk_size:
        Rows per tile in the chunked numpy strategy.  Default: 512.
    high_cardinality_threshold:
        Maximum category count per column for the FAISS strategy.  Default: 200.
    random_state:
        Reserved for future subsampling; currently unused.

    Examples
    --------
    >>> evaluator = PrivacyEvaluator(schema, real_train_df)
    >>> result4 = evaluator.evaluate(synth_df)
    >>> print(result4.summary)
    """

    def __init__(
        self,
        schema: DataSchema,
        real_train_df: pd.DataFrame,
        exact_match_threshold: float = _EXACT_MATCH_EPS,
        exact_match_warn_rate: float = 0.01,
        chunk_size: int = 512,
        high_cardinality_threshold: int = _HIGH_CARDINALITY_THRESHOLD,
        random_state: int = 42,
    ) -> None:
        self.schema = schema
        self.real_train_df = real_train_df.reset_index(drop=True)
        self.exact_match_threshold = exact_match_threshold
        self.exact_match_warn_rate = exact_match_warn_rate
        self.chunk_size = chunk_size
        self.high_cardinality_threshold = high_cardinality_threshold
        self.random_state = random_state

        self._cont_cols = [c.name for c in schema.continuous_columns]
        self._cat_cols = [c.name for c in schema.categorical_columns]

        self._scaler: Optional[MinMaxScaler] = None
        self._ohe: Optional[OneHotEncoder] = None
        self._strategy: str = ""
        self._max_cat_cardinality: int = 0

        self._fit()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        synth_df: pd.DataFrame,
        n_boot: int = 0,
        boot_ci: float = 0.95,
    ) -> Dim4Result:
        """Compute DCR-based privacy metrics for the synthetic dataset.

        Parameters
        ----------
        synth_df:
            Synthetic DataFrame to evaluate (Dim0-cleaned recommended).
        n_boot:
            Number of bootstrap replicates for sampling-uncertainty CIs.
            ``0`` (default) disables bootstrapping.  Recommended: 1000.
        boot_ci:
            Confidence level for bootstrap CIs.  Default: 0.95.

        Returns
        -------
        :class:`Dim4Result` with black-box fields populated.
        """
        synth_df = synth_df.reset_index(drop=True)

        logger.info(
            "Dim4: computing DCR (%s strategy, %d synth × %d real)...",
            self._strategy, len(synth_df), len(self.real_train_df),
        )

        min_dists = self._compute_dcr(synth_df, self.real_train_df)

        dcr_5th = float(np.percentile(min_dists, 5))
        exact_rate = float(np.mean(min_dists < self.exact_match_threshold))

        if exact_rate > self.exact_match_warn_rate:
            warnings.warn(
                f"Dim4: exact_match_rate={exact_rate:.2%} exceeds warning threshold "
                f"{self.exact_match_warn_rate:.2%}. The synthetic data may contain "
                "near-exact copies of real training records. "
                "Investigate before sharing or publishing.",
                DataCopyingWarning,
                stacklevel=2,
            )
            logger.warning(
                "Dim4: DataCopyingWarning — exact_match_rate=%.4f", exact_rate
            )

        logger.info(
            "Dim4: DCR_5th=%.6f, exact_match_rate=%.4f", dcr_5th, exact_rate
        )

        result = Dim4Result(
            dcr_5th_percentile=dcr_5th,
            exact_match_rate=exact_rate,
            distance_strategy=self._strategy,
        )

        # ── Bootstrap CIs (opt-in) ────────────────────────────────────────────
        if n_boot > 0:
            logger.info("Dim4: bootstrapping CIs (n_boot=%d, ci=%.2f)...", n_boot, boot_ci)
            from llm_gtd_benchmark.utils.bootstrap import (
                bootstrap_percentile_ci,
                bootstrap_proportion_ci,
            )

            rng = np.random.default_rng(self.random_state)

            # DCR 5th percentile CI — resample per-synthetic-row DCR distances
            result.dcr_5th_ci = bootstrap_percentile_ci(min_dists, 5.0, n_boot, boot_ci, rng)

            # Exact match rate CI — resample boolean indicators
            exact_bool = (min_dists < self.exact_match_threshold).astype(np.float64)
            result.exact_match_rate_ci = bootstrap_proportion_ci(
                exact_bool, n_boot, boot_ci, rng
            )

        return result

    # ── Initialisation ────────────────────────────────────────────────────────

    def _fit(self) -> None:
        """Fit preprocessors and select DCR computation strategy."""
        if self._cont_cols:
            self._scaler = MinMaxScaler()
            self._scaler.fit(
                self.real_train_df[self._cont_cols].fillna(0).astype(float).values
            )

        if self._cat_cols:
            cardinalities = [self.real_train_df[c].nunique() for c in self._cat_cols]
            self._max_cat_cardinality = max(cardinalities)
        else:
            self._max_cat_cardinality = 0

        # Route strategy: FAISS when cardinality is low and faiss is installed.
        if self._max_cat_cardinality <= self.high_cardinality_threshold:
            try:
                import faiss  # noqa: F401

                self._strategy = "faiss_l1"
                if self._cat_cols:
                    self._ohe = OneHotEncoder(
                        sparse_output=False,
                        handle_unknown="ignore",
                    )
                    self._ohe.fit(
                        self.real_train_df[self._cat_cols]
                        .fillna("__NA__")
                        .astype(str)
                    )
                logger.info("Dim4: strategy selected → faiss_l1.")
            except ImportError:
                self._strategy = "chunked_numpy"
                logger.info(
                    "Dim4: faiss not installed; strategy selected → chunked_numpy."
                )
        else:
            self._strategy = "chunked_numpy"
            logger.info(
                "Dim4: max categorical cardinality=%d > threshold=%d; "
                "strategy selected → chunked_numpy.",
                self._max_cat_cardinality,
                self.high_cardinality_threshold,
            )

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_features(
        self,
        df: pd.DataFrame,
        for_faiss: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Encode features for distance computation.

        Returns
        -------
        num_arr:
            MinMax-scaled continuous features, shape (n, n_cont).
            ``None`` when no continuous columns exist.
        cat_arr:
            - Chunked strategy: raw string object array, shape (n, n_cat).
            - FAISS strategy: one-hot array scaled by 0.5, shape (n, ohe_dim).
            ``None`` when no categorical columns exist.
        """
        num_arr = None
        cat_arr = None

        if self._cont_cols and self._scaler is not None:
            raw = df[self._cont_cols].fillna(0).astype(float).values
            num_arr = self._scaler.transform(raw).astype(np.float32)

        if self._cat_cols:
            if for_faiss and self._ohe is not None:
                raw_cat = df[self._cat_cols].fillna("__NA__").astype(str)
                # Scale OHE columns by 0.5: ||OHE(a) - OHE(b)||_L1 = 2 on mismatch;
                # × 0.5 → 1.0, matching the 0/1 Hamming indicator exactly.
                cat_arr = (self._ohe.transform(raw_cat) * 0.5).astype(np.float32)
            else:
                cat_arr = df[self._cat_cols].fillna("__NA__").astype(str).values

        return num_arr, cat_arr

    # ── DCR dispatch ──────────────────────────────────────────────────────────

    def _compute_dcr(
        self, synth_df: pd.DataFrame, real_df: pd.DataFrame
    ) -> np.ndarray:
        if self._strategy == "faiss_l1":
            return self._dcr_faiss(synth_df, real_df)
        return self._dcr_chunked(synth_df, real_df)

    def _dcr_chunked(
        self, synth_df: pd.DataFrame, real_df: pd.DataFrame
    ) -> np.ndarray:
        """Double-chunked numpy broadcasting.

        Processes (chunk_size × chunk_size) tiles at a time.  Memory cost is
        O(chunk_size² × n_features) — safe for large datasets.

        Continuous distance: Σ |x_k^cont − y_k^cont|  (L1, MinMax-normalized)
        Categorical distance: Σ 1(x_k^cat ≠ y_k^cat)  (Hamming)
        """
        s_num, s_cat = self._encode_features(synth_df, for_faiss=False)
        r_num, r_cat = self._encode_features(real_df, for_faiss=False)

        n_synth = len(synth_df)
        n_real = len(real_df)
        cs = self.chunk_size
        min_dists = np.full(n_synth, np.inf, dtype=np.float64)

        for s_start in range(0, n_synth, cs):
            s_end = min(s_start + cs, n_synth)
            chunk_min = np.full(s_end - s_start, np.inf, dtype=np.float64)

            for r_start in range(0, n_real, cs):
                r_end = min(r_start + cs, n_real)

                # (s_chunk, r_chunk) distance tile
                dist = np.zeros(
                    (s_end - s_start, r_end - r_start), dtype=np.float64
                )

                if s_num is not None and r_num is not None:
                    sn = s_num[s_start:s_end].astype(np.float64)  # (cs_s, n_cont)
                    rn = r_num[r_start:r_end].astype(np.float64)  # (cs_r, n_cont)
                    # (cs_s, cs_r): L1 sum over feature axis
                    dist += np.abs(sn[:, None, :] - rn[None, :, :]).sum(axis=2)

                if s_cat is not None and r_cat is not None:
                    sc = s_cat[s_start:s_end]  # (cs_s, n_cat) object array
                    rc = r_cat[r_start:r_end]  # (cs_r, n_cat) object array
                    # (cs_s, cs_r): Hamming sum over category axis
                    dist += (sc[:, None, :] != rc[None, :, :]).sum(axis=2)

                chunk_min = np.minimum(chunk_min, dist.min(axis=1))

            min_dists[s_start:s_end] = chunk_min

        return min_dists

    def _dcr_faiss(
        self, synth_df: pd.DataFrame, real_df: pd.DataFrame
    ) -> np.ndarray:
        """FAISS IndexFlat(METRIC_L1) strategy.

        Concatenates MinMax-scaled continuous features with 0.5-weighted OHE
        categorical features into a single dense vector, then uses exact L1
        nearest-neighbour search.
        """
        import faiss

        s_num, s_cat_ohe = self._encode_features(synth_df, for_faiss=True)
        r_num, r_cat_ohe = self._encode_features(real_df, for_faiss=True)

        def _concat(*arrays: Optional[np.ndarray]) -> np.ndarray:
            valid = [a for a in arrays if a is not None]
            if not valid:
                raise ValueError("Dim4 FAISS: no features available for encoding.")
            return np.ascontiguousarray(
                np.concatenate(valid, axis=1), dtype=np.float32
            )

        real_vecs = _concat(r_num, r_cat_ohe)
        synth_vecs = _concat(s_num, s_cat_ohe)

        dim = real_vecs.shape[1]
        index = faiss.IndexFlat(dim, faiss.METRIC_L1)
        index.add(real_vecs)

        distances, _ = index.search(synth_vecs, k=1)
        return distances[:, 0].astype(np.float64)


# ---------------------------------------------------------------------------
# Memorization Probe — white-box Masked DLT
# ---------------------------------------------------------------------------


class MemorizationProbe:
    """White-box memorization probe via Masked Conditional Perplexity (DLT).

    Receives the user's **generator** LLM (the model that produced the synthetic
    data) and probes it for evidence of training-set memorization by computing
    Masked Conditional PPL on real training vs. real test rows.

    **Masking mechanism:**
    Each table row is serialized as::

        ColumnA is ValueA, ColumnB is ValueB, ...

    The tokenizer's character offset mapping is used to identify which tokens
    correspond to actual data values vs. structural template tokens
    (column names, "is", ",").  Template token labels are set to −100
    (PyTorch's cross-entropy ignore index), so only value tokens contribute
    to the computed loss — eliminating template-noise dilution.

    **DLT gap:**
        DLT = Masked_PPL(D_test) − Masked_PPL(D_train)

    A large positive gap means the generator predicts training values far more
    confidently than held-out test values, indicating parameter-level over-fitting.

    Parameters
    ----------
    model:
        HuggingFace ``CausalLM`` model — the generator whose memorization is
        probed.  Must have been loaded with ``device_map`` pointing to an
        accessible device, or manually moved with ``.to(device)``.
    tokenizer:
        Corresponding HuggingFace tokenizer.  Must support
        ``return_offsets_mapping=True`` (most fast tokenizers do).
    max_seq_length:
        Maximum tokenised sequence length per row.  Default: 256.
    batch_size:
        Rows per forward-pass batch.  Default: 8.
    device:
        ``"cuda"`` or ``"cpu"``.  Auto-detected when ``None``.

    Examples
    --------
    >>> result4 = PrivacyEvaluator(schema, real_train_df).evaluate(synth_df)
    >>> probe   = MemorizationProbe(generator_model, generator_tokenizer)
    >>> result4 = probe.evaluate(real_train_df, real_test_df, result4)
    >>> print(result4.summary)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_seq_length: int = 256,
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        if device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        real_train_df: pd.DataFrame,
        real_test_df: pd.DataFrame,
        result: Optional[Dim4Result] = None,
    ) -> Dim4Result:
        """Compute Masked DLT scores and attach them to a Dim4Result.

        Parameters
        ----------
        real_train_df:
            Real training DataFrame.
        real_test_df:
            Real held-out test DataFrame.
        result:
            Existing :class:`Dim4Result` to update in-place.  If ``None``,
            a new empty result is created (black-box fields will be NaN).

        Returns
        -------
        :class:`Dim4Result` with white-box fields populated.
        """
        if result is None:
            result = Dim4Result()

        logger.info(
            "Dim4 DLT: computing Masked PPL on training set (%d rows)...",
            len(real_train_df),
        )
        ppl_train = self._compute_masked_ppl(real_train_df)

        logger.info(
            "Dim4 DLT: computing Masked PPL on test set (%d rows)...",
            len(real_test_df),
        )
        ppl_test = self._compute_masked_ppl(real_test_df)

        dlt_gap = (
            float(ppl_test - ppl_train)
            if not (np.isnan(ppl_train) or np.isnan(ppl_test))
            else _NAN
        )

        result.dlt_masked_ppl_train = ppl_train
        result.dlt_masked_ppl_test = ppl_test
        result.dlt_gap = dlt_gap

        logger.info(
            "Dim4 DLT: PPL_train=%.4f, PPL_test=%.4f, gap=%.4f",
            ppl_train,
            ppl_test,
            dlt_gap,
        )
        return result

    # ── Serialization ─────────────────────────────────────────────────────────

    def _serialize_with_spans(
        self, row: pd.Series
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """Serialize a DataFrame row to text with value character spans.

        Format: ``ColumnA is ValueA, ColumnB is ValueB``

        Returns
        -------
        text:
            Full serialized string.
        value_spans:
            List of (char_start, char_end) for each value substring.
            Used to build the token mask.
        """
        segments: List[str] = []
        value_spans: List[Tuple[int, int]] = []
        pos = 0

        for i, (col, val) in enumerate(row.items()):
            separator = ", " if i > 0 else ""
            prefix = f"{col} is "
            val_str = str(val) if pd.notna(val) else "NA"

            full_segment = separator + prefix + val_str
            val_start = pos + len(separator) + len(prefix)
            val_end = val_start + len(val_str)
            value_spans.append((val_start, val_end))

            segments.append(full_segment)
            pos += len(full_segment)

        return "".join(segments), value_spans

    # ── Masking ───────────────────────────────────────────────────────────────

    def _build_masked_labels(
        self,
        text: str,
        value_spans: List[Tuple[int, int]],
    ) -> Tuple[Any, Any]:
        """Tokenize and build a label tensor with template tokens masked.

        Uses the tokenizer's character offset mapping to align character spans
        to token boundaries precisely.  A token is treated as a value token if
        its character span is fully contained within a value span; all other
        tokens receive label −100 (ignored by cross-entropy loss).

        Returns
        -------
        input_ids:
            Token IDs tensor, shape (seq_len,).
        labels:
            Label tensor, shape (seq_len,).  Value positions hold the token ID;
            template positions hold −100.
        """
        import torch

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"][0]
        offset_mapping = encoding["offset_mapping"][0].tolist()

        labels = torch.full_like(input_ids, -100)

        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                continue  # special token (BOS / EOS / PAD)
            for val_start, val_end in value_spans:
                if tok_start >= val_start and tok_end <= val_end:
                    labels[tok_idx] = input_ids[tok_idx]
                    break

        return input_ids, labels

    # ── PPL computation ───────────────────────────────────────────────────────

    def _compute_masked_ppl(self, df: pd.DataFrame) -> float:
        """Compute Masked Conditional PPL over all rows in ``df``.

        For each batch:
        1. Serialize rows and build masked label tensors.
        2. Pad to batch max length.
        3. Forward-pass through the model (no gradient).
        4. Compute per-token cross-entropy, ignoring −100 labels.
        5. Accumulate loss and token counts.

        Final PPL = exp(total_loss / total_active_tokens).
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        total_loss_sum = 0.0
        total_active_tokens = 0

        with torch.no_grad():
            for batch_start in range(0, len(df), self.batch_size):
                batch = df.iloc[batch_start: batch_start + self.batch_size]

                batch_input_ids: List[Any] = []
                batch_labels: List[Any] = []

                for _, row in batch.iterrows():
                    text, spans = self._serialize_with_spans(row)
                    ids, labs = self._build_masked_labels(text, spans)
                    batch_input_ids.append(ids)
                    batch_labels.append(labs)

                if not batch_input_ids:
                    continue

                # Pad sequences within the batch to uniform length.
                max_len = max(x.size(0) for x in batch_input_ids)
                B = len(batch_input_ids)
                pad_id = self.tokenizer.pad_token_id or 0

                padded_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
                padded_labels = torch.full((B, max_len), -100, dtype=torch.long)
                attn_mask = torch.zeros(B, max_len, dtype=torch.long)

                for i, (ids, labs) in enumerate(zip(batch_input_ids, batch_labels)):
                    L = ids.size(0)
                    padded_ids[i, :L] = ids
                    padded_labels[i, :L] = labs
                    attn_mask[i, :L] = 1

                try:
                    outputs = self.model(
                        input_ids=padded_ids.to(self.device),
                        attention_mask=attn_mask.to(self.device),
                    )
                    # HuggingFace CausalLM: logits shape (B, T, V).
                    # Shift: position t predicts token t+1.
                    shift_logits = outputs.logits[:, :-1, :].contiguous()
                    shift_labels = padded_labels[:, 1:].to(self.device).contiguous()

                    # Per-token CE loss; −100 positions are automatically ignored.
                    per_token_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                        ignore_index=-100,
                    )
                    active_mask = shift_labels.view(-1) != -100
                    total_loss_sum += per_token_loss[active_mask].sum().item()
                    total_active_tokens += active_mask.sum().item()

                except Exception as exc:
                    logger.error("Dim4 DLT batch %d failed: %s", batch_start, exc)

        if total_active_tokens == 0:
            logger.warning(
                "Dim4 DLT: no active value tokens found across %d rows; "
                "returning NaN PPL.  Check tokenizer offset_mapping support.",
                len(df),
            )
            return _NAN

        mean_loss = total_loss_sum / total_active_tokens
        return float(np.exp(mean_loss))
