"""
ResultBundle — container and serializer for multi-dimension benchmark results.

Serialisation contract
----------------------
- Format: single UTF-8 JSON file.
- NaN floats ↔ JSON ``null`` (round-trip safe via custom helpers).
- ``Dim0Result.clean_df`` is intentionally excluded from serialisation
  (it can be hundreds of MB).  On load, ``result0.clean_df`` is an empty
  DataFrame.  Re-run :class:`~llm_gtd_benchmark.metrics.dimension0.StructuralInterceptor`
  to repopulate it.
- Bootstrap CI tuples ``(lo, hi)`` ↔ JSON ``[lo, hi]`` or ``null``.
- ``format_version`` is stored and checked on load; a version mismatch
  emits a warning but does **not** raise — forward compatibility via
  "unknown fields are silently ignored" policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from llm_gtd_benchmark.core.schema import DataSchema
from llm_gtd_benchmark.metrics.dimension0 import Dim0Result
from llm_gtd_benchmark.metrics.dimension1 import Dim1Result
from llm_gtd_benchmark.metrics.dimension2 import Dim2Result
from llm_gtd_benchmark.metrics.dimension3 import Dim3Result
from llm_gtd_benchmark.metrics.dimension4 import Dim4Result
from llm_gtd_benchmark.metrics.dimension5 import Dim5Result

logger = logging.getLogger(__name__)

_FORMAT_VERSION = "1.0"
_NAN = float("nan")


# ---------------------------------------------------------------------------
# JSON NaN ↔ null helpers
# ---------------------------------------------------------------------------


def _nan_to_null(v: Any) -> Any:
    """Recursively replace NaN floats with ``None`` (JSON ``null``)."""
    if isinstance(v, float):
        return None if (v != v) else v  # NaN check without importing math
    if isinstance(v, dict):
        return {k: _nan_to_null(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_nan_to_null(item) for item in v]
    return v


def _null_to_nan(v: Any) -> Any:
    """Recursively replace ``None`` (JSON ``null``) with NaN in numeric contexts."""
    if v is None:
        return _NAN
    if isinstance(v, dict):
        return {k: _null_to_nan(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_null_to_nan(item) for item in v]
    return v


def _ci_to_list(ci: Optional[Tuple[float, float]]) -> Optional[List]:
    if ci is None:
        return None
    return [_nan_to_null(ci[0]), _nan_to_null(ci[1])]


def _list_to_ci(lst: Optional[List]) -> Optional[Tuple[float, float]]:
    if lst is None:
        return None
    lo = _null_to_nan(lst[0]) if len(lst) > 0 else _NAN
    hi = _null_to_nan(lst[1]) if len(lst) > 1 else _NAN
    return (lo, hi)


def _dict_ci_to_list(
    d: Optional[Dict[str, Tuple[float, float]]],
) -> Optional[Dict[str, List]]:
    if d is None:
        return None
    return {k: _ci_to_list(v) for k, v in d.items()}


def _dict_list_to_ci(
    d: Optional[Dict[str, List]],
) -> Optional[Dict[str, Tuple[float, float]]]:
    if d is None:
        return None
    return {k: _list_to_ci(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Schema fingerprint
# ---------------------------------------------------------------------------


def schema_fingerprint(schema: DataSchema) -> str:
    """Stable MD5 fingerprint derived from column names + types (sorted).

    Used to detect schema drift between when a bundle was created and when
    it is loaded in a new experiment.
    """
    cols_repr = sorted(f"{c.name}:{c.col_type}" for c in schema.columns)
    return hashlib.md5("|".join(cols_repr).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Per-dimension serialisers
# ---------------------------------------------------------------------------


def _dim0_to_dict(r: Dim0Result) -> Dict:
    return {
        "irr": _nan_to_null(r.irr),
        "n_total": r.n_total,
        "n_invalid": r.n_invalid,
        "n_clean": r.n_clean,
        "defect_counts": dict(r.defect_counts),
        # clean_df intentionally excluded — too large; recompute with StructuralInterceptor
    }


def _dim0_from_dict(d: Dict) -> Dim0Result:
    return Dim0Result(
        irr=_null_to_nan(d.get("irr")),
        n_total=d.get("n_total", 0),
        n_invalid=d.get("n_invalid", 0),
        n_clean=d.get("n_clean", 0),
        defect_counts=d.get("defect_counts", {}),
        clean_df=pd.DataFrame(),  # not serialised
    )


def _dim1_to_dict(r: Dim1Result) -> Dict:
    return {
        "ks_per_column": _nan_to_null(dict(r.ks_per_column)),
        "tvd_per_column": _nan_to_null(dict(r.tvd_per_column)),
        "mean_ks": _nan_to_null(r.mean_ks),
        "mean_tvd": _nan_to_null(r.mean_tvd),
        "pearson_matrix_error": _nan_to_null(r.pearson_matrix_error),
        "cramerv_matrix_error": _nan_to_null(r.cramerv_matrix_error),
        "alpha_precision": _nan_to_null(r.alpha_precision),
        "beta_recall": _nan_to_null(r.beta_recall),
        "c2st_auc_xgb": _nan_to_null(r.c2st_auc_xgb),
        "c2st_auc_rf": _nan_to_null(r.c2st_auc_rf),
        "c2st_auc_mean": _nan_to_null(r.c2st_auc_mean),
        "skipped_columns": dict(r.skipped_columns),
        "mean_ks_ci": _ci_to_list(getattr(r, "mean_ks_ci", None)),
        "mean_tvd_ci": _ci_to_list(getattr(r, "mean_tvd_ci", None)),
        "alpha_precision_ci": _ci_to_list(getattr(r, "alpha_precision_ci", None)),
        "beta_recall_ci": _ci_to_list(getattr(r, "beta_recall_ci", None)),
    }


def _dim1_from_dict(d: Dict) -> Dim1Result:
    r = Dim1Result(
        ks_per_column=_null_to_nan(d.get("ks_per_column", {})),
        tvd_per_column=_null_to_nan(d.get("tvd_per_column", {})),
        mean_ks=_null_to_nan(d.get("mean_ks")),
        mean_tvd=_null_to_nan(d.get("mean_tvd")),
        pearson_matrix_error=_null_to_nan(d.get("pearson_matrix_error")),
        cramerv_matrix_error=_null_to_nan(d.get("cramerv_matrix_error")),
        alpha_precision=_null_to_nan(d.get("alpha_precision")),
        beta_recall=_null_to_nan(d.get("beta_recall")),
        c2st_auc_xgb=_null_to_nan(d.get("c2st_auc_xgb")),
        c2st_auc_rf=_null_to_nan(d.get("c2st_auc_rf")),
        c2st_auc_mean=_null_to_nan(d.get("c2st_auc_mean")),
        skipped_columns=d.get("skipped_columns", {}),
    )
    r.mean_ks_ci = _list_to_ci(d.get("mean_ks_ci"))
    r.mean_tvd_ci = _list_to_ci(d.get("mean_tvd_ci"))
    r.alpha_precision_ci = _list_to_ci(d.get("alpha_precision_ci"))
    r.beta_recall_ci = _list_to_ci(d.get("beta_recall_ci"))
    return r


def _dim2_to_dict(r: Dim2Result) -> Dict:
    return {
        "dsi_synth_ll": _nan_to_null(r.dsi_synth_ll),
        "dsi_real_ll": _nan_to_null(r.dsi_real_ll),
        "dsi_gap": _nan_to_null(r.dsi_gap),
        "gmm_n_components": r.gmm_n_components,
        "icvr": _nan_to_null(r.icvr),
        "icvr_per_fd": _nan_to_null(dict(r.icvr_per_fd)),
        "hcs_violation_rate": _nan_to_null(r.hcs_violation_rate),
        "hcs_per_chain": _nan_to_null(dict(r.hcs_per_chain)),
        "mdi_per_equation": _nan_to_null(dict(r.mdi_per_equation)),
        "mdi_mean": _nan_to_null(r.mdi_mean),
    }


def _dim2_from_dict(d: Dict) -> Dim2Result:
    return Dim2Result(
        dsi_synth_ll=_null_to_nan(d.get("dsi_synth_ll")),
        dsi_real_ll=_null_to_nan(d.get("dsi_real_ll")),
        dsi_gap=_null_to_nan(d.get("dsi_gap")),
        gmm_n_components=d.get("gmm_n_components", 0),
        icvr=_null_to_nan(d.get("icvr")),
        icvr_per_fd=_null_to_nan(d.get("icvr_per_fd", {})),
        hcs_violation_rate=_null_to_nan(d.get("hcs_violation_rate")),
        hcs_per_chain=_null_to_nan(d.get("hcs_per_chain", {})),
        mdi_per_equation=_null_to_nan(d.get("mdi_per_equation", {})),
        mdi_mean=_null_to_nan(d.get("mdi_mean")),
    )


def _dim3_to_dict(r: Dim3Result) -> Dict:
    return {
        "task_type": r.task_type,
        "target_col": r.target_col,
        "tuning_backend": r.tuning_backend,
        "tuning_mode": getattr(r, "tuning_mode", "shared"),
        "mle_tstr": _nan_to_null(dict(r.mle_tstr)),
        "mle_trtr": _nan_to_null(dict(r.mle_trtr)),
        "utility_gap": _nan_to_null(dict(r.utility_gap)),
        "lle_tstr": _nan_to_null(dict(r.lle_tstr)),
        "lle_model": r.lle_model,
    }


def _dim3_from_dict(d: Dict) -> Dim3Result:
    return Dim3Result(
        task_type=d.get("task_type", ""),
        target_col=d.get("target_col", ""),
        tuning_backend=d.get("tuning_backend", ""),
        tuning_mode=d.get("tuning_mode", "shared"),
        mle_tstr=_null_to_nan(d.get("mle_tstr", {})),
        mle_trtr=_null_to_nan(d.get("mle_trtr", {})),
        utility_gap=_null_to_nan(d.get("utility_gap", {})),
        lle_tstr=_null_to_nan(d.get("lle_tstr", {})),
        lle_model=d.get("lle_model", ""),
    )


def _dim4_to_dict(r: Dim4Result) -> Dict:
    return {
        "dcr_5th_percentile": _nan_to_null(r.dcr_5th_percentile),
        "exact_match_rate": _nan_to_null(r.exact_match_rate),
        "distance_strategy": r.distance_strategy,
        "dlt_masked_ppl_train": _nan_to_null(r.dlt_masked_ppl_train),
        "dlt_masked_ppl_test": _nan_to_null(r.dlt_masked_ppl_test),
        "dlt_gap": _nan_to_null(r.dlt_gap),
        "dcr_5th_ci": _ci_to_list(getattr(r, "dcr_5th_ci", None)),
        "exact_match_rate_ci": _ci_to_list(getattr(r, "exact_match_rate_ci", None)),
    }


def _dim4_from_dict(d: Dict) -> Dim4Result:
    r = Dim4Result(
        dcr_5th_percentile=_null_to_nan(d.get("dcr_5th_percentile")),
        exact_match_rate=_null_to_nan(d.get("exact_match_rate")),
        distance_strategy=d.get("distance_strategy", ""),
        dlt_masked_ppl_train=_null_to_nan(d.get("dlt_masked_ppl_train")),
        dlt_masked_ppl_test=_null_to_nan(d.get("dlt_masked_ppl_test")),
        dlt_gap=_null_to_nan(d.get("dlt_gap")),
    )
    r.dcr_5th_ci = _list_to_ci(d.get("dcr_5th_ci"))
    r.exact_match_rate_ci = _list_to_ci(d.get("exact_match_rate_ci"))
    return r


def _dim5_to_dict(r: Dim5Result) -> Dict:
    return {
        "protected_cols": list(r.protected_cols),
        "target_col": r.target_col,
        "task_type": r.task_type,
        "bias_nmi_real": _nan_to_null(dict(r.bias_nmi_real)),
        "bias_nmi_synth": _nan_to_null(dict(r.bias_nmi_synth)),
        "delta_dp": _nan_to_null(dict(r.delta_dp)) if r.delta_dp is not None else None,
        "delta_eop": _nan_to_null(dict(r.delta_eop)) if r.delta_eop is not None else None,
        "delta_eo": _nan_to_null(dict(r.delta_eo)) if r.delta_eo is not None else None,
        "stat_parity_gap": (
            _nan_to_null(dict(r.stat_parity_gap))
            if r.stat_parity_gap is not None
            else None
        ),
        "intersectional_delta_dp": _nan_to_null(r.intersectional_delta_dp),
        "group_collapse_warnings": list(r.group_collapse_warnings),
        "delta_eo_ci": _dict_ci_to_list(getattr(r, "delta_eo_ci", None)),
        "delta_dp_ci": _dict_ci_to_list(getattr(r, "delta_dp_ci", None)),
    }


def _dim5_from_dict(d: Dict) -> Dim5Result:
    r = Dim5Result(
        protected_cols=d.get("protected_cols", []),
        target_col=d.get("target_col", ""),
        task_type=d.get("task_type", ""),
        bias_nmi_real=_null_to_nan(d.get("bias_nmi_real", {})),
        bias_nmi_synth=_null_to_nan(d.get("bias_nmi_synth", {})),
        delta_dp=(
            _null_to_nan(d["delta_dp"])
            if d.get("delta_dp") is not None
            else {}
        ),
        delta_eop=(
            _null_to_nan(d["delta_eop"])
            if d.get("delta_eop") is not None
            else None
        ),
        delta_eo=(
            _null_to_nan(d["delta_eo"])
            if d.get("delta_eo") is not None
            else None
        ),
        stat_parity_gap=(
            _null_to_nan(d["stat_parity_gap"])
            if d.get("stat_parity_gap") is not None
            else None
        ),
        intersectional_delta_dp=_null_to_nan(d.get("intersectional_delta_dp")),
        group_collapse_warnings=d.get("group_collapse_warnings", []),
    )
    r.delta_eo_ci = _dict_list_to_ci(d.get("delta_eo_ci"))
    r.delta_dp_ci = _dict_list_to_ci(d.get("delta_dp_ci"))
    return r


# ---------------------------------------------------------------------------
# Bundle metadata
# ---------------------------------------------------------------------------


@dataclass
class BundleMetadata:
    """Provenance information stored alongside benchmark results.

    Attributes
    ----------
    model_name:
        Human-readable generator model identifier.
    dataset_name:
        Human-readable dataset identifier.
    schema_fingerprint:
        MD5 hash of the :class:`~llm_gtd_benchmark.core.schema.DataSchema`
        column names and types — used to detect schema drift on load.
    timestamp:
        ISO-8601 UTC timestamp of when the bundle was created.
    format_version:
        Serialisation schema version (currently ``"1.0"``).
    notes:
        Optional free-text annotation.
    """

    model_name: str
    dataset_name: str
    schema_fingerprint: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    format_version: str = _FORMAT_VERSION
    notes: str = ""


# ---------------------------------------------------------------------------
# ResultBundle
# ---------------------------------------------------------------------------


@dataclass
class ResultBundle:
    """Container for all six benchmark dimension results and provenance metadata.

    Supports round-trip JSON serialisation via :meth:`save` / :meth:`load`.

    Parameters
    ----------
    metadata:
        :class:`BundleMetadata` with model, dataset, timestamp, and schema
        fingerprint.
    result0 … result5:
        Optional per-dimension result objects.  ``None`` indicates the
        dimension was not evaluated or failed (check :attr:`errors`).
    errors:
        Maps ``"dim0"`` … ``"dim5"`` to a human-readable error string or
        full Python traceback.  Empty dict when all requested dimensions
        succeeded.

    Notes
    -----
    ``result0.clean_df`` is always an **empty DataFrame** when loaded from
    disk — the clean DataFrame is not serialised.  Re-run
    :class:`~llm_gtd_benchmark.metrics.dimension0.StructuralInterceptor` to
    repopulate it.
    """

    metadata: BundleMetadata

    result0: Optional[Dim0Result] = None
    result1: Optional[Dim1Result] = None
    result2: Optional[Dim2Result] = None
    result3: Optional[Dim3Result] = None
    result4: Optional[Dim4Result] = None
    result5: Optional[Dim5Result] = None

    errors: Dict[str, str] = field(default_factory=dict)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialise to a JSON-safe dict (NaN → ``null``)."""
        return {
            "format_version": _FORMAT_VERSION,
            "metadata": {
                "model_name": self.metadata.model_name,
                "dataset_name": self.metadata.dataset_name,
                "schema_fingerprint": self.metadata.schema_fingerprint,
                "timestamp": self.metadata.timestamp,
                "format_version": self.metadata.format_version,
                "notes": self.metadata.notes,
            },
            "results": {
                "dim0": _dim0_to_dict(self.result0) if self.result0 is not None else None,
                "dim1": _dim1_to_dict(self.result1) if self.result1 is not None else None,
                "dim2": _dim2_to_dict(self.result2) if self.result2 is not None else None,
                "dim3": _dim3_to_dict(self.result3) if self.result3 is not None else None,
                "dim4": _dim4_to_dict(self.result4) if self.result4 is not None else None,
                "dim5": _dim5_to_dict(self.result5) if self.result5 is not None else None,
            },
            "errors": dict(self.errors),
        }

    def save(self, path: str | Path, indent: int = 2) -> None:
        """Write to *path* as UTF-8 JSON.

        Parent directories are created automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=indent, ensure_ascii=False)
        logger.info("ResultBundle saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ResultBundle":
        """Load a JSON bundle from *path*.

        Parameters
        ----------
        path:
            Path to a ``.json`` file previously written by :meth:`save`.

        Raises
        ------
        FileNotFoundError:
            When *path* does not exist.
        json.JSONDecodeError:
            When the file is not valid JSON.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        fv = data.get("format_version", "unknown")
        if fv != _FORMAT_VERSION:
            logger.warning(
                "ResultBundle.load: format_version '%s' differs from current '%s'. "
                "Fields added in later versions will silently default to None / 0.",
                fv,
                _FORMAT_VERSION,
            )

        meta_d = data.get("metadata", {})
        metadata = BundleMetadata(
            model_name=meta_d.get("model_name", ""),
            dataset_name=meta_d.get("dataset_name", ""),
            schema_fingerprint=meta_d.get("schema_fingerprint", ""),
            timestamp=meta_d.get("timestamp", ""),
            format_version=meta_d.get("format_version", fv),
            notes=meta_d.get("notes", ""),
        )

        results = data.get("results", {})
        bundle = cls(metadata=metadata)
        bundle.result0 = _dim0_from_dict(results["dim0"]) if results.get("dim0") else None
        bundle.result1 = _dim1_from_dict(results["dim1"]) if results.get("dim1") else None
        bundle.result2 = _dim2_from_dict(results["dim2"]) if results.get("dim2") else None
        bundle.result3 = _dim3_from_dict(results["dim3"]) if results.get("dim3") else None
        bundle.result4 = _dim4_from_dict(results["dim4"]) if results.get("dim4") else None
        bundle.result5 = _dim5_from_dict(results["dim5"]) if results.get("dim5") else None
        bundle.errors = data.get("errors", {})

        logger.info("ResultBundle loaded ← %s", path)
        return bundle

    def validate_schema(self, schema: DataSchema) -> bool:
        """Check that this bundle's schema fingerprint matches *schema*.

        Returns ``True`` when they match; logs a warning and returns ``False``
        when they differ (indicating the bundle was evaluated on a different
        schema version).
        """
        fp = schema_fingerprint(schema)
        if self.metadata.schema_fingerprint != fp:
            logger.warning(
                "ResultBundle.validate_schema: fingerprint mismatch — "
                "bundle was computed against a different schema.  "
                "bundle=%s  current=%s",
                self.metadata.schema_fingerprint,
                fp,
            )
            return False
        return True

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def dimensions_computed(self) -> List[str]:
        """Names of dimensions that have results (not ``None``)."""
        return [
            f"dim{i}"
            for i, r in enumerate(
                [self.result0, self.result1, self.result2,
                 self.result3, self.result4, self.result5]
            )
            if r is not None
        ]

    @property
    def has_errors(self) -> bool:
        """``True`` when at least one dimension reported a failure."""
        return bool(self.errors)

    def __repr__(self) -> str:
        dims = ", ".join(self.dimensions_computed) or "none"
        return (
            f"ResultBundle(model='{self.metadata.model_name}', "
            f"dataset='{self.metadata.dataset_name}', "
            f"dims=[{dims}], errors={len(self.errors)})"
        )
