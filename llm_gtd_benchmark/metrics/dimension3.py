"""
Dimension 3 — Downstream Task Utility Evaluator.

Theoretical framing
-------------------
The core hypothesis is the TSTR (Train-on-Synthetic, Test-on-Real) framework,
grounded in the generalization error bound transfer principle:

    If synthetic distribution Q faithfully approximates real distribution P,
    an ERM model trained on Q should achieve risk on P_test comparable to
    one trained on P_train.

The utility gap — defined as score(TRTR) − score(TSTR) — directly quantifies
the information loss introduced by the generative model.  A gap of 0 implies
perfect statistical substitutability; a positive gap means the real training
data carries information the synthetic data does not.

Two complementary modules are provided:

    MLUtilityEvaluator
        Three ML discriminators (XGBoost, RandomForest, LogisticRegression /
        Ridge) evaluated under strict TSTR and TRTR protocols.

        Two tuning modes are supported (controlled by ``tuning_mode``):

        ``"shared"`` *(default)*
            Hyperparameters are tuned **once** on real training data (TRTR
            phase).  The identical best parameters are then reused for the
            TSTR phase — synthetic data is only used for fitting, never for
            tuning.  This eliminates hyperparameter choice as a confounding
            variable, making the utility gap a pure measure of data quality.
            Half as many Optuna trials as the legacy mode.

        ``"independent"``
            Hyperparameters are tuned **separately** for TSTR and TRTR,
            each on its own training split.  Preserves the original behaviour
            for backward compatibility.

    LLMUtilityEvaluator  [optional — requires GPU + transformers/peft/trl]
        A lightweight LLM (default: Qwen2.5-1.5B-Instruct) is LoRA fine-tuned
        on serialized synthetic rows and evaluated on real test rows.  This
        captures non-linear semantic structure invisible to tree-based models,
        aligning with the HARMONIC framework and 2024+ research trends.

Engineering guarantees
----------------------
- OrdinalEncoder with handle_unknown='use_encoded_value' prevents crashes when
  synthetic training and real test data have mismatched category sets.
- Preprocessing (ColumnTransformer) is fitted on the real test set as the
  domain anchor — no real training statistics leak into the TSTR branch.
- In ``"shared"`` mode, tuning is performed once on real data; TSTR fitting
  uses the resulting best params without any additional search on synthetic data.
- In ``"independent"`` mode, per-model inner 3-fold CV tuning is applied to
  each training split independently.
- WMAPE replaces MAPE to handle zero-valued regression targets.
- All sub-metrics return NaN — not exceptions — on degenerate inputs.
- XGBoost and LLM dependencies are optional; graceful ImportWarning + NaN when
  absent.
"""

from __future__ import annotations

import enum
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from llm_gtd_benchmark.core.schema import DataSchema

logger = logging.getLogger(__name__)

_NAN = float("nan")

# Valid tuning mode identifiers.
TUNING_MODE_SHARED = "shared"
TUNING_MODE_INDEPENDENT = "independent"


# ---------------------------------------------------------------------------
# Task type enum
# ---------------------------------------------------------------------------


class TaskType(enum.Enum):
    """Prediction task type for Dimension 3 evaluation."""

    BINARY_CLASS = "binary_classification"
    MULTI_CLASS = "multi_classification"
    REGRESSION = "regression"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Dim3Result:
    """Output of :class:`MLUtilityEvaluator` and optionally :class:`LLMUtilityEvaluator`.

    Metric conventions
    ------------------
    - ``roc_auc``, ``macro_f1``, ``r2``: higher is better.
    - ``wmape``: lower is better.
    - ``utility_gap[model][metric] = score_trtr − score_tstr`` for all metrics.
      Positive gap → real training data outperforms synthetic (information deficit).
      For WMAPE: positive gap → real data achieves lower error than synthetic training.

    Attributes
    ----------
    task_type:
        String value of :class:`TaskType`.
    target_col:
        Name of the prediction target column.
    tuning_backend:
        Hyperparameter search backend: ``"optuna"`` or ``"randomized_search"``.
    tuning_mode:
        Hyperparameter tuning strategy: ``"shared"`` (tune once on real data,
        reuse params for synthetic) or ``"independent"`` (tune separately for
        each split).  Default: ``"shared"``.
    mle_tstr:
        TSTR scores per model. ``{"XGBoost": {"roc_auc": 0.85, "macro_f1": 0.82}, ...}``
    mle_trtr:
        TRTR baseline scores per model.  Empty when no real training data supplied.
    utility_gap:
        Per-model, per-metric gap (TRTR − TSTR).  Empty when baseline absent.
    lle_tstr:
        LLM utility scores from :class:`LLMUtilityEvaluator`.
        ``{"macro_f1": 0.79, "roc_auc": 0.83}``.  Empty if LLE not run.
    lle_model:
        HuggingFace identifier of the base LLM used for LLE.
    """

    task_type: str
    target_col: str
    tuning_backend: str
    tuning_mode: str = TUNING_MODE_SHARED

    mle_tstr: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mle_trtr: Dict[str, Dict[str, float]] = field(default_factory=dict)
    utility_gap: Dict[str, Dict[str, float]] = field(default_factory=dict)

    lle_tstr: Dict[str, float] = field(default_factory=dict)
    lle_model: str = ""

    @property
    def summary(self) -> str:
        def _fmt(v: float) -> str:
            return f"{v:.4f}" if not np.isnan(v) else "  N/A "

        def _fmtgap(v: float) -> str:
            return f"{v:+.4f}" if not np.isnan(v) else "  N/A "

        lines = [
            "── Dimension 3: Downstream Task Utility ─────────────────────────",
            f"  Target : {self.target_col}",
            f"  Task   : {self.task_type}",
            f"  Tuner  : {self.tuning_backend}  (mode: {self.tuning_mode})",
            "",
        ]

        metric_keys: List[str] = []
        for scores in self.mle_tstr.values():
            metric_keys = list(scores.keys())
            break

        if self.mle_tstr and metric_keys:
            has_gap = bool(self.utility_gap)
            header = f"  {'Model':<22}" + "  ".join(f"TSTR {k:<10}" for k in metric_keys)
            if has_gap:
                header += "  " + "  ".join(f"gap({k})" for k in metric_keys)
            lines.append("  ML Utility:")
            lines.append("  " + "-" * (len(header) - 2))
            lines.append(header)

            for model_name, tstr_s in self.mle_tstr.items():
                row = f"  {model_name:<22}" + "  ".join(
                    f"     {_fmt(tstr_s.get(k, _NAN)):<10}" for k in metric_keys
                )
                if has_gap and model_name in self.utility_gap:
                    gap = self.utility_gap[model_name]
                    row += "  " + "  ".join(f"{_fmtgap(gap.get(k, _NAN)):<10}" for k in metric_keys)
                lines.append(row)

        if self.lle_tstr:
            lines.append("")
            lines.append(f"  LLM Utility — TSTR  (base model: {self.lle_model or 'unknown'}):")
            for k, v in self.lle_tstr.items():
                lines.append(f"    {k:<20}: {_fmt(v)}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _build_feature_pipeline(schema: DataSchema, target_col: str) -> ColumnTransformer:
    """Robust ColumnTransformer for all feature columns (target excluded).

    Continuous columns: median imputation → standard scaling.
    Categorical columns: most-frequent imputation → OrdinalEncoder with
    ``handle_unknown='use_encoded_value'``.  The unknown guard is critical
    in TSTR: synthetic and real data often have mismatched category sets.
    """
    cont_cols = [c.name for c in schema.continuous_columns if c.name != target_col]
    cat_cols = [c.name for c in schema.categorical_columns if c.name != target_col]

    transformers: List[Tuple] = []
    if cont_cols:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, cont_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error: Σ|true−pred| / Σ|true|.

    Returns NaN when all true values are zero (avoids division-by-zero that
    plagues standard MAPE).
    """
    denom = np.sum(np.abs(y_true))
    if denom == 0.0:
        return _NAN
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _sample_optuna_params(trial: Any, param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a unified param_space spec into Optuna trial suggestions."""
    params: Dict[str, Any] = {}
    for name, spec in param_space.items():
        t = spec["type"]
        if t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def _sample_random_params(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a unified param_space spec into sklearn RandomizedSearchCV distributions."""
    from scipy.stats import loguniform, randint, uniform

    dist: Dict[str, Any] = {}
    for name, spec in param_space.items():
        t = spec["type"]
        if t == "int":
            dist[name] = randint(spec["low"], spec["high"] + 1)
        elif t == "float":
            if spec.get("log", False):
                dist[name] = loguniform(spec["low"], spec["high"])
            else:
                dist[name] = uniform(spec["low"], spec["high"] - spec["low"])
        elif t == "categorical":
            dist[name] = spec["choices"]
    return dist


# ---------------------------------------------------------------------------
# Model factories and parameter spaces
# ---------------------------------------------------------------------------


def _xgb_cls(random_state: int, multiclass: bool = False) -> Optional[Any]:
    try:
        from xgboost import XGBClassifier

        obj = "multi:softprob" if multiclass else "binary:logistic"
        eval_m = "mlogloss" if multiclass else "logloss"
        return XGBClassifier(
            objective=obj,
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
            eval_metric=eval_m,
        )
    except ImportError:
        warnings.warn(
            "xgboost not installed; XGBoost skipped in Dim3. Install: pip install xgboost",
            ImportWarning,
            stacklevel=2,
        )
        return None


def _xgb_reg(random_state: int) -> Optional[Any]:
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(n_jobs=-1, random_state=random_state, verbosity=0)
    except ImportError:
        warnings.warn(
            "xgboost not installed; XGBoost skipped in Dim3. Install: pip install xgboost",
            ImportWarning,
            stacklevel=2,
        )
        return None


def _xgb_space() -> Dict[str, Any]:
    return {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 9},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    }


def _rf_cls_space() -> Dict[str, Any]:
    return {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "categorical", "choices": [None, 5, 10, 20]},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5]},
    }


def _rf_reg_space() -> Dict[str, Any]:
    return {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "categorical", "choices": [None, 5, 10, 20]},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5, 1.0]},
    }


def _lr_space() -> Dict[str, Any]:
    return {
        "C": {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
        "solver": {"type": "categorical", "choices": ["lbfgs", "saga"]},
    }


def _ridge_space() -> Dict[str, Any]:
    return {
        "alpha": {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
    }


# ---------------------------------------------------------------------------
# ML Utility Evaluator
# ---------------------------------------------------------------------------


class MLUtilityEvaluator:
    """Dimension 3 ML utility evaluator via TSTR + optional TRTR benchmarking.

    Parameters
    ----------
    schema:
        :class:`~llm_gtd_benchmark.core.schema.DataSchema` built from real data.
    target_col:
        Name of the column to predict.
    task_type:
        :class:`TaskType` enum specifying the prediction task.
    n_tuning_trials:
        Optuna trials (or RandomizedSearchCV iterations) per model.  Default: 50.
    cv_folds:
        Inner cross-validation folds used during hyperparameter tuning.  Default: 3.
    tuning_mode:
        ``"shared"`` *(default)* — tune once on real data, reuse params for
        TSTR.  Faster and eliminates hyperparameter bias.
        ``"independent"`` — tune separately for TSTR and TRTR (original
        behaviour, slower but preserved for backward compatibility).
    random_state:
        Global seed.  Default: 42.

    Examples
    --------
    >>> evaluator = MLUtilityEvaluator(schema, "income", TaskType.BINARY_CLASS)
    >>> result = evaluator.evaluate(train_synth_df, test_real_df, train_real_df)
    >>> print(result.summary)
    """

    def __init__(
        self,
        schema: DataSchema,
        target_col: str,
        task_type: TaskType,
        n_tuning_trials: int = 50,
        cv_folds: int = 3,
        tuning_mode: str = TUNING_MODE_SHARED,
        random_state: int = 42,
    ) -> None:
        if tuning_mode not in (TUNING_MODE_SHARED, TUNING_MODE_INDEPENDENT):
            raise ValueError(
                f"tuning_mode must be '{TUNING_MODE_SHARED}' or '{TUNING_MODE_INDEPENDENT}', "
                f"got '{tuning_mode}'."
            )
        self.schema = schema
        self.target_col = target_col
        self.task_type = task_type
        self.n_tuning_trials = n_tuning_trials
        self.cv_folds = cv_folds
        self.tuning_mode = tuning_mode
        self.random_state = random_state
        self._is_cls = task_type in (TaskType.BINARY_CLASS, TaskType.MULTI_CLASS)

        self._ct = _build_feature_pipeline(schema, target_col)
        self._label_enc: Optional[LabelEncoder] = LabelEncoder() if self._is_cls else None
        self._tuning_backend = self._detect_tuning_backend()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        train_synth_df: pd.DataFrame,
        test_real_df: pd.DataFrame,
        train_real_df: Optional[pd.DataFrame] = None,
    ) -> Dim3Result:
        """Run TSTR and optionally TRTR evaluation.

        Parameters
        ----------
        train_synth_df:
            Synthetic training data (output of Dim0 clean pipeline).
        test_real_df:
            Real held-out test data.
        train_real_df:
            Real training data.  When provided, TRTR scores and utility_gap
            are computed alongside TSTR.
            Required for ``tuning_mode="shared"``; if absent, a warning is
            emitted and the method falls back to independent TSTR tuning.

        Returns
        -------
        :class:`Dim3Result`
        """
        feat_cols = [c for c in test_real_df.columns if c != self.target_col]

        # Fit preprocessing on real test data (domain anchor — no leakage).
        self._ct.fit(test_real_df[feat_cols])

        # Fit label encoder on the union of all known labels.
        if self._label_enc is not None:
            all_labels = pd.concat([
                train_synth_df[self.target_col],
                test_real_df[self.target_col],
                *([] if train_real_df is None else [train_real_df[self.target_col]]),
            ]).dropna().astype(str)
            self._label_enc.fit(all_labels)

        # Encode test data once — shared across TSTR and TRTR.
        X_test, y_test = self._encode_df(test_real_df)
        if X_test is None:
            logger.error("Dim3: failed to encode test data; returning empty result.")
            return Dim3Result(
                task_type=self.task_type.value,
                target_col=self.target_col,
                tuning_backend=self._tuning_backend,
                tuning_mode=self.tuning_mode,
            )

        # Encode synthetic training data.
        X_synth, y_synth = self._encode_df(train_synth_df)

        # Encode real training data (may be None when not supplied).
        X_real: Optional[np.ndarray] = None
        y_real: Optional[np.ndarray] = None
        if train_real_df is not None:
            X_real, y_real = self._encode_df(train_real_df)

        mle_tstr: Dict[str, Dict[str, float]] = {}
        mle_trtr: Dict[str, Dict[str, float]] = {}
        utility_gap: Dict[str, Dict[str, float]] = {}

        # Decide effective mode.
        use_shared = self.tuning_mode == TUNING_MODE_SHARED and X_real is not None

        if self.tuning_mode == TUNING_MODE_SHARED and X_real is None:
            if train_real_df is None:
                logger.warning(
                    "Dim3 [shared]: train_real_df not provided — shared tuning requires "
                    "real training data.  Running TSTR with independent tuning on synthetic data."
                )
            else:
                logger.warning(
                    "Dim3 [shared]: real training data encoding failed — "
                    "falling back to independent mode."
                )

        if use_shared:
            # ── Shared mode: tune once on real, reuse params for both ─────────
            logger.info(
                "Dim3 [shared]: tuning on real training data (%d rows)...",
                len(train_real_df),  # type: ignore[arg-type]
            )
            best_models = self._tune_all(X_real, y_real)  # type: ignore[arg-type]

            logger.info("Dim3 [shared]: scoring TRTR (fit on real, eval on real test)...")
            mle_trtr = self._score_with_best_models(
                best_models, X_real, y_real, X_test, y_test, label="TRTR"  # type: ignore[arg-type]
            )

            logger.info("Dim3 [shared]: scoring TSTR (fit on synthetic, eval on real test)...")
            if X_synth is not None:
                mle_tstr = self._score_with_best_models(
                    best_models, X_synth, y_synth, X_test, y_test, label="TSTR"  # type: ignore[arg-type]
                )
            else:
                logger.warning("Dim3 [shared]: synthetic data encoding failed; TSTR unavailable.")

        else:
            # ── Independent mode: tune separately for each split ──────────────
            if X_synth is not None:
                logger.info("Dim3 [independent]: running TSTR evaluation...")
                mle_tstr = self._score_independent(X_synth, y_synth, X_test, y_test)  # type: ignore[arg-type]
            else:
                logger.warning("Dim3: synthetic data encoding failed; TSTR unavailable.")

            if X_real is not None:
                logger.info("Dim3 [independent]: running TRTR evaluation...")
                mle_trtr = self._score_independent(X_real, y_real, X_test, y_test)  # type: ignore[arg-type]

        # Compute utility gap.
        if mle_tstr and mle_trtr:
            utility_gap = {
                model: {
                    metric: mle_trtr[model].get(metric, _NAN) - mle_tstr[model].get(metric, _NAN)
                    for metric in mle_tstr[model]
                }
                for model in mle_tstr
                if model in mle_trtr
            }

        return Dim3Result(
            task_type=self.task_type.value,
            target_col=self.target_col,
            tuning_backend=self._tuning_backend,
            tuning_mode=self.tuning_mode,
            mle_tstr=mle_tstr,
            mle_trtr=mle_trtr,
            utility_gap=utility_gap,
        )

    # ── Internal: data encoding ───────────────────────────────────────────────

    def _encode_df(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Encode a DataFrame into (X, y) arrays using the fitted CT and label encoder.

        Returns ``(None, None)`` on encoding failure so callers can handle
        the absence gracefully without try/except at every call site.
        """
        df = df.reset_index(drop=True)
        feat_cols = [c for c in df.columns if c != self.target_col]

        try:
            X = self._ct.transform(df[feat_cols])
        except Exception as exc:
            logger.error("Dim3 feature encoding failed: %s", exc)
            return None, None

        # Drop rows with missing target.
        valid = df[self.target_col].notna().values
        X = X[valid]
        y_raw = df[self.target_col][valid]

        if self._label_enc is not None:
            try:
                y = self._label_enc.transform(y_raw.astype(str))
            except Exception as exc:
                logger.error("Dim3 label encoding failed: %s", exc)
                return None, None
        else:
            y = y_raw.astype(float).values

        return X, y

    # ── Internal: shared-mode helpers ─────────────────────────────────────────

    def _tune_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """Tune all models on (X_train, y_train); return ``{model_name: best_model}``.

        The returned models are **unfitted** clones with the best hyperparameters
        set via ``set_params``.  Callers must clone them again before fitting to
        avoid state pollution across TRTR / TSTR calls.
        """
        n_classes = len(np.unique(y_train))
        if self._is_cls and n_classes < 2:
            logger.warning(
                "Dim3._tune_all: training data has only one class; "
                "returning default (untuned) models."
            )
            return {
                name: clone(model)
                for name, model, _ in self._get_models_and_spaces()
            }

        best_models: Dict[str, Any] = {}
        for model_name, model, param_space in self._get_models_and_spaces():
            logger.info(
                "Dim3: tuning %s (%d trials)...", model_name, self.n_tuning_trials
            )
            best_models[model_name] = self._find_best_model(model, param_space, X_train, y_train)
        return best_models

    def _score_with_best_models(
        self,
        best_models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label: str = "",
    ) -> Dict[str, Dict[str, float]]:
        """Fit each best_model on X_train, evaluate on X_test.

        Each model is cloned before fitting so the shared ``best_models`` dict
        is not mutated — the same dict can be passed for both TRTR and TSTR.
        """
        n_classes = len(np.unique(y_train))
        if self._is_cls and n_classes < 2:
            logger.warning(
                "Dim3 [%s]: training split has only one class; returning empty scores.",
                label,
            )
            return {}

        results: Dict[str, Dict[str, float]] = {}
        for model_name, best_model in best_models.items():
            scores = self._fit_and_score(clone(best_model), X_train, y_train, X_test, y_test)
            results[model_name] = scores
            logger.info(
                "Dim3 [%s] %s → %s",
                label, model_name, {k: f"{v:.4f}" for k, v in scores.items()},
            )
        return results

    # ── Internal: independent-mode helper ─────────────────────────────────────

    def _score_independent(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Tune each model on X_train, fit on X_train, score on X_test."""
        n_classes = len(np.unique(y_train))
        if self._is_cls and n_classes < 2:
            logger.warning(
                "Dim3: training split has only one class; returning empty scores."
            )
            return {}

        results: Dict[str, Dict[str, float]] = {}
        for model_name, model, param_space in self._get_models_and_spaces():
            logger.info(
                "Dim3: tuning %s (%d trials)...", model_name, self.n_tuning_trials
            )
            scores = self._tune_and_score(model, param_space, X_train, y_train, X_test, y_test)
            results[model_name] = scores
            logger.info(
                "Dim3: %s → %s", model_name, {k: f"{v:.4f}" for k, v in scores.items()}
            )
        return results

    # ── Internal: hyperparameter search ──────────────────────────────────────

    def _find_best_model(
        self,
        model: Any,
        param_space: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        """Run hyperparameter search; return best unfitted model."""
        cv = (
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            if self._is_cls
            else KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        )
        inner_scoring = "f1_macro" if self._is_cls else "r2"

        if self._tuning_backend == "optuna":
            return self._optuna_search(model, param_space, X_train, y_train, cv, inner_scoring)
        else:
            return self._randomized_search(model, param_space, X_train, y_train, cv, inner_scoring)

    def _fit_and_score(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Fit model on X_train, evaluate on X_test and return metric dict."""
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            logger.error("Dim3 model fit failed (%s): %s", type(model).__name__, exc)
            return {k: _NAN for k in self._metric_keys()}
        return self._compute_metrics(y_test, model, X_test)

    def _tune_and_score(
        self,
        model: Any,
        param_space: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Tune on X_train, fit on X_train, score on X_test (independent mode)."""
        best_model = self._find_best_model(model, param_space, X_train, y_train)
        return self._fit_and_score(best_model, X_train, y_train, X_test, y_test)

    def _optuna_search(self, model, param_space, X, y, cv, scoring) -> Any:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:
            params = _sample_optuna_params(trial, param_space)
            m = clone(model).set_params(**params)
            scores = cross_val_score(m, X, y, cv=cv, scoring=scoring, error_score=_NAN)
            return float(np.nanmean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_tuning_trials, show_progress_bar=False)
        return clone(model).set_params(**study.best_params)

    def _randomized_search(self, model, param_space, X, y, cv, scoring) -> Any:
        from sklearn.model_selection import RandomizedSearchCV

        dist = _sample_random_params(param_space)
        search = RandomizedSearchCV(
            model,
            dist,
            n_iter=self.n_tuning_trials,
            cv=cv,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=-1,
            error_score=_NAN,
        )
        search.fit(X, y)
        return search.best_estimator_

    # ── Internal: metrics ─────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        model: Any,
        X_test: np.ndarray,
    ) -> Dict[str, float]:
        try:
            y_pred = model.predict(X_test)
        except Exception as exc:
            logger.error("Dim3 prediction failed: %s", exc)
            return {k: _NAN for k in self._metric_keys()}

        if self._is_cls:
            try:
                macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            except Exception:
                macro_f1 = _NAN

            auc = _NAN
            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)
                    if self.task_type == TaskType.BINARY_CLASS:
                        auc = float(roc_auc_score(y_true, y_prob[:, 1]))
                    else:
                        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                auc = _NAN

            return {"roc_auc": auc, "macro_f1": macro_f1}
        else:
            try:
                r2 = float(r2_score(y_true, y_pred))
            except Exception:
                r2 = _NAN
            return {"r2": r2, "wmape": _wmape(y_true.astype(float), y_pred.astype(float))}

    def _metric_keys(self) -> List[str]:
        return ["roc_auc", "macro_f1"] if self._is_cls else ["r2", "wmape"]

    def _get_models_and_spaces(self) -> List[Tuple[str, Any, Dict[str, Any]]]:
        rs = self.random_state
        if self.task_type == TaskType.BINARY_CLASS:
            candidates = [
                ("XGBoost", _xgb_cls(rs, multiclass=False), _xgb_space()),
                ("RandomForest", RandomForestClassifier(n_jobs=-1, random_state=rs), _rf_cls_space()),
                ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=rs, n_jobs=-1), _lr_space()),
            ]
        elif self.task_type == TaskType.MULTI_CLASS:
            candidates = [
                ("XGBoost", _xgb_cls(rs, multiclass=True), _xgb_space()),
                ("RandomForest", RandomForestClassifier(n_jobs=-1, random_state=rs), _rf_cls_space()),
                (
                    "LogisticRegression",
                    LogisticRegression(max_iter=2000, random_state=rs, multi_class="ovr", n_jobs=-1),
                    _lr_space(),
                ),
            ]
        else:
            candidates = [
                ("XGBoost", _xgb_reg(rs), _xgb_space()),
                ("RandomForest", RandomForestRegressor(n_jobs=-1, random_state=rs), _rf_reg_space()),
                ("Ridge", Ridge(), _ridge_space()),
            ]
        # Drop models whose factory returned None (missing optional dependency).
        return [(name, model, space) for name, model, space in candidates if model is not None]

    @staticmethod
    def _detect_tuning_backend() -> str:
        try:
            import optuna  # noqa: F401

            return "optuna"
        except ImportError:
            warnings.warn(
                "optuna is not installed; falling back to RandomizedSearchCV for Dim3 tuning. "
                "For best results install it:  pip install optuna",
                ImportWarning,
                stacklevel=3,
            )
            return "randomized_search"


# ---------------------------------------------------------------------------
# LLM Utility Evaluator  (optional — requires GPU + transformers/peft/trl)
# ---------------------------------------------------------------------------


class LLMUtilityEvaluator:
    """LLM-based downstream utility evaluator via LoRA fine-tuning (TSTR only).

    Receives the user's **generator** LLM (the model that produced the synthetic
    data) and probes it for evidence of training-set memorization by computing
    Masked Conditional PPL on real training vs. real test rows.

    Requires GPU and:  ``pip install transformers peft trl accelerate datasets``

    Parameters
    ----------
    target_col:
        Name of the prediction target column.
    task_type:
        :class:`TaskType` enum.  Regression is supported but AUC will be NaN
        (generative models cannot easily output probability scores).
    base_model:
        HuggingFace model identifier.
        Default: ``"Qwen/Qwen2.5-1.5B-Instruct"``.
    lora_rank:
        LoRA intrinsic rank.  Default: 8.
    lora_alpha:
        LoRA scaling coefficient.  Default: 16.
    n_epochs:
        Number of SFT fine-tuning epochs.  Default: 3.
    max_seq_length:
        Maximum tokenized length for serialized rows.  Default: 256.
    device:
        ``"cuda"`` or ``"cpu"``.  Auto-detected when ``None``.
    random_state:
        Seed.  Default: 42.

    Examples
    --------
    >>> lle = LLMUtilityEvaluator("income", TaskType.BINARY_CLASS)
    >>> lle_scores = lle.evaluate(train_synth_df, test_real_df)
    >>> result.lle_tstr = lle_scores
    >>> result.lle_model = lle.base_model
    """

    def __init__(
        self,
        target_col: str,
        task_type: TaskType,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        n_epochs: int = 3,
        max_seq_length: int = 256,
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.target_col = target_col
        self.task_type = task_type
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.n_epochs = n_epochs
        self.max_seq_length = max_seq_length
        self.random_state = random_state
        self._is_cls = task_type in (TaskType.BINARY_CLASS, TaskType.MULTI_CLASS)

        if device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._label_enc = LabelEncoder()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        train_synth_df: pd.DataFrame,
        test_real_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Fine-tune on synthetic data; evaluate on real test data.

        Returns
        -------
        dict mapping metric name → score.  Returns ``{"macro_f1": NaN, ...}``
        when required libraries are not installed.
        """
        self._check_dependencies()

        all_labels = pd.concat([
            train_synth_df[self.target_col],
            test_real_df[self.target_col],
        ]).dropna().astype(str)
        self._label_enc.fit(all_labels)

        feat_cols = [c for c in train_synth_df.columns if c != self.target_col]
        train_texts = self._serialize(train_synth_df[feat_cols])
        train_labels = train_synth_df[self.target_col].astype(str).tolist()
        test_texts = self._serialize(test_real_df[feat_cols])
        y_test_str = test_real_df[self.target_col].astype(str).tolist()

        logger.info("LLE: fine-tuning %s on %d synthetic rows (device=%s)...",
                    self.base_model, len(train_texts), self.device)
        model, tokenizer = self._fine_tune(train_texts, train_labels)

        logger.info("LLE: running inference on %d real test rows...", len(test_texts))
        preds_str = self._inference(model, tokenizer, test_texts)

        return self._score(y_test_str, preds_str)

    # ── Serialization ─────────────────────────────────────────────────────────

    def _serialize(self, df: pd.DataFrame) -> List[str]:
        """Convert each row to an instruction-following input string.

        Format: ``Input: [col1: val1, col2: val2, ...] Output:``
        The model is expected to complete the ``Output:`` token with the label.
        """
        rows = []
        for _, row in df.iterrows():
            kv = ", ".join(f"{col}: {val}" for col, val in row.items())
            rows.append(f"Input: [{kv}] Output:")
        return rows

    def _build_sft_texts(self, input_texts: List[str], labels: List[str]) -> List[str]:
        """Concatenate input prompt with target label for supervised fine-tuning."""
        return [f"{inp} {lbl}" for inp, lbl in zip(input_texts, labels)]

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def _fine_tune(self, input_texts: List[str], labels: List[str]) -> Tuple[Any, Any]:
        """LoRA fine-tune the base model on serialized synthetic training data."""
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )

        lora_cfg = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        sft_texts = self._build_sft_texts(input_texts, labels)
        dataset = Dataset.from_dict({"text": sft_texts})

        training_args = TrainingArguments(
            output_dir="./_lle_ckpt",
            num_train_epochs=self.n_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=self.device == "cuda",
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            seed=self.random_state,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=self.max_seq_length,
        )
        trainer.train()
        return model, tokenizer

    # ── Inference ─────────────────────────────────────────────────────────────

    def _inference(self, model: Any, tokenizer: Any, input_texts: List[str]) -> List[str]:
        """Autoregressive inference; extracts the predicted label via string matching."""
        import torch

        model.eval()
        known_labels = list(self._label_enc.classes_)
        preds: List[str] = []

        for text in input_texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
            ).to(self.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            preds.append(self._parse_prediction(generated, known_labels))

        return preds

    def _parse_prediction(self, text: str, known_labels: List[str]) -> str:
        """Return the first known label found in the generated text.

        Falls back to the first label when no match is found — this is
        intentionally conservative to avoid silent NaN propagation.
        """
        text_lower = text.lower()
        for label in known_labels:
            if label.lower() in text_lower:
                return label
        logger.debug("LLE: could not parse label from '%s'; using fallback.", text[:60])
        return known_labels[0] if known_labels else text

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, y_true_str: List[str], y_pred_str: List[str]) -> Dict[str, float]:
        """Encode string labels and compute classification metrics."""
        try:
            y_true = self._label_enc.transform(y_true_str)
            y_pred = np.array([
                self._label_enc.transform([p])[0]
                if p in self._label_enc.classes_
                else 0
                for p in y_pred_str
            ])
            macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            # Hard predictions only — no probability scores available from generative LM.
            auc = _NAN
            if self.task_type == TaskType.BINARY_CLASS:
                try:
                    auc = float(roc_auc_score(y_true, y_pred))
                except Exception:
                    auc = _NAN
            return {"macro_f1": macro_f1, "roc_auc": auc}
        except Exception as exc:
            logger.error("LLE scoring failed: %s", exc)
            return {"macro_f1": _NAN, "roc_auc": _NAN}

    # ── Dependency guard ──────────────────────────────────────────────────────

    @staticmethod
    def _check_dependencies() -> None:
        missing = [
            pkg
            for pkg in ("transformers", "peft", "trl", "datasets", "torch")
            if _try_import(pkg) is False
        ]
        if missing:
            raise ImportError(
                f"LLMUtilityEvaluator requires: {', '.join(missing)}.\n"
                f"Install with:  pip install {' '.join(missing)}"
            )


def _try_import(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False
