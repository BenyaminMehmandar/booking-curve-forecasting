"""
Booking-curve model: feature engineering, training helpers, and inference utilities.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Constants (aligned with starter/baseline_model.py)
# ---------------------------------------------------------------------------

CHECKPOINTS: List[int] = [90, 60, 45, 30, 21, 14, 7, 3, 1, 0]

TRAIN_END = pd.Timestamp("2025-06-30")
TEST_START = pd.Timestamp("2025-07-01")
TEST_END = pd.Timestamp("2025-09-30")

MAJOR_HOLIDAYS = pd.to_datetime(
    [
        "2022-01-01",
        "2022-07-04",
        "2022-11-24",
        "2022-12-25",
        "2023-01-01",
        "2023-07-04",
        "2023-11-23",
        "2023-12-25",
        "2024-01-01",
        "2024-07-04",
        "2024-11-28",
        "2024-12-25",
        "2025-01-01",
        "2025-07-04",
        "2025-11-27",
        "2025-12-25",
    ]
)

DEFAULT_CV_FOLD_SPECS: List[Tuple[pd.Timestamp, pd.Timestamp]] = [
    (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-28")),
    (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-30")),
    (pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-30")),
]

DEFAULT_PARAM_GRID: List[Dict[str, Any]] = [
    dict(
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
    ),
    dict(
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
    ),
    dict(
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
    ),
]

DEFAULT_BASE_LGB_PARAMS: Dict[str, Any] = dict(
    n_estimators=2000,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def build_actual_curves(reservations_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Reconstruct actual booking curves from reservation data.

    For each (property, room_type, target_night), compute the fraction of
    rooms booked at each checkpoint.

    A reservation contributes to every night it covers: a guest checking in
    June 13 and checking out June 16 occupies a room on nights June 13, 14,
    and 15. So for target_night = June 14, that reservation counts if
    stay_date <= June 14 AND checkout_date > June 14.
    """
    active = reservations_df[reservations_df["status"] != "cancelled"].copy()
    active["booking_date"] = pd.to_datetime(active["booking_date"])
    active["stay_date"] = pd.to_datetime(active["stay_date"])
    active["checkout_date"] = pd.to_datetime(active["checkout_date"])

    all_nights = pd.date_range(
        active["stay_date"].min(),
        active["checkout_date"].max() - pd.Timedelta(days=1),
    )

    curves: List[dict] = []

    for prop_id in active["property_id"].unique():
        prop_data = active[active["property_id"] == prop_id]

        for room_type in prop_data["room_type"].unique():
            rt_data = prop_data[prop_data["room_type"] == room_type]
            total_rooms = metadata[prop_id]["room_types"].get(room_type, {}).get("inventory_count", 1)

            for target_night in all_nights:
                covering = rt_data[
                    (rt_data["stay_date"] <= target_night) & (rt_data["checkout_date"] > target_night)
                ]

                if len(covering) == 0:
                    continue

                curve = {}
                for cp in CHECKPOINTS:
                    cutoff_date = target_night - pd.Timedelta(days=cp)
                    booked_by_cutoff = (covering["booking_date"] <= cutoff_date).sum()
                    curve[str(cp)] = min(booked_by_cutoff / total_rooms, 1.0)

                curves.append(
                    {
                        "property_id": prop_id,
                        "room_type": room_type,
                        "stay_date": target_night.strftime("%Y-%m-%d"),
                        "month": target_night.month,
                        "dow": target_night.dayofweek,
                        "actuals": curve,
                    }
                )

    return pd.DataFrame(curves)


def _holiday_min_abs_days_scalar(stay: pd.Timestamp) -> float:
    return float(np.min(np.abs((MAJOR_HOLIDAYS - stay).days)))


def prepare_daily_events(daily_events: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar date, keyed as ``stay_date`` for merging."""
    out = daily_events.rename(columns={"date": "stay_date"}).copy()
    out["stay_date"] = pd.to_datetime(out["stay_date"])
    return out.sort_values("stay_date").drop_duplicates("stay_date", keep="last")


def enrich_curve_features(
    curves: pd.DataFrame,
    metadata: dict,
    daily_events_prepared: pd.DataFrame,
    reference_min_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Apply the same feature engineering used in training to one or many curve rows.

    ``curves`` must include ``property_id``, ``room_type``, ``stay_date``.
    Training rows also include ``actuals``; prediction rows omit ``actuals``.
    """
    feat = curves.copy()
    feat["stay_date"] = pd.to_datetime(feat["stay_date"])

    if "month" not in feat.columns:
        feat["month"] = feat["stay_date"].dt.month
    if "dow" not in feat.columns:
        feat["dow"] = feat["stay_date"].dt.dayofweek

    feat["year"] = feat["stay_date"].dt.year
    feat["week_of_year"] = feat["stay_date"].dt.isocalendar().week.astype(int)
    feat["day_of_year"] = feat["stay_date"].dt.dayofyear
    feat["quarter"] = feat["stay_date"].dt.quarter
    feat["is_weekend"] = feat["dow"].isin([5, 6]).astype(int)
    feat["days_since_data_start"] = (feat["stay_date"] - reference_min_date).dt.days

    def _rooms(row: pd.Series) -> int:
        return int(metadata[row["property_id"]]["room_types"][row["room_type"]]["inventory_count"])

    feat["total_rooms"] = feat.apply(_rooms, axis=1)
    feat["property_total_rooms"] = feat["property_id"].map(lambda p: int(metadata[p]["total_rooms"]))

    feat = feat.merge(daily_events_prepared, on="stay_date", how="left")

    feat["sin_month"] = np.sin(2 * math.pi * (feat["month"] / 12.0))
    feat["cos_month"] = np.cos(2 * math.pi * (feat["month"] / 12.0))
    feat["sin_dow"] = np.sin(2 * math.pi * (feat["dow"] / 7.0))
    feat["cos_dow"] = np.cos(2 * math.pi * (feat["dow"] / 7.0))
    feat["sin_doy"] = np.sin(2 * math.pi * (feat["day_of_year"] / 366.0))
    feat["cos_doy"] = np.cos(2 * math.pi * (feat["day_of_year"] / 366.0))

    feat["holiday_min_abs_days"] = feat["stay_date"].apply(_holiday_min_abs_days_scalar)
    feat["is_holiday_window_3d"] = (feat["holiday_min_abs_days"] <= 3).astype(int)

    return feat


def expand_actuals_to_wide(feat_base: pd.DataFrame) -> pd.DataFrame:
    """Add ``cp_*`` label columns from nested ``actuals`` dict."""
    wide = feat_base.copy()
    cp_cols = pd.json_normalize(wide["actuals"]).rename(
        columns={str(cp): f"cp_{cp}" for cp in CHECKPOINTS}
    )
    return wide.drop(columns=["actuals"]).join(cp_cols)


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    numeric_features: Tuple[str, ...] = (
        "month",
        "dow",
        "year",
        "week_of_year",
        "day_of_year",
        "quarter",
        "is_weekend",
        "days_since_data_start",
        "sin_month",
        "cos_month",
        "sin_dow",
        "cos_dow",
        "sin_doy",
        "cos_doy",
        "holiday_min_abs_days",
        "is_holiday_window_3d",
        "total_rooms",
        "property_total_rooms",
    )
    categorical_features: Tuple[str, ...] = ("property_id", "room_type")


def category_levels_from_train(df: pd.DataFrame, cfg: FeatureConfig) -> Dict[str, List[str]]:
    return {
        c: sorted(df[c].astype(str).unique().tolist()) for c in cfg.categorical_features
    }


def prepare_X(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    *,
    category_levels: Dict[str, List[str]] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    feats = list(cfg.numeric_features + cfg.categorical_features)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feats].copy()
    for c in cfg.numeric_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cfg.categorical_features:
        if category_levels and c in category_levels:
            X[c] = pd.Categorical(X[c], categories=category_levels[c])
        else:
            X[c] = X[c].astype("category")

    return X, feats


def split_train_test_baseline(feat_wide: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.to_datetime(feat_wide["stay_date"])
    train_df = feat_wide[s <= TRAIN_END].copy()
    test_df = feat_wide[(s >= TEST_START) & (s <= TEST_END)].copy()
    return train_df, test_df


def build_time_folds(
    train_wide: pd.DataFrame,
    fold_specs: Sequence[Tuple[pd.Timestamp, pd.Timestamp]] = DEFAULT_CV_FOLD_SPECS,
) -> List[Tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]]:
    folds: List[Tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]] = []
    stay = pd.to_datetime(train_wide["stay_date"])
    for val_start, val_end in fold_specs:
        tr_idx = train_wide.index[stay < val_start].to_numpy()
        va_idx = train_wide.index[(stay >= val_start) & (stay <= val_end)].to_numpy()
        if len(tr_idx) > 0 and len(va_idx) > 0:
            folds.append((tr_idx, va_idx, val_start, val_end))
    return folds


def categorical_column_indices(feature_names: Sequence[str], cfg: FeatureConfig) -> List[int]:
    return [i for i, c in enumerate(feature_names) if c in cfg.categorical_features]


# ---------------------------------------------------------------------------
# Training: CV + refit
# ---------------------------------------------------------------------------


def tune_checkpoints_with_cv(
    train_wide: pd.DataFrame,
    feature_cfg: FeatureConfig,
    category_levels: Dict[str, List[str]],
    *,
    param_grid: Sequence[Dict[str, Any]] = DEFAULT_PARAM_GRID,
    base_params: Dict[str, Any] | None = None,
    fold_specs: Sequence[Tuple[pd.Timestamp, pd.Timestamp]] = DEFAULT_CV_FOLD_SPECS,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], Dict[int, int]]:
    """Per-checkpoint LightGBM tuning with time-based CV and early stopping."""
    base = dict(DEFAULT_BASE_LGB_PARAMS) if base_params is None else {**DEFAULT_BASE_LGB_PARAMS, **base_params}
    X_all, feature_names = prepare_X(train_wide, feature_cfg, category_levels=category_levels)
    cat_idx = categorical_column_indices(feature_names, feature_cfg)
    folds = build_time_folds(train_wide, fold_specs)

    checkpoint_rows: List[dict] = []
    best_params_by_cp: Dict[int, Dict[str, Any]] = {}
    best_iter_by_cp: Dict[int, int] = {}

    for cp in CHECKPOINTS:
        y_all = train_wide[f"cp_{cp}"].astype(float)
        best_score = np.inf
        best_cfg: Dict[str, Any] | None = None
        best_iters: List[int] = []

        for cfg in param_grid:
            fold_maes: List[float] = []
            fold_iters: List[int] = []

            for tr_idx, va_idx, _, _ in folds:
                X_tr = X_all.loc[tr_idx]
                y_tr = y_all.loc[tr_idx].values
                X_va = X_all.loc[va_idx]
                y_va = y_all.loc[va_idx].values

                model = lgb.LGBMRegressor(**base, **cfg)
                model.fit(
                    X_tr,
                    y_tr,
                    categorical_feature=cat_idx,
                    eval_set=[(X_va, y_va)],
                    eval_metric="l1",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=100, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )

                pred_va = model.predict(X_va, num_iteration=model.best_iteration_)
                fold_maes.append(mean_absolute_error(y_va, pred_va))
                fold_iters.append(int(model.best_iteration_ or model.n_estimators))

            cv_mae = float(np.mean(fold_maes))
            if cv_mae < best_score:
                best_score = cv_mae
                best_cfg = dict(cfg)
                best_iters = fold_iters[:]

        assert best_cfg is not None
        best_params_by_cp[cp] = best_cfg
        med_it = int(np.median(best_iters)) if best_iters else 600
        best_iter_by_cp[cp] = med_it
        checkpoint_rows.append({"checkpoint": cp, "cv_mae": best_score, "best_iter_median": med_it, **best_cfg})

    tuning_df = pd.DataFrame(checkpoint_rows).sort_values("checkpoint", ascending=False)
    return tuning_df, best_params_by_cp, best_iter_by_cp


def fit_per_checkpoint_models(
    train_wide: pd.DataFrame,
    feature_cfg: FeatureConfig,
    category_levels: Dict[str, List[str]],
    best_params_by_cp: Dict[int, Dict[str, Any]],
    best_iter_by_cp: Dict[int, int],
) -> Tuple[Dict[int, lgb.LGBMRegressor], List[str]]:
    """Final fit on the full training window (no early stopping)."""
    X_train, feature_names = prepare_X(train_wide, feature_cfg, category_levels=category_levels)
    cat_idx = categorical_column_indices(feature_names, feature_cfg)
    models: Dict[int, lgb.LGBMRegressor] = {}

    for cp in CHECKPOINTS:
        y = train_wide[f"cp_{cp}"].astype(float).values
        cfg = best_params_by_cp[cp]
        n_est = best_iter_by_cp[cp]
        model = lgb.LGBMRegressor(
            n_estimators=max(50, int(n_est)),
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            **cfg,
        )
        model.fit(X_train, y, categorical_feature=cat_idx)
        models[cp] = model

    return models, feature_names


# ---------------------------------------------------------------------------
# Prediction post-processing
# ---------------------------------------------------------------------------


def predict_raw_matrix(
    X: pd.DataFrame,
    models: Dict[int, lgb.LGBMRegressor],
    checkpoints: Sequence[int] = CHECKPOINTS,
) -> np.ndarray:
    """Shape ``(n_rows, len(checkpoints))`` in checkpoint order."""
    mat = np.zeros((len(X), len(checkpoints)), dtype=float)
    for j, cp in enumerate(checkpoints):
        mat[:, j] = models[cp].predict(X)
    return mat


def postprocess_monotonic_clip(pred_mat: np.ndarray) -> np.ndarray:
    """Non-decreasing curve over checkpoints (far-out → arrival), clipped to [0, 1]."""
    out = np.maximum.accumulate(pred_mat, axis=1)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


@dataclass
class BookingCurveModelBundle:
    """Serializable training state for inference."""

    models: Dict[int, Any]
    feature_config: FeatureConfig
    feature_names: List[str]
    category_levels: Dict[str, List[str]]
    reference_min_date: str
    checkpoints: List[int] = field(default_factory=lambda: list(CHECKPOINTS))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> BookingCurveModelBundle:
        with open(path, "rb") as f:
            return pickle.load(f)


def load_metadata(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    with open(data_dir / "property_metadata.json", encoding="utf-8") as f:
        return json.load(f)


def load_reservations(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(
        data_dir / "reservations.csv",
        parse_dates=["booking_date", "stay_date", "checkout_date"],
    )


def load_daily_event_impacts(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "daily_event_impacts.csv", parse_dates=["date"])


def build_prediction_row(
    property_id: str,
    room_type: str,
    stay_date: str,
    metadata: dict,
    daily_events_prepared: pd.DataFrame,
    reference_min_date: pd.Timestamp,
) -> pd.DataFrame:
    """Single-row frame for inference (same schema as training features)."""
    row = pd.DataFrame(
        [
            {
                "property_id": property_id,
                "room_type": room_type,
                "stay_date": stay_date,
            }
        ]
    )
    return enrich_curve_features(row, metadata, daily_events_prepared, reference_min_date)


def predict_booking_curve(
    property_id: str,
    room_type: str,
    stay_date: str,
    bundle: BookingCurveModelBundle,
    metadata: dict,
    daily_events_prepared: pd.DataFrame,
    *,
    as_of_date: str | None = None,
) -> dict:
    """
    Return predicted occupancy fractions at standard checkpoints.

    ``as_of_date`` is reserved for future point-in-time feature cuts; the current
    pipeline uses full calendar features consistent with training.
    """
    if as_of_date is not None:
        warnings.warn(
            "as_of_date is not used by the current feature pipeline; predictions use the same "
            "stay-date features as training.",
            UserWarning,
            stacklevel=2,
        )

    ref = pd.Timestamp(bundle.reference_min_date)
    row_df = build_prediction_row(
        property_id, room_type, stay_date, metadata, daily_events_prepared, ref
    )
    X, _ = prepare_X(row_df, bundle.feature_config, category_levels=bundle.category_levels)
    raw = predict_raw_matrix(X, bundle.models, bundle.checkpoints)
    fixed = postprocess_monotonic_clip(raw)
    preds = {str(cp): float(fixed[0, j]) for j, cp in enumerate(bundle.checkpoints)}
    return {
        "property_id": property_id,
        "room_type": room_type,
        "stay_date": pd.Timestamp(stay_date).strftime("%Y-%m-%d"),
        "predictions": preds,
    }
