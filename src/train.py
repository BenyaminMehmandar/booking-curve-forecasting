"""
Train booking-curve models: CV tuning, full-window refit, save artifact, optional test JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model import (
    BookingCurveModelBundle,
    CHECKPOINTS,
    FeatureConfig,
    build_actual_curves,
    category_levels_from_train,
    enrich_curve_features,
    expand_actuals_to_wide,
    fit_per_checkpoint_models,
    load_daily_event_impacts,
    load_reservations,
    prepare_daily_events,
    prepare_X,
    postprocess_monotonic_clip,
    predict_raw_matrix,
    split_train_test_baseline,
    tune_checkpoints_with_cv,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM booking-curve models.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_project_root() / "data",
        help="Directory with reservations.csv, property_metadata.json, daily_event_impacts.csv",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        # default=_project_root() / "Submission" / "artifacts" / "booking_curve_bundle.pkl",
        default=_project_root() / "artifacts" / "booking_curve_bundle.pkl",
        help="Where to save the fitted BookingCurveModelBundle",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="Optional path to write test-window predictions JSON (baseline schema)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    print("Loading data...")
    reservations = load_reservations(data_dir)
    with open(data_dir / "property_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    daily_raw = load_daily_event_impacts(data_dir)
    daily_prepared = prepare_daily_events(daily_raw)

    print("Building actual curves...")
    all_curves = build_actual_curves(reservations, metadata)
    reference_min_date = pd.Timestamp(all_curves["stay_date"].min())

    print("Feature engineering...")
    feat_base = enrich_curve_features(all_curves, metadata, daily_prepared, reference_min_date)
    feat_wide = expand_actuals_to_wide(feat_base)

    train_wide, test_wide = split_train_test_baseline(feat_wide)
    print(f"Train rows: {len(train_wide):,} | Test rows: {len(test_wide):,}")

    feature_cfg = FeatureConfig()
    cat_levels = category_levels_from_train(train_wide, feature_cfg)

    print("CV tuning (per checkpoint)...")
    tuning_df, best_params_by_cp, best_iter_by_cp = tune_checkpoints_with_cv(
        train_wide, feature_cfg, cat_levels
    )
    print(tuning_df.to_string(index=False))

    print("Refitting on full training window...")
    models, feature_names = fit_per_checkpoint_models(
        train_wide, feature_cfg, cat_levels, best_params_by_cp, best_iter_by_cp
    )

    bundle = BookingCurveModelBundle(
        models=models,
        feature_config=feature_cfg,
        feature_names=feature_names,
        category_levels=cat_levels,
        reference_min_date=reference_min_date.strftime("%Y-%m-%d"),
    )
    bundle.save(args.artifact_path)
    print(f"Saved bundle to {args.artifact_path.resolve()}")

    if args.predictions_out is not None:
        print("Generating test predictions...")
        X_test, _ = prepare_X(test_wide, feature_cfg, category_levels=cat_levels)
        raw = predict_raw_matrix(X_test, models)
        fixed = postprocess_monotonic_clip(raw)

        predictions = []
        for i, row in test_wide.reset_index(drop=True).iterrows():
            preds = {str(cp): float(fixed[i, j]) for j, cp in enumerate(CHECKPOINTS)}
            predictions.append(
                {
                    "property_id": row["property_id"],
                    "room_type": row["room_type"],
                    "stay_date": pd.Timestamp(row["stay_date"]).strftime("%Y-%m-%d"),
                    "predictions": preds,
                }
            )
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.predictions_out, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)
        print(f"Wrote {len(predictions):,} rows to {args.predictions_out.resolve()}")


if __name__ == "__main__":
    main()
