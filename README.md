# Submission: ML Booking-Curve Model

This document describes the **end-to-end solution**: how curves are built from reservations, how features and models are trained, what is saved to disk, and how to run training, prediction, and evaluation.

**Python:** 3.10+ recommended.

---

## Problem (brief)

For each **property**, **room type**, and **stay date**, we predict the **cumulative occupancy fraction** (rooms booked ÷ inventory) at fixed **checkpoints** before the stay: **90, 60, 45, 30, 21, 14, 7, 3, 1, 0** days out. Values are in **[0, 1]** and should form a **non-decreasing** curve from far-out to arrival.

---

## Approach

1. **Reconstruct historical curves** from `reservations.csv` using the same rules as the provided baseline (nights covered by each reservation, checkpoints defined by booking cutoffs).
2. **Engineer features** per `(property_id, room_type, stay_date)` row (calendar, cyclical encodings, simple holiday proximity, room/property capacity, merge of area-level `daily_event_impacts.csv`).
3. **Train ten gradient-boosted regressors** (LightGBM), **one per checkpoint**, on the training date range. Each model predicts that checkpoint’s occupancy fraction from the **same** feature row.
4. **Tune** with **time-based cross-validation** inside the training window and a **small hyperparameter grid**, using **early stopping** on validation folds.
5. **Refit** each checkpoint model on the **full** training window with the chosen settings and a tree count derived from CV (median best iteration per checkpoint).
6. **Save** all models plus metadata needed for inference in **one serialized bundle** (`pickle`).
7. At **prediction time**, build features for a single stay, run all ten models, then **post-process**: enforce **monotonicity** (cumulative maximum along checkpoint order) and **clip** to **[0, 1]**.

---

## Repository layout

| Path | Role |
|------|------|
| `src/model.py` | Curve construction, feature engineering, CV/refit helpers, `BookingCurveModelBundle`, core prediction utilities, data loaders. |
| `src/train.py` | CLI: load data → curves → features → CV tuning → refit → **save bundle**; optional test-set JSON. |
| `src/predict.py` | CLI and **`predict_booking_curve(...)`** for inference (loads bundle + metadata + daily impacts). |
| `artifacts/booking_curve_bundle.pkl` | **Created by training** (not committed unless you choose to). Holds 10 fitted models and training metadata. |

Project **data** are expected at the repository root: `data/` (see below). Paths are resolved relative to the repo layout described in the scripts.

---

## Data assumptions and sources

Training and prediction assume the assignment layout:

| File | Use in this solution |
|------|----------------------|
| `data/reservations.csv` | Build **actual** curves (non-cancelled bookings only for curve construction). |
| `data/property_metadata.json` | **Inventory** per room type and property totals; required for curve denominators and capacity features. |
| `data/daily_event_impacts.csv` | **Merged** on `stay_date` for feature parity with training (see “Feature note” below). |

Files such as `data/events.csv` or `data/daily_occupancy.csv` are **not** used by this pipeline (they may be useful for extensions).

### Curve construction (aligned with baseline)

- A reservation counts toward a **target night** if it **covers** that night: `stay_date <= target_night < checkout_date`.
- **Cancelled** reservations are **excluded** from curve construction.
- Occupancy at checkpoint \(d\) days before stay: fraction of inventory with `booking_date <= target_night - d days`, capped at 1.

### Train / test split (evaluation)

Consistent with the assignment and `starter/baseline_model.py`:

- **Train:** stay dates **≤ 2025-06-30**
- **Test:** stay dates **2025-07-01** through **2025-09-30**

### Feature note (merged vs modeled columns)

`daily_event_impacts.csv` is **joined** so the feature table matches the training pipeline used during development. The **current** `FeatureConfig` uses **calendar, holiday, and capacity** fields only (the merged event columns are present in the frame but **not** included as model inputs). Adding event-derived numeric columns is a one-line change in `FeatureConfig` in `model.py` if you want them to contribute.

### `as_of_date` (optional API parameter)

`predict_booking_curve(..., as_of_date=None)` is part of the required interface. The **current** implementation does **not** change features by `as_of_date`; it issues a **warning** if provided. A production **point-in-time** model would restrict reservations and calendars to information available as of `as_of_date`.

### Reference date for trend feature

`days_since_data_start` uses the **minimum stay date** in the **full** curve table built from the training run, stored in the bundle as `reference_min_date`. This keeps the feature **consistent** between batch training and single-row prediction.

---

## How training works (`train.py`)

1. Load `reservations.csv`, `property_metadata.json`, `daily_event_impacts.csv`.
2. `build_actual_curves` → one row per property / room type / stay night with nested `actuals` (checkpoint → fraction).
3. `enrich_curve_features` → wide features; `expand_actuals_to_wide` → `cp_*` label columns.
4. Split into train / test by stay date.
5. **Category levels** for `property_id` and `room_type` are taken from the **training** split so inference uses the same categorical encoding.
6. **Time-based CV** (three validation windows inside the training period) × **small grid** of LightGBM settings; **early stopping** per fold; pick best config by **mean MAE** across folds per checkpoint.
7. **Refit** on all training rows with best config; `n_estimators` = `max(50, median best iteration from CV)` per checkpoint.
8. Save `BookingCurveModelBundle` to disk (default path below).
9. Optionally write **test-window** predictions JSON for `evaluation/evaluate.py`.

### Default paths

- Data: `<repo_root>/data`
- Bundle: `<repo_root>/artifacts/booking_curve_bundle.pkl`

### Example

```bash
# From repository root
cd /src
pip install -r ../requirements.txt

python train.py
```

Optional: write test predictions for evaluation:

```bash
python train.py --predictions-out ../../Predictions/my_predictions.json
```

---

## What gets saved (the bundle)

The artifact is a **single pickle file** containing:

- **`models`:** `dict` mapping each checkpoint (e.g. `90`, `60`, …) to a fitted **`LGBMRegressor`** — **10 models** total.
- **`feature_config`:** Names of numeric and categorical inputs.
- **`feature_names`:** Column order used at fit time.
- **`category_levels`:** Allowed categories for `property_id` and `room_type` (must match training for LightGBM).
- **`reference_min_date`:** ISO date string for `days_since_data_start`.
- **`checkpoints`:** List of checkpoint days (same as global `CHECKPOINTS`).

**Inference requires** this file plus **`property_metadata.json`** and **`daily_event_impacts.csv`** (for the same merges as training).

---

## How prediction works (`predict.py`)

1. Load **`BookingCurveModelBundle`** from the default path or `BOOKING_CURVE_BUNDLE_PATH`.
2. Load **`property_metadata.json`** and **`daily_event_impacts.csv`** from the data directory (`BOOKING_CURVE_DATA_DIR` or `<repo_root>/data`).
3. Build **one row** of engineered features for `(property_id, room_type, stay_date)`.
4. `prepare_X` with stored **category levels**.
5. For each checkpoint, **`models[cp].predict(X)`** — one vector of length 10 per row.
6. **Post-processing:** `np.maximum.accumulate` along checkpoint order (far-out → arrival), then **clip** to [0, 1].

### Public API

```python
from predict import predict_booking_curve

out = predict_booking_curve(
    property_id="property_A",
    room_type="standard_king",
    stay_date="2025-08-15",
)
# out["predictions"] -> {"90": float, "60": float, ...}
```

Optional keyword-only overrides: `bundle_path=`, `data_dir=` (useful for tests or custom deployments).

### CLI

```bash
cd /src
python predict.py --property-id property_A --room-type standard_king --stay-date 2025-08-15 --json
```

Environment variables:

- **`BOOKING_CURVE_BUNDLE_PATH`:** Path to the pickle bundle.
- **`BOOKING_CURVE_DATA_DIR`:** Directory containing `property_metadata.json` and `daily_event_impacts.csv`.

---

## Evaluation

After generating a predictions JSON (same schema as the baseline: list of objects with `property_id`, `room_type`, `stay_date`, `predictions`):

```bash
python evaluation/evaluate.py --predictions path/to/predictions.json
```

Use the **test window** (Jul–Sep 2025 stay dates) for scores comparable to the assignment.

---

## Design choices (summary)

| Topic | Choice |
|-------|--------|
| **Why ten models?** | Each checkpoint has a different label distribution; separate heads are simple and match the baseline evaluation structure. |
| **Monotonicity** | Enforced **after** prediction; the base learners do not guarantee monotone curves by construction. |
| **Leakage** | Train/test split by **stay date**; CV folds are **inside** the training period only. |
| **Serialization** | **`pickle`** for a single bundle (standard library; avoids extra runtime deps beyond sklearn/lightgbm). |

---

## Troubleshooting

- **`ModuleNotFoundError` (e.g. lightgbm):** Install from `requirements.txt`.
- **Categorical / prediction error:** `property_id` and `room_type` must be among **training** category levels; new categories require retraining or explicit handling.
- **Missing bundle:** Run `train.py` first so `/artifacts/booking_curve_bundle.pkl` exists (or set `BOOKING_CURVE_BUNDLE_PATH`).

---

## Reproducibility

- LightGBM and sklearn use **`random_state=42`** where applicable.
- CV fold definitions and the grid are **fixed in `model.py`** so runs are repeatable for the same data and code version.
