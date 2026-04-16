"""
Load a trained bundle and expose ``predict_booking_curve`` for inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from model import (
    BookingCurveModelBundle,
    load_daily_event_impacts,
    load_metadata,
    prepare_daily_events,
    predict_booking_curve as run_predict,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_data_dir() -> Path:
    return Path(os.environ.get("BOOKING_CURVE_DATA_DIR", _project_root() / "data"))


def default_bundle_path() -> Path:
    env = os.environ.get("BOOKING_CURVE_BUNDLE_PATH")
    if env:
        return Path(env)
    # return _project_root() / "Submission" / "artifacts" / "booking_curve_bundle.pkl"
    return _project_root() / "artifacts" / "booking_curve_bundle.pkl"


# Lazy caches for interactive / repeated calls
_bundle: BookingCurveModelBundle | None = None
_metadata: dict | None = None
_daily_prepared: Any = None
_cached_bundle_path: Path | None = None
_cached_data_dir: Path | None = None


def _load_context(
    bundle_path: Path | None = None,
    data_dir: Path | None = None,
) -> tuple[BookingCurveModelBundle, dict, Any]:
    global _bundle, _metadata, _daily_prepared, _cached_bundle_path, _cached_data_dir
    bp = (bundle_path or default_bundle_path()).resolve()
    dd = (data_dir or default_data_dir()).resolve()

    if _bundle is not None and _cached_bundle_path == bp and _cached_data_dir == dd:
        assert _metadata is not None and _daily_prepared is not None
        return _bundle, _metadata, _daily_prepared

    _bundle = BookingCurveModelBundle.load(bp)
    _metadata = load_metadata(dd)
    _daily_prepared = prepare_daily_events(load_daily_event_impacts(dd))
    _cached_bundle_path = bp
    _cached_data_dir = dd
    return _bundle, _metadata, _daily_prepared


def predict_booking_curve(
    property_id: str,
    room_type: str,
    stay_date: str,
    as_of_date: str | None = None,
    *,
    bundle_path: Path | None = None,
    data_dir: Path | None = None,
) -> dict:
    """
    Returns predicted occupancy fractions at standard checkpoints.

    Example output:
        {
            "property_id": "property_A",
            "room_type": "standard_king",
            "stay_date": "2025-08-15",
            "predictions": {"90": 0.05, "60": 0.15, ...},
        }

    Requires the trained artifact from ``train.py`` and the same ``data`` directory
    used for exogenous merges (e.g. ``daily_event_impacts.csv``) and property metadata.
    """
    bundle, metadata, daily_prepared = _load_context(bundle_path=bundle_path, data_dir=data_dir)
    return run_predict(
        property_id,
        room_type,
        stay_date,
        bundle,
        metadata,
        daily_prepared,
        as_of_date=as_of_date,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one booking curve.")
    parser.add_argument("--property-id", required=True)
    parser.add_argument("--room-type", required=True)
    parser.add_argument("--stay-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--as-of-date", default=None, help="Optional; reserved for future use")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--bundle-path", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print one JSON object to stdout")
    args = parser.parse_args()

    out = predict_booking_curve(
        args.property_id,
        args.room_type,
        args.stay_date,
        as_of_date=args.as_of_date,
        bundle_path=args.bundle_path,
        data_dir=args.data_dir,
    )
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(out)


if __name__ == "__main__":
    main()
