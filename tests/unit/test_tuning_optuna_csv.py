import csv
import datetime as dt
from pathlib import Path

import pytest

from src.tuning.optuna_csv import (
    TRIALS_CSV_HEADER,
    TUNED_PARAM_KEYS,
    compatibility_mismatches,
    load_rows,
    load_previous_best,
    row_dt,
    row_value,
)


class TestRowValue:
    def test_returns_float_value(self):
        assert row_value({"value": "0.5"}) == 0.5
        assert row_value({"value": "1.0"}) == 1.0

    def test_returns_inf_when_missing(self):
        assert row_value({}) == float("inf")
        assert row_value({"other": "x"}) == float("inf")

    def test_returns_inf_on_invalid(self):
        assert row_value({"value": "nope"}) == float("inf")


class TestRowDt:
    def test_uses_session_started_at(self):
        r = {"session_started_at": "2026-02-08T12:00:00"}
        assert row_dt(r) == dt.datetime(2026, 2, 8, 12, 0, 0)

    def test_falls_back_to_exported_at(self):
        r = {"exported_at": "2026-02-07T10:30:00"}
        assert row_dt(r) == dt.datetime(2026, 2, 7, 10, 30, 0)

    def test_returns_min_when_no_valid_timestamp(self):
        assert row_dt({}) == dt.datetime.min
        assert row_dt({"session_started_at": ""}) == dt.datetime.min


class TestCompatibilityMismatches:
    def test_empty_when_identical(self):
        meta = {"mode": "dev", "proximity_token": "proximity20"}
        assert compatibility_mismatches(meta, meta) == {}

    def test_returns_mismatched_keys(self):
        prev = {"mode": "dev", "proximity_token": "proximity10"}
        cur = {"mode": "dev", "proximity_token": "proximity20"}
        got = compatibility_mismatches(prev, cur)
        assert "proximity_token" in got
        assert got["proximity_token"] == {"previous": "proximity10", "current": "proximity20"}

    def test_ignores_keys_not_in_compatibility_list(self):
        prev = {"mode": "dev", "extra": "a"}
        cur = {"mode": "dev", "extra": "b"}
        assert "extra" not in compatibility_mismatches(prev, cur)


class TestLoadRows:
    def test_returns_empty_for_missing_file(self, tmp_path):
        assert load_rows(tmp_path / "missing.csv") == []

    def test_parses_csv(self, tmp_path):
        path = tmp_path / "trials.csv"
        path.write_text(
            "trial_number,state,value\n1,COMPLETE,0.5\n2,COMPLETE,0.3\n",
            encoding="utf-8",
        )
        rows = load_rows(path)
        assert len(rows) == 2
        assert rows[0]["state"] == "COMPLETE" and rows[0]["value"] == "0.5"


class TestLoadPreviousBest:
    def test_returns_none_for_missing_file(self, tmp_path):
        assert load_previous_best(tmp_path / "missing.csv") is None

    def test_returns_best_complete_row(self, tmp_path):
        path = tmp_path / "trials.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["trial_number", "state", "value"])
            w.writeheader()
            w.writerow({"trial_number": "1", "state": "COMPLETE", "value": "0.5"})
            w.writerow({"trial_number": "2", "state": "COMPLETE", "value": "0.3"})
        best = load_previous_best(path)
        assert best is not None
        assert best["value"] == "0.3"

    def test_returns_none_when_no_complete(self, tmp_path):
        path = tmp_path / "trials.csv"
        path.write_text("trial_number,state,value\n1,PRUNED,0.5\n", encoding="utf-8")
        assert load_previous_best(path) is None


class TestConstants:
    def test_tuned_param_keys_contains_expected(self):
        assert "learning_rate" in TUNED_PARAM_KEYS
        assert "batch_size" in TUNED_PARAM_KEYS

    def test_trials_csv_header_contains_tuned_params(self):
        for k in TUNED_PARAM_KEYS:
            assert k in TRIALS_CSV_HEADER
