"""
Tests for apis/commons.py

Covers:
- label_load()              – operating-zone classification (CB-aware)
- normalize3()              – normalization
- denormalize3()            – denormalization
- convert_timestamp()       – ISO string → pd.Timestamp
- percentage2severity()     – severity-level mapping
- fetch_between_dates()     – date-range query from SQLite
- fetch_last_rows()         – last-N-rows query
- get_LastdateLastRow()     – latest timestamp helper
- process_operationZone()   – zone distribution SQL
- process_operationMode()   – mode distribution SQL
- hampel_filter()           – outlier filter
- order_objects_by_keys()   – dict ordering helper
"""
import sys
import os
import sqlite3

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Django bootstrap (via conftest) must already have run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import apis.commons as commons


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kpi_db(tmp_path, n=100):
    """Create a minimal kpi.db with LGS1_timeline table."""
    db_path = str(tmp_path / "kpi.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS LGS1_timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT UNIQUE,
            active_power REAL,
            rpm REAL,
            cb REAL,
            aux REAL,
            aux_1 REAL
        )
    """)
    ts_base = datetime(2026, 1, 1)
    rows = [
        ((ts_base + timedelta(minutes=i)).isoformat(),
         50.0 + i * 0.1, 280.0, 1.0, 0.0, 0.0)
        for i in range(n)
    ]
    cur.executemany(
        "INSERT OR REPLACE INTO LGS1_timeline (timestamp, active_power, rpm, cb, aux, aux_1) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


def _make_sensor_db(tmp_path, feature_set, n=50):
    db_path = str(tmp_path / "original_data.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    col_defs = ", ".join([f'"{f.replace(" ", "_")}" REAL' for f in feature_set])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS original_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT UNIQUE,
            {col_defs}
        )
    """)
    ts_base = datetime(2026, 1, 1)
    col_list = ", ".join([f'"{f.replace(" ", "_")}"' for f in feature_set])
    placeholders = ", ".join(["?"] * (len(feature_set) + 1))
    sql = f"INSERT OR REPLACE INTO original_data (timestamp, {col_list}) VALUES ({placeholders})"
    rows = [
        ((ts_base + timedelta(minutes=i)).isoformat(), *[float(j) for j in range(len(feature_set))])
        for i in range(n)
    ]
    cur.executemany(sql, rows)
    conn.commit()
    conn.close()
    return db_path


# ===========================================================================
# label_load (CB-aware version in apis/commons.py)
# ===========================================================================

class TestLabelLoad:
    def _row(self, ap, rpm, cb):
        return {"Active Power": ap, "Governor speed actual": rpm, "CB": cb}

    def test_shutdown_when_cb_zero(self):
        assert commons.label_load(self._row(60, 280, 0)) == "Shutdown"

    def test_warming_low_rpm(self):
        assert commons.label_load(self._row(2, 50, 1)) == "Warming"

    def test_no_load_high_rpm(self):
        assert commons.label_load(self._row(2, 280, 1)) == "No Load"

    def test_low_load(self):
        assert commons.label_load(self._row(10, 280, 1)) == "Low Load"

    def test_rough_zone(self):
        assert commons.label_load(self._row(30, 280, 1)) == "Rough Zone"

    def test_part_load(self):
        assert commons.label_load(self._row(45, 280, 1)) == "Part Load"

    def test_efficient_load(self):
        assert commons.label_load(self._row(57, 280, 1)) == "Efficient Load"

    def test_high_load(self):
        assert commons.label_load(self._row(70, 280, 1)) == "High Load"

    def test_boundary_ap_20(self):
        assert commons.label_load(self._row(20, 280, 1)) == "Rough Zone"

    def test_cb_minus_one_is_shutdown(self):
        """Any CB value != 1 should be Shutdown."""
        assert commons.label_load(self._row(60, 280, -1)) == "Shutdown"


# ===========================================================================
# normalize3 / denormalize3
# ===========================================================================

class TestNormalizeDenormalize:
    def test_normalized_in_unit_range(self):
        a = np.array([[0.0, 100.0], [50.0, 200.0]])
        normed, min_a, max_a = commons.normalize3(a)
        assert np.all(normed >= -1e-6)

    def test_round_trip(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        normed, min_a, max_a = commons.normalize3(a)
        recovered = commons.denormalize3(normed, min_a, max_a)
        np.testing.assert_allclose(recovered, a, atol=1e-4)

    def test_explicit_min_max(self):
        a = np.array([[5.0], [10.0]])
        normed, _, _ = commons.normalize3(a, np.array([0.0]), np.array([10.0]))
        np.testing.assert_allclose(normed, [[0.4999], [0.9999]], atol=1e-3)

    def test_constant_array_no_nan(self):
        a = np.ones((10, 3))
        normed, _, _ = commons.normalize3(a)
        assert not np.any(np.isnan(normed))


# ===========================================================================
# convert_timestamp
# ===========================================================================

class TestConvertTimestamp:
    def test_returns_pandas_timestamp(self):
        assert isinstance(commons.convert_timestamp("2026-01-01T00:00:00"), pd.Timestamp)

    def test_correct_date(self):
        ts = commons.convert_timestamp("2026-03-07T12:30:00")
        assert ts.year == 2026 and ts.month == 3 and ts.day == 7

    def test_with_microseconds(self):
        ts = commons.convert_timestamp("2026-06-01T00:00:00.999999")
        assert ts.year == 2026


# ===========================================================================
# percentage2severity
# ===========================================================================

class TestPercentage2Severity:
    @pytest.mark.parametrize("value,expected", [
        (0,    1),
        (4.99, 1),
        (5,    2),
        (19.9, 2),
        (20,   3),
        (39.9, 3),
        (40,   4),
        (74.9, 4),
        (75,   5),
        (100,  5),
        (-1,   6),  # outside range → 6
        (101,  6),
    ])
    def test_thresholds(self, value, expected):
        assert commons.percentage2severity(value) == expected


# ===========================================================================
# fetch_between_dates
# ===========================================================================

class TestFetchBetweenDates:
    def test_returns_all_rows_in_range(self, tmp_path, feature_set):
        db_path = _make_sensor_db(tmp_path, feature_set, n=20)
        start = "2026-01-01T00:00:00"
        end   = "2026-01-01T00:20:00"
        result = commons.fetch_between_dates(start, end, db_path, "original_data", resampling=False)
        assert result.shape[0] == 20

    def test_empty_range_returns_empty_array(self, tmp_path, feature_set):
        db_path = _make_sensor_db(tmp_path, feature_set, n=10)
        start = "2025-01-01T00:00:00"
        end   = "2025-01-02T00:00:00"
        result = commons.fetch_between_dates(start, end, db_path, "original_data")
        assert len(result) == 0

    def test_resampling_reduces_rows(self, tmp_path, feature_set):
        db_path = _make_sensor_db(tmp_path, feature_set, n=500)
        start = "2026-01-01T00:00:00"
        end   = "2026-01-01T08:19:00"
        result_full = commons.fetch_between_dates(start, end, db_path, "original_data",
                                                   max_rows=500, resampling=False)
        result_sampled = commons.fetch_between_dates(start, end, db_path, "original_data",
                                                      max_rows=100, resampling=True)
        assert result_sampled.shape[0] <= result_full.shape[0]

    def test_returns_numpy_array(self, tmp_path, feature_set):
        db_path = _make_sensor_db(tmp_path, feature_set, n=5)
        start = "2026-01-01T00:00:00"
        end   = "2026-01-01T00:10:00"
        result = commons.fetch_between_dates(start, end, db_path, "original_data", resampling=False)
        assert isinstance(result, np.ndarray)


# ===========================================================================
# fetch_last_rows
# ===========================================================================

class TestFetchLastRows:
    def test_returns_n_rows(self, tmp_path, feature_set):
        db_path = _make_sensor_db(tmp_path, feature_set, n=50)
        result = commons.fetch_last_rows(10, db_path, "original_data")
        assert result.shape[0] == 10

    def test_returns_most_recent_rows(self, tmp_path, feature_set):
        db_path = _make_sensor_db(tmp_path, feature_set, n=20)
        result = commons.fetch_last_rows(5, db_path, "original_data")
        # Timestamps should be among the last 5 (i.e. later than the 15th)
        ts_last = pd.to_datetime(result[:, 1])
        ts_threshold = datetime(2026, 1, 1, 0, 14, 0)
        assert all(t >= pd.Timestamp(ts_threshold) for t in ts_last)

    def test_empty_table_returns_empty_array(self, tmp_path, feature_set):
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS data (id INTEGER, timestamp TEXT)")
        conn.commit()
        conn.close()
        result = commons.fetch_last_rows(5, db_path, "data")
        assert len(result) == 0


# ===========================================================================
# get_LastdateLastRow
# ===========================================================================

class TestGetLastdateLastRow:
    def test_returns_datetime(self, tmp_path, feature_set):
        db_path = _make_sensor_db(tmp_path, feature_set, n=10)
        result = commons.get_LastdateLastRow(db_path)
        assert isinstance(result, datetime)

    def test_returns_correct_last_timestamp(self, tmp_path, feature_set):
        db_path = _make_sensor_db(tmp_path, feature_set, n=10)
        result = commons.get_LastdateLastRow(db_path)
        expected = datetime(2026, 1, 1, 0, 9, 0)
        assert result == expected


# ===========================================================================
# process_operationZone
# ===========================================================================

class TestProcessOperationZone:
    def test_returns_list_of_tuples(self, tmp_path):
        db_path = _make_kpi_db(tmp_path, n=50)
        result = commons.process_operationZone(
            "2026-01-01T00:00:00", "2026-01-01T01:00:00",
            db_name=db_path, table_name="LGS1_timeline"
        )
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2  # (label, count)

    def test_all_high_load_returns_high_load_label(self, tmp_path):
        db_path = _make_kpi_db(tmp_path, n=30)
        result = commons.process_operationZone(
            "2026-01-01T00:00:00", "2026-01-01T00:30:00",
            db_name=db_path, table_name="LGS1_timeline"
        )
        labels = [r[0] for r in result]
        assert "High Load" in labels or "Efficient Load" in labels  # 50+ MW → High or Efficient


# ===========================================================================
# process_operationMode
# ===========================================================================

class TestProcessOperationMode:
    def test_returns_list(self, tmp_path):
        db_path = _make_kpi_db(tmp_path, n=20)
        result = commons.process_operationMode(
            "2026-01-01T00:00:00", "2026-01-01T00:20:00",
            db_name=db_path, table_name="LGS1_timeline"
        )
        assert isinstance(result, list)

    def test_all_zeros_returns_grid(self, tmp_path):
        db_path = _make_kpi_db(tmp_path, n=10)
        result = commons.process_operationMode(
            "2026-01-01T00:00:00", "2026-01-01T00:10:00",
            db_name=db_path, table_name="LGS1_timeline"
        )
        labels = [r[0] for r in result]
        assert "Grid" in labels


# ===========================================================================
# hampel_filter
# ===========================================================================

class TestHampelFilter:
    def test_output_same_shape(self):
        series = np.random.randn(100)
        result = commons.hampel_filter(series, window_size=3, n_sigmas=3)
        assert result.shape == series.shape

    def test_replaces_spike_with_median(self):
        series = np.ones(50)
        series[25] = 1000.0  # big spike
        result = commons.hampel_filter(series, window_size=5, n_sigmas=3)
        assert result[25] < 1000.0, "Spike was not reduced by hampel filter"

    def test_no_outliers_unchanged(self):
        series = np.ones(50)
        result = commons.hampel_filter(series, window_size=3, n_sigmas=3)
        np.testing.assert_array_equal(result, series)


# ===========================================================================
# order_objects_by_keys
# ===========================================================================

class TestOrderObjectsByKeys:
    def test_basic_ordering(self):
        data = {"b": 2, "a": 1, "c": 3}
        result = commons.order_objects_by_keys(data, ["a", "b", "c"])
        assert list(result.keys()) == ["a", "b", "c"]

    def test_missing_key_skipped(self):
        data = {"x": 10, "y": 20}
        result = commons.order_objects_by_keys(data, ["y", "z", "x"])
        assert list(result.keys()) == ["y", "x"]

    def test_empty_order_returns_empty(self):
        data = {"a": 1}
        result = commons.order_objects_by_keys(data, [])
        assert result == {}
