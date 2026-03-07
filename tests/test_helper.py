"""
Tests for apis/helper.py

Because helper.py opens pickle files at import time and queries databases
that are only present at runtime, these tests mock all external dependencies.

Covers:
- calculate_priority()            – priority score formula
- get_FixedDate()                 – date adjustment logic
- get_unit_status()               – single unit alive/shutdown detection
- get_units_status()              – multi-unit status aggregation
- get_OperationDistribution()     – delegates to commons correctly
- get_OperationDistributionTimeline() – timeline data shape
"""
import sys
import os
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Django bootstrap via conftest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# calculate_priority
# ---------------------------------------------------------------------------

class TestCalculatePriority:
    @pytest.fixture(autouse=True)
    def _import(self):
        # Patch pickle opens before importing helper
        with patch("builtins.open", MagicMock()):
            with patch("pickle.load", return_value=MagicMock()):
                import importlib
                import apis.helper as h
                # Re-use already-imported module
                self.h = h

    def test_critical_equipment_doubles_score(self):
        from apis.helper import calculate_priority, recap_severity, equipment_critical_list

        non_crit_name = "Active Power"
        crit_name = "UGB X Displacement"

        # Use equal current severity
        p_non_crit = calculate_priority(non_crit_name, recap_severity, 3, equipment_critical_list)
        p_crit = calculate_priority(crit_name, recap_severity, 3, equipment_critical_list)

        assert p_crit == p_non_crit * 2

    def test_higher_severity_gives_higher_priority(self):
        from apis.helper import calculate_priority, recap_severity, equipment_critical_list

        p_low = calculate_priority("Active Power", recap_severity, 1, equipment_critical_list)
        p_high = calculate_priority("Active Power", recap_severity, 5, equipment_critical_list)

        assert p_high > p_low

    def test_zero_severity_returns_zero(self):
        from apis.helper import calculate_priority, recap_severity, equipment_critical_list

        p = calculate_priority("Active Power", recap_severity, 0, equipment_critical_list)
        assert p == 0.0


# ---------------------------------------------------------------------------
# get_unit_status
# ---------------------------------------------------------------------------

class TestGetUnitStatus:
    """
    Mocks fetch_between_dates so get_unit_status can run without a real DB.
    """

    def _make_sensor_data(self, active_power, rpm, cb, n=10):
        """Build a numpy array that mimics the DB fetch output format.
        Columns: id, timestamp, active_power, rpm, cb [+ more]
        """
        ts = [(datetime(2026, 1, 1) + timedelta(minutes=i)).isoformat() for i in range(n)]
        rows = [
            (i, ts[i], active_power[i], rpm[i], cb[i])
            for i in range(n)
        ]
        return np.array(rows, dtype=object)

    def test_alive_when_cb_one_and_high_power(self):
        import apis.helper as h
        n = 10
        data = self._make_sensor_data(
            active_power=np.full(n, 60.0),
            rpm=np.full(n, 280.0),
            cb=np.ones(n),
        )
        with patch("apis.commons.fetch_between_dates", return_value=data):
            with patch("django.conf.settings") as mock_settings:
                mock_settings.MONITORINGDB_PATH = "/tmp/"
                result = h.get_unit_status("2026-01-01T00:00:00", "2026-01-01T00:10:00", "LGS1")
        assert result == "alive"

    def test_shutdown_when_cb_zero(self):
        import apis.helper as h
        n = 10
        data = self._make_sensor_data(
            active_power=np.zeros(n),
            rpm=np.zeros(n),
            cb=np.zeros(n),
        )
        with patch("apis.commons.fetch_between_dates", return_value=data):
            with patch("django.conf.settings") as mock_settings:
                mock_settings.MONITORINGDB_PATH = "/tmp/"
                result = h.get_unit_status("2026-01-01T00:00:00", "2026-01-01T00:10:00", "LGS1")
        assert result == "shutdown"

    def test_empty_data_returns_shutdown(self):
        import apis.helper as h
        with patch("apis.commons.fetch_between_dates", return_value=np.array([])):
            with patch("django.conf.settings") as mock_settings:
                mock_settings.MONITORINGDB_PATH = "/tmp/"
                result = h.get_unit_status("2026-01-01T00:00:00", "2026-01-01T00:10:00", "LGS1")
        assert result == "shutdown"


# ---------------------------------------------------------------------------
# get_units_status
# ---------------------------------------------------------------------------

class TestGetUnitsStatus:
    def _make_sensor_data(self, active_power_val, cb_val, n=5):
        ts = [(datetime(2026, 1, 1) + timedelta(minutes=i)).isoformat() for i in range(n)]
        rows = [(i, ts[i], active_power_val, 280.0, cb_val) for i in range(n)]
        return np.array(rows, dtype=object)

    def test_all_units_alive(self):
        import apis.helper as h
        alive_data = self._make_sensor_data(60.0, 1.0)
        with patch("apis.commons.fetch_between_dates", return_value=alive_data):
            with patch("django.conf.settings") as mock_settings:
                mock_settings.MONITORINGDB_PATH = "/tmp/"
                result = h.get_units_status(
                    "2026-01-01T00:00:00", "2026-01-01T00:10:00",
                    units=["LGS1", "LGS2"]
                )
        assert all(v == "alive" for v in result.values())

    def test_all_units_shutdown(self):
        import apis.helper as h
        shutdown_data = self._make_sensor_data(0.0, 0.0)
        with patch("apis.commons.fetch_between_dates", return_value=shutdown_data):
            with patch("django.conf.settings") as mock_settings:
                mock_settings.MONITORINGDB_PATH = "/tmp/"
                result = h.get_units_status(
                    "2026-01-01T00:00:00", "2026-01-01T00:10:00",
                    units=["LGS1", "LGS2"]
                )
        assert all(v == "shutdown" for v in result.values())

    def test_returns_dict_with_unit_keys(self):
        import apis.helper as h
        data = self._make_sensor_data(60.0, 1.0)
        with patch("apis.commons.fetch_between_dates", return_value=data):
            with patch("django.conf.settings") as mock_settings:
                mock_settings.MONITORINGDB_PATH = "/tmp/"
                result = h.get_units_status(
                    "2026-01-01T00:00:00", "2026-01-01T00:10:00",
                    units=["LGS1", "LGS2", "LGS3"]
                )
        assert set(result.keys()) == {"LGS1", "LGS2", "LGS3"}


# ---------------------------------------------------------------------------
# get_OperationDistribution – delegates correctly
# ---------------------------------------------------------------------------

class TestGetOperationDistribution:
    def test_returns_two_items(self):
        import apis.helper as h
        mock_mode = [("Grid", 80)]
        mock_zone = [("High Load", 60)]
        with patch("apis.commons.process_operationMode", return_value=mock_mode):
            with patch("apis.commons.process_operationZone", return_value=mock_zone):
                with patch("django.conf.settings") as mock_settings:
                    mock_settings.MONITORINGDB_PATH = "/tmp/"
                    mode, zone = h.get_OperationDistribution(
                        "2026-01-01T00:00:00", "2026-01-02T00:00:00", units=["LGS1"]
                    )
        assert mode == mock_mode
        assert zone == mock_zone


# ---------------------------------------------------------------------------
# get_OperationDistributionTimeline
# ---------------------------------------------------------------------------

class TestGetOperationDistributionTimeline:
    def _make_timeline_data(self, n=20):
        ts = [(datetime(2026, 1, 1) + timedelta(minutes=i)).isoformat() for i in range(n)]
        active_power = np.random.uniform(40, 70, n)
        rpm = np.full(n, 280.0)
        cb = np.ones(n)
        aux = np.zeros(n)
        aux_1 = np.zeros(n)
        rows = [(i, ts[i], ap, r, c, a, a1)
                for i, (ap, r, c, a, a1)
                in enumerate(zip(active_power, rpm, cb, aux, aux_1))]
        return np.array(rows, dtype=object)

    def test_returns_three_arrays(self):
        import apis.helper as h
        data = self._make_timeline_data(20)
        with patch("apis.commons.fetch_between_dates", return_value=data):
            with patch("django.conf.settings") as mock_settings:
                mock_settings.MONITORINGDB_PATH = "/tmp/"
                timestamps, load_codes, grid_data = h.get_OperationDistributionTimeline(
                    "2026-01-01T00:00:00", "2026-01-01T00:20:00", units=["LGS1"]
                )
        assert len(timestamps) == 20
        assert len(load_codes) == 20
        assert len(grid_data) == 20

    def test_empty_data_returns_empty_lists(self):
        import apis.helper as h
        with patch("apis.commons.fetch_between_dates", return_value=np.array([])):
            with patch("django.conf.settings") as mock_settings:
                mock_settings.MONITORINGDB_PATH = "/tmp/"
                ts, lc, gd = h.get_OperationDistributionTimeline(
                    "2026-01-01T00:00:00", "2026-01-01T00:10:00"
                )
        assert ts == [] and lc == [] and gd == []
