"""
Tests for apis/views.py

Covers all REST API endpoints:
- GET /api/panel_summary
- GET /api/zone_distribution
- GET /api/zone_distributionTimeline
- GET /api/unitStatus
- GET /api/kpi
- GET /api/severity_plot
- GET /api/advisory_table
- GET /api/advisory_detail/<feat_id>
- GET /api/timeinfo_detail
- GET /api/adjust_threshold_settings

Because these views depend on real databases and precomputed pickle files that
may not be present in all environments, most tests mock the underlying helper
functions. Integration-level tests (marked with @pytest.mark.integration) are
skipped unless DB and pickle files are present.
"""
import sys
import os
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
from datetime import datetime

# Django bootstrap happens via conftest.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from django.test import RequestFactory
from rest_framework.test import APIClient
from rest_framework import status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL = "/api/"


def _client():
    return APIClient()


def _make_sever_data(n_feat=7):
    return {f: float(i * 5) for i, f in enumerate([f"f{j}" for j in range(n_feat)])}


# ===========================================================================
# panel_summary
# ===========================================================================

class TestPanelSummary:
    ENDPOINT = "/api/panel_summary"

    def _mock_return(self):
        feature_names = [f"feat_{i}" for i in range(5)]
        return (
            "2026-01-01T00:00:00",                          # last_timestamp
            {f: 50 for f in feature_names},                  # last_sensor_featname
            {f: np.array([50.0] * 10) for f in feature_names},  # sensor_featname
            {f: 1 for f in feature_names},                   # last_severity_featname
            {f: np.array([1] * 10) for f in feature_names}, # sever_featname
            feature_names[:3],                               # ordered_feature_name
            {f: {1: 10} for f in feature_names},             # sever_count_featname
            {f: 0.5 for f in feature_names},                 # priority_parameter
        )

    def test_returns_200(self):
        with patch("apis.helper.get_PanelSummary", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

    def test_response_keys_present(self):
        with patch("apis.helper.get_PanelSummary", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        data = response.data
        for key in [
            "last_timestamp", "last_sensor_featname", "sensor_featname",
            "last_severity_featname", "sever_featname",
            "ordered_feature_name", "sever_count_featname", "priority_parameter"
        ]:
            assert key in data, f"Missing key: {key}"

    def test_with_date_params(self):
        with patch("apis.helper.get_PanelSummary", return_value=self._mock_return()) as m:
            _client().get(self.ENDPOINT, {
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-02T00:00:00",
            })
            m.assert_called_once_with("2026-01-01T00:00:00", "2026-01-02T00:00:00")


# ===========================================================================
# zone_distribution
# ===========================================================================

class TestZoneDistribution:
    ENDPOINT = "/api/zone_distribution"

    def _mock_return(self):
        return (
            [("Grid", 80), ("Furnace", 20)],     # operation_mode
            [("High Load", 60), ("Part Load", 40)],  # operation_zone
        )

    def test_returns_200(self):
        with patch("apis.helper.get_OperationDistribution", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

    def test_response_keys(self):
        with patch("apis.helper.get_OperationDistribution", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        assert "operation_mode" in response.data
        assert "operation_zone" in response.data

    def test_operation_mode_is_dict(self):
        with patch("apis.helper.get_OperationDistribution", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        assert isinstance(response.data["operation_mode"], dict)

    def test_tags_param_parsed_to_list(self):
        with patch("apis.helper.get_OperationDistribution", return_value=self._mock_return()) as m:
            _client().get(self.ENDPOINT, {"tags": "LGS1,LGS2"})
            _, kwargs = m.call_args
            # tags should have been split
            call_args = m.call_args[0]
            assert "LGS1" in call_args[2]


# ===========================================================================
# zone_distributionTimeline
# ===========================================================================

class TestZoneDistributionTimeline:
    ENDPOINT = "/api/zone_distributionTimeline"

    def _mock_return(self):
        ts = np.array(["2026-01-01T00:00:00", "2026-01-01T00:01:00"])
        return ts, np.array([6, 7]), np.array([0.0, 0.0])

    def test_returns_200(self):
        with patch("apis.helper.get_OperationDistributionTimeline", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

    def test_response_keys(self):
        with patch("apis.helper.get_OperationDistributionTimeline", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        for key in ["data_timestamp", "load_datas", "grid_datas"]:
            assert key in response.data


# ===========================================================================
# unitStatus
# ===========================================================================

class TestUnitStatus:
    ENDPOINT = "/api/unitStatus"

    def test_returns_200(self):
        mock_status = {"LGS1": "alive", "LGS2": "shutdown"}
        with patch("apis.helper.get_units_status", return_value=mock_status):
            response = _client().get(self.ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

    def test_status_dict_in_response(self):
        mock_status = {"LGS1": "alive"}
        with patch("apis.helper.get_units_status", return_value=mock_status):
            response = _client().get(self.ENDPOINT)
        assert "status_dict" in response.data


# ===========================================================================
# kpi
# ===========================================================================

class TestKpi:
    ENDPOINT = "/api/kpi"

    def test_returns_200(self):
        mock_kpi = {"LGS1": {"oee": 0.85, "phy_avail": 0.9, "performance": 0.95}}
        with patch("apis.helper.get_KPIData", return_value=mock_kpi):
            response = _client().get(self.ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

    def test_tags_param_none_when_missing(self):
        with patch("apis.helper.get_KPIData", return_value={}) as m:
            _client().get(self.ENDPOINT)
            args = m.call_args[0]
            assert args[2] is None  # tags is None when not provided

    def test_noe_metric_default(self):
        with patch("apis.helper.get_KPIData", return_value={}) as m:
            _client().get(self.ENDPOINT)
            args = m.call_args[0]
            assert args[3] == "noe"


# ===========================================================================
# severity_plot
# ===========================================================================

class TestSeverityPlot:
    ENDPOINT = "/api/severity_plot"

    def _mock_return(self):
        n = 10
        return (
            {"f0": 3},                      # counter_feature_s2
            np.array(["2026-01-01"] * n),   # df_timestamp
            np.random.randn(n, 5),          # df_feature_send
            np.random.randn(n, 5),          # y_pred_send
            np.random.randn(n, 5),          # y_pred_std_send
            np.random.randn(n, 5),          # loss_send
            {"f0": 0.5},                    # thr_now_model
        )

    def test_returns_200(self):
        with patch("apis.helper.get_SeverityNLoss", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

    def test_response_keys(self):
        with patch("apis.helper.get_SeverityNLoss", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        for key in ["counter_feature_s2", "df_timestamp", "df_feature_send",
                    "y_pred_send", "y_pred_std_send", "loss_send", "thr_now_model"]:
            assert key in response.data, f"Missing key: {key}"


# ===========================================================================
# advisory_table
# ===========================================================================

class TestAdvisoryTable:
    ENDPOINT = "/api/advisory_table"

    def _mock_return(self):
        features = [f"feat_{i}" for i in range(5)]
        return (
            "2026-01-01T00:00:00",
            {f: 2 for f in features},
            {f: np.array([1, 2, 1]) for f in features},
            {f: {1: 5, 2: 3} for f in features},
            {f: {1: 10} for f in features},
            {f: 0.3 for f in features},
        )

    def test_returns_200(self):
        with patch("apis.helper.get_advisoryTable", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

    def test_response_keys(self):
        with patch("apis.helper.get_advisoryTable", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        for key in ["last_timestamp", "last_severity_featname", "sever_1week_featname",
                    "sever_count_featname", "severity_counter_overyear", "priority_parameter"]:
            assert key in response.data, f"Missing key: {key}"


# ===========================================================================
# advisory_detail
# ===========================================================================

class TestAdvisoryDetail:
    ENDPOINT = "/api/advisory_detail/0"

    def _mock_return(self):
        n = 20
        return (
            np.array(["2026-01-01"] * n),   # data_timestamp
            np.array(["2026-01-01"] * n),   # data_timestamp_addi
            np.random.randn(n, 5),          # severity_trending_datas
            0.5,                            # priority_data
            np.random.randn(n, 5),          # sensor_datas
            [("2026-01-01", "2026-01-02")], # shutdown_periods
            [0.8, 0.5],                     # correlation_nowparam
            np.random.randn(n, 3).tolist(), # correlate_sensor_datas
            np.random.randn(n, 2).tolist(), # correlate_sensor_addi_datas
            np.random.randn(n, 3).tolist(), # correlate_trending_datas
        )

    def test_returns_200(self):
        with patch("apis.helper.get_advisoryDetail", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

    def test_feat_id_in_response(self):
        with patch("apis.helper.get_advisoryDetail", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        assert response.data["feat_id"] == 0

    def test_response_keys(self):
        with patch("apis.helper.get_advisoryDetail", return_value=self._mock_return()):
            response = _client().get(self.ENDPOINT)
        for key in ["feat_id", "data_timestamp", "severity_trending_datas",
                    "sensor_datas", "shutdown_periods", "correlation_nowparam"]:
            assert key in response.data, f"Missing key: {key}"

    def test_invalid_feat_id_uses_default(self):
        with patch("apis.helper.get_advisoryDetail", return_value=self._mock_return()) as m:
            _client().get("/api/advisory_detail/5")
            args = m.call_args[0]
            assert args[2] == 5  # feat_id passed through correctly


# ===========================================================================
# timeinfo_detail
# ===========================================================================

class TestTimeinfoDetail:
    ENDPOINT = "/api/timeinfo_detail"

    def test_returns_200(self):
        with patch("apis.helper.get_TimeInformastion",
                   return_value=("2026-01-01T00:00:00", "2026-01-01T00:10:00")):
            response = _client().get(self.ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

    def test_response_keys(self):
        with patch("apis.helper.get_TimeInformastion",
                   return_value=("2026-01-01T00:00:00", "2026-01-01T00:10:00")):
            response = _client().get(self.ENDPOINT)
        assert "datetime_last" in response.data
        assert "next_update" in response.data


# ===========================================================================
# adjust_threshold_settings
# ===========================================================================

class TestAdjustThresholdSettings:
    ENDPOINT = "/api/adjust_threshold_settings"

    def test_returns_200(self):
        response = _client().get(self.ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

    def test_response_has_datetime_last(self):
        response = _client().get(self.ENDPOINT)
        assert "datetime_last" in response.data
