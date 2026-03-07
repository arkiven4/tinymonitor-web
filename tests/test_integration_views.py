"""
Integration tests for tinymonitor-web views.

These tests hit the actual Django URL router with mocked helper functions
to verify the full request→response pipeline works end-to-end without
requiring live databases or pickle files.

Tests are grouped to mirror the API structure that cbm_vale feeds into.
"""
import sys
import os
from unittest.mock import patch

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rest_framework.test import APIClient
from rest_framework import status


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

FEATURE_NAMES = [f"Feature_{i}" for i in range(7)]


def _panel_summary_stub():
    return (
        "2026-03-07T10:00:00",
        {f: 55 for f in FEATURE_NAMES},
        {f: list(range(10)) for f in FEATURE_NAMES},
        {f: 2 for f in FEATURE_NAMES},
        {f: list(range(10)) for f in FEATURE_NAMES},
        FEATURE_NAMES[:3],
        {f: {1: 5, 2: 3, 3: 2} for f in FEATURE_NAMES},
        {f: 1.5 for f in FEATURE_NAMES},
    )


def _kpi_stub():
    return {
        "LGS1": {"oee": 0.80, "phy_avail": 0.90, "performance": 0.88, "uo_avail": 0.92},
        "LGS2": {"oee": 0.75, "phy_avail": 0.88, "performance": 0.85, "uo_avail": 0.90},
    }


# ===========================================================================
# Full request lifecycle tests
# ===========================================================================

@pytest.fixture
def client():
    return APIClient()


class TestApiRoutes:
    """Verify every registered URL returns a valid response when mocked."""

    def test_panel_summary_route(self, client):
        with patch("apis.helper.get_PanelSummary", return_value=_panel_summary_stub()):
            r = client.get("/api/panel_summary")
        assert r.status_code == status.HTTP_200_OK

    def test_zone_distribution_route(self, client):
        with patch("apis.helper.get_OperationDistribution",
                   return_value=([("Grid", 80)], [("High Load", 60)])):
            r = client.get("/api/zone_distribution")
        assert r.status_code == status.HTTP_200_OK

    def test_zone_distribution_timeline_route(self, client):
        ts = np.array(["2026-01-01T00:00:00"])
        with patch("apis.helper.get_OperationDistributionTimeline",
                   return_value=(ts, np.array([6]), np.array([0.0]))):
            r = client.get("/api/zone_distributionTimeline")
        assert r.status_code == status.HTTP_200_OK

    def test_unit_status_route(self, client):
        with patch("apis.helper.get_units_status",
                   return_value={"LGS1": "alive", "LGS2": "shutdown"}):
            r = client.get("/api/unitStatus")
        assert r.status_code == status.HTTP_200_OK

    def test_kpi_route(self, client):
        with patch("apis.helper.get_KPIData", return_value=_kpi_stub()):
            r = client.get("/api/kpi")
        assert r.status_code == status.HTTP_200_OK

    def test_severity_plot_route(self, client):
        n = 10
        mock_data = (
            {"Feature_0": 2},
            np.array(["2026-01-01"] * n),
            np.random.randn(n, 5),
            np.random.randn(n, 5),
            np.random.randn(n, 5),
            np.random.randn(n, 5),
            {"Feature_0": 0.3},
        )
        with patch("apis.helper.get_SeverityNLoss", return_value=mock_data):
            r = client.get("/api/severity_plot")
        assert r.status_code == status.HTTP_200_OK

    def test_advisory_table_route(self, client):
        mock_data = (
            "2026-03-07T00:00:00",
            {f: 3 for f in FEATURE_NAMES},
            {f: [1, 2, 3] for f in FEATURE_NAMES},
            {f: {1: 5} for f in FEATURE_NAMES},
            {f: {1: 10} for f in FEATURE_NAMES},
            {f: 0.8 for f in FEATURE_NAMES},
        )
        with patch("apis.helper.get_advisoryTable", return_value=mock_data):
            r = client.get("/api/advisory_table")
        assert r.status_code == status.HTTP_200_OK

    def test_advisory_detail_route(self, client):
        n = 15
        mock_data = (
            np.array(["2026-01-01"] * n),
            np.array(["2026-01-01"] * n),
            np.random.randn(n, 5),
            0.7,
            np.random.randn(n, 5),
            [],
            [0.9, 0.6],
            [],
            [],
            [],
        )
        with patch("apis.helper.get_advisoryDetail", return_value=mock_data):
            r = client.get("/api/advisory_detail/2")
        assert r.status_code == status.HTTP_200_OK
        assert r.data["feat_id"] == 2

    def test_timeinfo_detail_route(self, client):
        with patch("apis.helper.get_TimeInformastion",
                   return_value=("2026-03-07T10:00:00", "2026-03-07T10:10:00")):
            r = client.get("/api/timeinfo_detail")
        assert r.status_code == status.HTTP_200_OK

    def test_adjust_threshold_settings_route(self, client):
        r = client.get("/api/adjust_threshold_settings")
        assert r.status_code == status.HTTP_200_OK


# ===========================================================================
# Query-parameter propagation
# ===========================================================================

class TestQueryParamPropagation:
    """Verify that query parameters are correctly forwarded to helper functions."""

    START = "2026-01-01T00:00:00"
    END   = "2026-01-02T00:00:00"

    def test_panel_summary_date_params(self):
        with patch("apis.helper.get_PanelSummary", return_value=_panel_summary_stub()) as m:
            APIClient().get("/api/panel_summary", {"start_date": self.START, "end_date": self.END})
        m.assert_called_once_with(self.START, self.END)

    def test_zone_distribution_tags_split(self):
        with patch("apis.helper.get_OperationDistribution",
                   return_value=([("Grid", 1)], [])) as m:
            APIClient().get("/api/zone_distribution", {
                "start_date": self.START, "end_date": self.END, "tags": "LGS1,LGS2"
            })
        _, kwargs = m.call_args if m.call_args.kwargs else (m.call_args[0], {})
        call_args = m.call_args[0]
        assert call_args[2] == ["LGS1", "LGS2"]

    def test_kpi_noe_metric_custom(self):
        with patch("apis.helper.get_KPIData", return_value={}) as m:
            APIClient().get("/api/kpi", {"start_date": self.START, "end_date": self.END, "noe_metric": "custom_noe"})
        args = m.call_args[0]
        assert args[3] == "custom_noe"

    def test_advisory_detail_feat_correlate_parsed(self):
        n = 5
        mock_data = (
            np.array(["2026-01-01"] * n),
            np.array(["2026-01-01"] * n),
            np.zeros((n, 5)),
            0.0, np.zeros((n, 5)), [], [], [], [], []
        )
        with patch("apis.helper.get_advisoryDetail", return_value=mock_data) as m:
            APIClient().get("/api/advisory_detail/1", {"feat_correlate": "0,1,2"})
        args = m.call_args[0]
        assert args[3] == [0, 1, 2]  # feat_correlate parsed as list of ints


# ===========================================================================
# Edge cases: missing parameters
# ===========================================================================

class TestMissingParameters:
    def test_panel_summary_without_dates(self):
        """Endpoint should still return 200 even without date params (None values)."""
        with patch("apis.helper.get_PanelSummary", return_value=_panel_summary_stub()) as m:
            r = APIClient().get("/api/panel_summary")
        assert r.status_code == status.HTTP_200_OK
        m.assert_called_once_with(None, None)

    def test_zone_distribution_without_tags(self):
        with patch("apis.helper.get_OperationDistribution",
                   return_value=([("Grid", 1)], [])) as m:
            APIClient().get("/api/zone_distribution")
        args = m.call_args[0]
        assert args[2] is None

    def test_advisory_detail_invalid_correlate_string(self):
        """Invalid feat_correlate string should result in empty list, not crash."""
        n = 5
        mock_data = (
            np.array(["2026-01-01"] * n), np.array(["2026-01-01"] * n),
            np.zeros((n, 5)), 0.0, np.zeros((n, 5)), [], [], [], [], []
        )
        with patch("apis.helper.get_advisoryDetail", return_value=mock_data) as m:
            r = APIClient().get("/api/advisory_detail/0", {"feat_correlate": "abc,xyz"})
        assert r.status_code == status.HTTP_200_OK
        args = m.call_args[0]
        assert args[3] == []  # invalid strings → empty list
