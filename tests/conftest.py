"""
Pytest configuration and shared fixtures for tinymonitor-web tests.

Sets up a minimal Django environment so tests can import and use Django
modules (models, views, settings) without running the full server.
"""
import os
import sys
import sqlite3
import tempfile
from unittest.mock import MagicMock

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Stub out matplotlib BEFORE django.setup() loads apis.helper.
#
# matplotlib 3.10.x has a bug where rcParams['path.simplify_threshold'] and
# rcParams['backend_fallback'] raise KeyError on certain Linux configurations.
# Since tests never actually render plots, we replace every matplotlib
# submodule that apis/helper.py imports with MagicMock stubs so the import
# chain succeeds without ever touching the broken rcParams initialisation.
#
# Submodules used by apis/helper.py:
#   import matplotlib.dates as mdates
#   import matplotlib.pyplot as plt
#   from matplotlib.colors import LinearSegmentedColormap
#   from matplotlib.gridspec import GridSpec
#   from matplotlib.dates import DateFormatter
# ---------------------------------------------------------------------------
_mpl_dates   = MagicMock()
_mpl_dates.DateFormatter = MagicMock()

_mpl_colors  = MagicMock()
_mpl_colors.LinearSegmentedColormap = MagicMock()

_mpl_gridspec = MagicMock()
_mpl_gridspec.GridSpec = MagicMock()

_mpl_root = MagicMock()
_mpl_root.dates   = _mpl_dates
_mpl_root.pyplot  = MagicMock()
_mpl_root.colors  = _mpl_colors
_mpl_root.gridspec = _mpl_gridspec

for _key, _val in [
    ("matplotlib",              _mpl_root),
    ("matplotlib.dates",        _mpl_dates),
    ("matplotlib.pyplot",       _mpl_root.pyplot),
    ("matplotlib.colors",       _mpl_colors),
    ("matplotlib.gridspec",     _mpl_gridspec),
    ("matplotlib.figure",       MagicMock()),
    ("matplotlib.axes",         MagicMock()),
    ("matplotlib.patches",      MagicMock()),
]:
    sys.modules.setdefault(_key, _val)

# ---------------------------------------------------------------------------
# Django bootstrap – must happen before any django import
# ---------------------------------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")

# Add the tinymonitor-web root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import django
django.setup()


# ---------------------------------------------------------------------------
# Feature constants (mirrors apis/commons.py)
# ---------------------------------------------------------------------------

FEATURE_SET = [
    "Active Power",
    "Reactive Power",
    "Governor speed actual",
    "Opening Wicked Gate",
    "Penstock pressure",
    "UGB X displacement",
    "UGB Y displacement",
]

ADDITIONAL_FEATURE_SET = ["Grid Selection", "TGB temperature"]

LABEL_TO_CODE = {
    "Shutdown": 0,
    "Warming": 1,
    "No Load": 2,
    "Low Load": 3,
    "Rough Zone": 4,
    "Part Load": 5,
    "Efficient Load": 6,
    "High Load": 7,
    "Undefined": 8,
}


# ---------------------------------------------------------------------------
# Timestamp fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_timestamps():
    start = datetime(2026, 1, 1, 0, 0, 0)
    return [start + timedelta(minutes=i) for i in range(100)]


@pytest.fixture
def feature_set():
    return FEATURE_SET


@pytest.fixture
def additional_feature_set():
    return ADDITIONAL_FEATURE_SET


# ---------------------------------------------------------------------------
# SQLite DB helpers
# ---------------------------------------------------------------------------

def _init_table(db_path: str, table: str, feature_set: list):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cols = ", ".join([f'"{f.replace(" ", "_")}" REAL' for f in feature_set])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT UNIQUE,
            {cols}
        )
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ts ON {table}(timestamp)")
    conn.commit()
    conn.close()


def _insert_rows(db_path: str, table: str, timestamps, data: np.ndarray, feature_set: list):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    col_list = ", ".join([f'"{f.replace(" ", "_")}"' for f in feature_set])
    placeholders = ", ".join(["?"] * (len(feature_set) + 1))
    sql = f'INSERT OR REPLACE INTO {table} (timestamp, {col_list}) VALUES ({placeholders})'
    rows = [(pd.to_datetime(ts).isoformat(), *[float(v) for v in data[i]])
            for i, ts in enumerate(timestamps)]
    cur.executemany(sql, rows)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixture: temporary DB with sensor_data
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def populated_sensor_db(tmp_path, sample_timestamps, feature_set):
    """
    SQLite DB with 'original_data' table populated with 100 rows.
    """
    db_path = str(tmp_path / "original_data.db")
    data = np.random.uniform(0, 100, (len(sample_timestamps), len(feature_set)))
    _init_table(db_path, "original_data", feature_set)
    _insert_rows(db_path, "original_data", sample_timestamps, data, feature_set)
    return db_path


@pytest.fixture
def populated_severity_db(tmp_path, sample_timestamps, feature_set):
    """
    SQLite DB with 'severity_trendings' table populated with 100 rows
    of severity percentages (0-100).
    """
    db_path = str(tmp_path / "severity_trendings.db")
    np.random.seed(42)
    data = np.random.uniform(0, 100, (len(sample_timestamps), len(feature_set)))
    _init_table(db_path, "severity_trendings", feature_set)
    _insert_rows(db_path, "severity_trendings", sample_timestamps, data, feature_set)
    return db_path


@pytest.fixture
def populated_kpi_db(tmp_path, sample_timestamps):
    """
    SQLite DB with 'LGS1_timeline' KPI table.
    Columns: active_power, rpm, cb, aux, aux_1
    """
    db_path = str(tmp_path / "kpi.db")
    n = len(sample_timestamps)
    np.random.seed(0)
    active_power = np.random.uniform(40, 70, n)
    rpm          = np.ones(n) * 280
    cb           = np.ones(n)
    aux          = np.zeros(n)
    aux_1        = np.zeros(n)

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
    rows = [
        (pd.to_datetime(ts).isoformat(), ap, r, c, a, a1)
        for ts, ap, r, c, a, a1
        in zip(sample_timestamps, active_power, rpm, cb, aux, aux_1)
    ]
    cur.executemany(
        "INSERT OR REPLACE INTO LGS1_timeline (timestamp, active_power, rpm, cb, aux, aux_1) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Django test client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    from rest_framework.test import APIClient
    return APIClient()
