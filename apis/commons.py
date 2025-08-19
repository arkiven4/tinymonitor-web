import pickle, os, sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# from pathlib import Path
# print(Path(__file__).resolve())

feature_set = ['Active Power', 'Reactive Power', 'Governor speed actual', 'UGB X displacement', 'UGB Y displacement',
    'LGB X displacement', 'LGB Y displacement', 'TGB X displacement',
    'TGB Y displacement', 'Stator winding temperature 13',
    'Stator winding temperature 14', 'Stator winding temperature 15',
    'Surface Air Cooler Air Outlet Temperature',
    'Surface Air Cooler Water Inlet Temperature',
    'Surface Air Cooler Water Outlet Temperature',
    'Stator core temperature', 'UGB metal temperature',
    'LGB metal temperature 1', 'LGB metal temperature 2',
    'LGB oil temperature', 'Penstock Flow', 'Turbine flow',
    'UGB cooling water flow', 'LGB cooling water flow',
    'Generator cooling water flow', 'Governor Penstock Pressure',
    'Penstock pressure', 'Opening Wicked Gate', 'UGB Oil Contaminant',
    'Gen Thrust Bearing Oil Contaminant']

model_array = ["Attention", "DTAAD", "MAD_GAN", "TranAD", "DAGMM", "USAD", "OmniAnomaly"]

label_to_code = {
        'Shutdown': 0,
        'Warming': 1,
        'No Load': 2,
        'Low Load': 3,
        'Rough Zone': 4,
        'Part Load': 5,
        'Efficient Load': 6,
        'High Load': 7,
        'Undefined': 8
    }

def label_load(row):
    ap = row['Active Power']
    rpm = row['Governor speed actual']
    if ap < 1 and rpm < 1:
        return 'Shutdown'
    elif ap < 3 and rpm < 200:
        return 'Warming'
    elif ap < 3 and rpm > 200:
        return 'No Load'
    elif 1 <= ap < 20 and rpm > 200:
        return 'Low Load'
    elif 20 <= ap < 40 and rpm > 200:
        return 'Rough Zone'
    elif 40 <= ap < 50 and rpm > 200:
        return 'Part Load'
    elif 50 <= ap < 65 and rpm > 200:
        return 'Efficient Load'
    elif ap >= 65 and rpm > 200:
        return 'High Load'
    else:
        return 'Undefined'

def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return ((a - min_a) / (max_a - min_a + 0.0001)), min_a, max_a

def denormalize3(a_norm, min_a, max_a):
    return a_norm * (max_a - min_a + 0.0001) + min_a


def fetch_between_dates(start_date, end_date, db_name="data.db", table_name="sensor_data"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        SELECT * FROM {table_name} WHERE timestamp BETWEEN ? AND ?
    """, (start_date, end_date))
    
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return np.array([])
    
    return np.array(rows)

def fetch_last_rows(num_row, db_name="data.db", table_name="sensor_data"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")

    cursor.execute(f"""
        SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT ?
    """, (num_row,))
    
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return np.array([])
    
    return np.array(rows)

def get_LastdateLastRow(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"""SELECT * FROM original_data order by rowid desc LIMIT 1""")
    rows = cursor.fetchall()
    conn.close()
    datetime_last = np.datetime64(np.array(rows)[:, 1][0]).astype(datetime)

    return datetime_last

def convert_timestamp(timestamp_str):
    dt = datetime.fromisoformat(timestamp_str)
    return pd.Timestamp(dt.strftime('%Y-%m-%d %H:%M:%S'))

def percentage2severity(value):
    return (
        1 if 0 <= value < 5 else
        2 if 5 <= value < 20 else
        3 if 20 <= value < 40 else
        4 if 40 <= value < 75 else
        5 if 75 <= value <= 100 else
        6
    )

def calc_counterPercentage(threshold_percentages):
    counter_feature = {}
    for modex_idx, values_pred in threshold_percentages.items():
        values_pred = dict(sorted(values_pred.items(), key=lambda item: item[1], reverse=True)[:10])
        for name_feat, percentage in values_pred.items():
            if name_feat in counter_feature:
                counter_feature[name_feat]["count"] = counter_feature[name_feat]["count"] + 1
                counter_feature[name_feat]["percentage"] = counter_feature[name_feat]["percentage"] + percentage
            else:
                counter_feature[name_feat] = {"count": 1, "percentage": percentage}

    counter_feature_s1 = dict(sorted(counter_feature.items(), key=lambda item: item[1]['count'], reverse=True)[:10])
    counter_feature_s2 = dict(sorted(counter_feature_s1.items(), key=lambda item: item[1]['percentage'] // len(model_array), reverse=True))
    #counter_feature_s2_rank = dict(sorted(counter_feature_s1.items(), key=lambda item: item[1]['count'], reverse=True))

    for key, value in counter_feature_s2.items():
        counter_feature_s2[key]['count'] = (counter_feature_s2[key]['count'] / len(model_array)) * 100
        counter_feature_s2[key]['severity'] = percentage2severity(counter_feature_s2[key]['percentage'] // len(model_array))
        counter_feature_s2[key]['percentage'] = (counter_feature_s2[key]['percentage'] // len(model_array))

    # Find Which Model Have Highest Confidence
    counter_feature_plot = {}
    for index, value in counter_feature_s2.items():
        higher_data = {"model": 0, "percentage": 0}
        for model_idx in threshold_percentages:
            if index in threshold_percentages[model_idx]:
                if higher_data["percentage"] <= threshold_percentages[model_idx][index]:
                    higher_data["model"] = model_idx
                    higher_data["percentage"] = threshold_percentages[model_idx][index]
        
        counter_feature_plot[index] = higher_data['model']

    return counter_feature_s2, counter_feature_plot




def order_objects_by_keys(data, key_order):
    ordered_dict = {}
    for key in key_order:
        if key in data:
            ordered_dict[key] = data[key]
    return ordered_dict


def process_shutdownTimestamp(data_timestamp, sensor_datas):
    activepower_data = sensor_datas[:, 0].astype(float)
    rpm_data = sensor_datas[:, 2].astype(float)
    shutdown_mask = (activepower_data <= 3) & (rpm_data <= 10)
    change_points = np.diff(shutdown_mask.astype(int), prepend=0)

    start_indices = np.where(change_points == 1)[0]
    end_indices = np.where(change_points == -1)[0]

    if shutdown_mask[-1]:
        end_indices = np.append(end_indices, len(shutdown_mask))

    if shutdown_mask[0]:
        start_indices = np.insert(start_indices, 0, 0)

    shutdown_periods = []
    for start, end in zip(start_indices, end_indices):
        start_time = data_timestamp[start]
        end_time = data_timestamp[end - 1]
        shutdown_periods.append((start_time, end_time))
    return shutdown_periods

def process_SNLTimestamp(data_timestamp, sensor_datas):
    activepower_data = sensor_datas[:, 0].astype(float)
    rpm_data = sensor_datas[:, 2].astype(float)
    shutdown_mask = (activepower_data <= 3) & (rpm_data >= 259.35) & (rpm_data <= 286.65)
    change_points = np.diff(shutdown_mask.astype(int), prepend=0)

    start_indices = np.where(change_points == 1)[0]
    end_indices = np.where(change_points == -1)[0]

    if shutdown_mask[-1]:
        end_indices = np.append(end_indices, len(shutdown_mask))

    if shutdown_mask[0]:
        start_indices = np.insert(start_indices, 0, 0)

    shutdown_periods = []
    for start, end in zip(start_indices, end_indices):
        start_time = data_timestamp[start]
        end_time = data_timestamp[end - 1]
        shutdown_periods.append((start_time, end_time))
    return shutdown_periods

def fetch_column_threshold_counts(start_date, db_name="data.db", table_name="sensor_data", threshold=5):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Get current year start and end
    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
    year = start_date.year
    start_of_year = f"{year}-01-01 00:00:00"
    end_of_year = f"{year}-12-31 23:59:59"

    # Get column names, skip 'timestamp' column
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall() if row[1].lower() != "timestamp"][1:]

    result = {}
    for col in columns:
        # 1. Find first timestamp in the year where the column > threshold
        cursor.execute(f"""
            SELECT timestamp FROM {table_name}
            WHERE "{col}" > ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC LIMIT 1
        """, (threshold, start_of_year, end_of_year))
        first = cursor.fetchone()

        if not first:
            result[col] = {"first_timestamp": "2011-01-01T00:00:00", "count_above_5": 0}
            continue

        first_timestamp = first[0]

        # 2. Count how many values are > threshold from that timestamp to end of year
        cursor.execute(f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE "{col}" > ? AND timestamp BETWEEN ? AND ?
        """, (threshold, first_timestamp, end_of_year))
        count = cursor.fetchone()[0]
        
        result[col] = {"first_timestamp": first_timestamp, "count_above_5": count}

    conn.close()
    return result

def process_operationZone(start_date, end_date, db_name="data.db", table_name="sensor_data"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    query = f"""
    SELECT
    CASE
        WHEN "active_power" < 1 AND "rpm" < 1 THEN 'Shutdown'
        WHEN "active_power" < 3 AND "rpm" < 250 THEN 'Warming'
        WHEN "active_power" < 3 AND "rpm" > 250 THEN 'No Load'
        WHEN "active_power" >= 1 AND "active_power" < 20 AND "rpm" > 250 THEN 'Low Load'
        WHEN "active_power" >= 20 AND "active_power" < 40 AND "rpm" > 250 THEN 'Rough Zone'
        WHEN "active_power" >= 40 AND "active_power" < 50 AND "rpm" > 250 THEN 'Part Load'
        WHEN "active_power" >= 50 AND "active_power" < 65 AND "rpm" > 250 THEN 'Efficient Load'
        WHEN "active_power" >= 65 AND "rpm" > 250 THEN 'High Load'
        ELSE 'Shutdown'
    END AS Label,
    COUNT(*) AS Count
    FROM {table_name}
    WHERE timestamp BETWEEN ? AND ?
    GROUP BY Label
    ORDER BY Count DESC
    """

    cursor.execute(query, (start_date, end_date))
    results = cursor.fetchall()
    return results

def process_operationMode(start_date, end_date, db_name="data.db", table_name="sensor_data"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    query = f"""
    SELECT aux_1, COUNT(*) as Count
    FROM {table_name}
    WHERE timestamp BETWEEN ? AND ?
    GROUP BY aux_1
    ORDER BY aux_1
    """

    cursor.execute(query, (start_date, end_date))
    results = cursor.fetchall()

    #results = [item for item in results if item[0] in (0.0, 1.0)]
    results = [
        ("Grid" if val == 0.0 else "Furnace", count)
        for val, count in results if val in (0.0, 1.0)
    ]
    return results

def hampel_filter(series, window_size=3, n_sigmas=3):
    new_series = series.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    n = len(series)

    for i in range(window_size, n - window_size):
        window = series[i - window_size:i + window_size + 1]
        median = np.median(window)
        mad = k * np.median(np.abs(window - median))
        if np.abs(series[i] - median) > n_sigmas * mad:
            new_series[i] = median
    return new_series