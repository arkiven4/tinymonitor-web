import pickle
import os
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import Counter
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter

from django.conf import settings

from pathlib import Path
print(Path(__file__).resolve())

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
with open('model_thr.pickle', 'rb') as handle:
    model_thr = pickle.load(handle)

with open('normalize_2023.pickle', 'rb') as handle:
    normalize_obj = pickle.load(handle)
    min_a, max_a = normalize_obj['min_a'], normalize_obj['max_a']

with open('param_statistic.pickle', 'rb') as handle:
    param_statistic = pickle.load(handle) # (4, 30)

with open('correlation.pickle', 'rb') as handle:
    correlation_param = pickle.load(handle) # (4, 30)

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

    cursor.execute(f"""
        SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT ?
    """, (num_row,))
    
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return np.array([])
    
    return np.array(rows)

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


def get_LastdateLastRow(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"""SELECT * FROM original_data order by rowid desc LIMIT 1""")
    rows = cursor.fetchall()
    conn.close()
    datetime_last = np.datetime64(np.array(rows)[:, 1][0]).astype(datetime)

    return datetime_last

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
        WHEN "Active_Power" < 1 AND "Governor_speed_actual" < 1 THEN 'Shutdown'
        WHEN "Active_Power" < 3 AND "Governor_speed_actual" < 250 THEN 'Warming'
        WHEN "Active_Power" < 3 AND "Governor_speed_actual" > 250 THEN 'No Load'
        WHEN "Active_Power" >= 1 AND "Active_Power" < 20 AND "Governor_speed_actual" > 250 THEN 'Low Load'
        WHEN "Active_Power" >= 20 AND "Active_Power" < 40 AND "Governor_speed_actual" > 250 THEN 'Rough Zone'
        WHEN "Active_Power" >= 40 AND "Active_Power" < 50 AND "Governor_speed_actual" > 250 THEN 'Part Load'
        WHEN "Active_Power" >= 50 AND "Active_Power" < 65 AND "Governor_speed_actual" > 250 THEN 'Efficient Load'
        WHEN "Active_Power" >= 65 AND "Governor_speed_actual" > 250 THEN 'High Load'
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
    SELECT Grid_Selection, COUNT(*) as Count
    FROM {table_name}
    WHERE timestamp BETWEEN ? AND ?
    GROUP BY Grid_Selection
    ORDER BY Grid_Selection
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

#########################################################
#
# Main Function For Web
#
#########################################################

def get_PanelSummary(start_date=None, end_date=None):
    end_dateLate = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S") - timedelta(minutes=30)
    # Last Row Safety
    try:
        now_fetched = fetch_between_dates(end_dateLate.strftime("%Y-%m-%dT%H:%M:%S"), end_date, settings.MONITORINGDB_PATH + "db/threshold_data.db", model_array[0])[-1, 2:]
    except:
        datetime_last = get_LastdateLastRow(settings.MONITORINGDB_PATH + "db/original_data.db")
        end_dateLate = datetime_last - timedelta(minutes=30)
        start_date = (datetime_last - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        end_date = datetime_last.strftime("%Y-%m-%dT%H:%M:%S")

    raw_original_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/original_data.db", "original_data")
    raw_original_datas = raw_original_datas[::-1]
    data_timestamp = raw_original_datas[:, 1]
    sensor_datas = raw_original_datas[:, 2:].astype(float)

    raw_trending_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    raw_trending_datas = raw_trending_datas[::-1]
    severity_trending_datas = raw_trending_datas[:, 2:].astype(float)

    vectorized_func = np.vectorize(percentage2severity)
    severity_level_datas = vectorized_func(severity_trending_datas)

    last_severity_featname = {}
    last_sensor_featname = {}
    sensor_featname = {}
    sever_featname = {}
    sever_count_featname = {}
    for idx, feature_name in enumerate(feature_set):
        sever_featname[feature_name] = severity_level_datas[:, idx]
        last_severity_featname[feature_name] = int(severity_level_datas[-1, idx])

        sensor_featname[feature_name] = sensor_datas[:, idx]
        last_sensor_featname[feature_name] = int(sensor_datas[-1, idx])

        temp_severity_counts = Counter(severity_level_datas[:, idx])
        for level in range(1, 7):
            temp_severity_counts.setdefault(level, 0)
        sever_count_featname[feature_name] = {int(k): v for k, v in temp_severity_counts.items()}

    ordered_feature_name = list(dict(sorted(last_severity_featname.items(), key=lambda item: item[1], reverse=True)).keys())[:5]

    # Try Priority
    start_dateLate = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S") - timedelta(days=30)
    severity_trending_datas = fetch_between_dates(start_dateLate, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    sensor_datas = fetch_between_dates(start_dateLate, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "original_sensor")
    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:].astype(float)
    sensor_datas = sensor_datas[:, 2:].astype(float)
    priority_parameter = {}
    datetime_index = pd.to_datetime(data_timestamp)
    for idx, feature_name in enumerate(feature_set):
        series = pd.Series(severity_trending_datas[:, idx], index=datetime_index)
        if len(series) >= 400:
            series = series[~series.index.duplicated(keep='first')]
            series = series.asfreq('15min').ffill()
            result = seasonal_decompose(series, model='additive', period=96 * 2)
            trend = result.trend.dropna()
            x = np.arange(len(trend))
            corr, _ = spearmanr(x, trend)
            if np.isnan(corr) or np.isinf(corr):
                corr = 0
            if corr <= 0:
                priority_parameter[feature_name] = float((corr + 1) * 25)
            else:
                priority_parameter[feature_name] = float(25 + corr * 75)
        else:
            priority_parameter[feature_name] = float(1)

    return data_timestamp[-1], last_sensor_featname, sensor_featname, last_severity_featname, sever_featname, ordered_feature_name, sever_count_featname, priority_parameter

def get_OperationDistribution(start_date=None, end_date=None):
    operation_mode = process_operationMode(start_date, end_date, settings.MONITORINGDB_PATH + "db/original_data.db", "additional_original_data")
    operation_zone = process_operationZone(start_date, end_date, settings.MONITORINGDB_PATH + "db/original_data.db", "original_data")
    print(operation_mode)
    print(operation_zone)
    return operation_mode, operation_zone


def get_SeverityNLoss(start_date=None, end_date=None):
    end_dateLate = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S") - timedelta(minutes=30)

    # Last Row Safety
    try:
        now_fetched = fetch_between_dates(end_dateLate.strftime("%Y-%m-%dT%H:%M:%S"), end_date, settings.MONITORINGDB_PATH + "db/threshold_data.db", model_array[0])[-1, 2:]
    except:
        datetime_last = get_LastdateLastRow(settings.MONITORINGDB_PATH + "db/original_data.db")
        end_dateLate = datetime_last - timedelta(minutes=30)
        start_date = (datetime_last - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        end_date = datetime_last.strftime("%Y-%m-%dT%H:%M:%S")

    threshold_percentages = {}
    threshold_percentages_sorted = {}
    for idx_model, (model_name) in enumerate(model_array):
        now_fetched = fetch_between_dates(end_dateLate.strftime("%Y-%m-%dT%H:%M:%S"), end_date, settings.MONITORINGDB_PATH + "db/threshold_data.db", model_name)[-1, 2:]

        threshold_pass = {}
        for idx_sensor, sensor_thre in enumerate(now_fetched):
            threshold_pass[feature_set[idx_sensor]] = float(sensor_thre)

        threshold_percentages_sorted[idx_model] = dict(sorted(threshold_pass.items(), key=lambda item: item[1], reverse=True)[:10])
        threshold_percentages[idx_model] = threshold_pass

    temp_original_data = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/original_data.db", "original_data")
    df_timestamp, df_feature = temp_original_data[:, 1], temp_original_data[:, 2:].astype(np.float16)

    temp_ypreds = {}
    for idx_model, (model_name) in enumerate(model_array):
        temp_ypreds[idx_model] = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/pred_data.db", model_name)[:, 2:].astype(np.float16)

    counter_feature_s2, counter_feature_plot = calc_counterPercentage(threshold_percentages_sorted)
    df_feature_send = []
    y_pred_send = []
    loss_send = []
    thr_now_model = []

    feature_index_list = [feature_set.index(feat_name) for feat_name in list(counter_feature_s2.keys())]
    for idx, (feature_index_now) in enumerate(feature_index_list[:4]):
        model_idx_highest = counter_feature_plot[feature_set[feature_index_now]]

        y_true, _, _ = normalize3(df_feature, min_a, max_a)
        y_pred, _, _ = normalize3(temp_ypreds[model_idx_highest], min_a, max_a)

        loss = denormalize3((y_true - y_pred) ** 2, min_a, max_a)
        model_thr_temp = denormalize3(model_thr[model_array[model_idx_highest]], min_a, max_a)

        loss_send.append(loss[:, feature_index_now])
        thr_now_model.append(float(model_thr_temp[feature_index_now]))

        df_feature_send.append(temp_ypreds[model_idx_highest][:, feature_index_now])
        y_pred_send.append(df_feature[:, feature_index_now])

    df_feature_send = np.vstack(df_feature_send).T
    y_pred_send = np.vstack(y_pred_send).T
    loss_send = np.vstack(loss_send).T
    return counter_feature_s2, df_timestamp, df_feature_send, y_pred_send, loss_send, thr_now_model

def get_top10Charts(start_date, end_date):
    end_dateLate = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S") - timedelta(minutes=30)

    # Last Row Safety
    try:
        now_fetched = fetch_between_dates(end_dateLate.strftime("%Y-%m-%dT%H:%M:%S"), end_date, settings.MONITORINGDB_PATH + "db/threshold_data.db", model_array[0])[-1, 2:]
    except:
        datetime_last = get_LastdateLastRow(settings.MONITORINGDB_PATH + "db/original_data.db")
        end_dateLate = datetime_last - timedelta(minutes=30)
        start_date = (datetime_last - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        end_date = datetime_last.strftime("%Y-%m-%dT%H:%M:%S")

    threshold_percentages = {}
    for idx_model, (model_name) in enumerate(model_array):
        now_fetched = fetch_between_dates(end_dateLate.strftime("%Y-%m-%dT%H:%M:%S"), end_date, settings.MONITORINGDB_PATH + "db/threshold_data.db", model_name)[-1, 2:]

        threshold_pass = {}
        for idx_sensor, sensor_thre in enumerate(now_fetched):
            threshold_pass[feature_set[idx_sensor]] = float(sensor_thre)

        threshold_percentages[idx_model] = threshold_pass

    counter_feature_s2, counter_feature_plot = calc_counterPercentage(threshold_percentages)
    counter_feature_s2 = counter_feature_s2.keys()
    index_top10feat = []
    for feat_top in counter_feature_s2:
        index_top10feat.append(feature_set.index(feat_top))

    #index_top10feat = list(reversed(index_top10feat))
    ## ^ Get 10 Feature

    severity_trending_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    sensor_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "original_sensor")

    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:][:, index_top10feat].astype(float)
    sensor_datas = sensor_datas[:, 2:][:, index_top10feat].astype(float)
    sensor_statistic_current = param_statistic[:, index_top10feat]

    # window_size = 15
    # kernel = np.ones(window_size) / window_size

    # for i in range(sensor_datas.shape[-1]):
    #     sensor_datas[:, i] = np.convolve(sensor_datas[:, i], kernel, mode='same')

    return counter_feature_s2, data_timestamp, severity_trending_datas, sensor_datas, sensor_statistic_current

def get_sensorNtrend(start_date, end_date):
    severity_trending_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    sensor_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "original_sensor")

    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:].astype(float)
    sensor_datas = sensor_datas[:, 2:].astype(float)

    window_size = 15
    kernel = np.ones(window_size) / window_size

    for i in range(len(feature_set)):
        sensor_datas[:, i] = np.convolve(sensor_datas[:, i], kernel, mode='same')

    return data_timestamp, severity_trending_datas, sensor_datas

def get_advisoryTable(start_date, end_date): # 2529
    # Try Priority
    start_dateLate = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S") - timedelta(days=30)
    severity_trending_datas = fetch_between_dates(start_dateLate, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    sensor_datas = fetch_between_dates(start_dateLate, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "original_sensor")
    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:].astype(float)
    sensor_datas = sensor_datas[:, 2:].astype(float)
    for i in range(len(feature_set)):
        severity_trending_datas[:, i] = hampel_filter(severity_trending_datas[:, i], window_size=93, n_sigmas=3)
    priority_parameter = {}
    datetime_index = pd.to_datetime(data_timestamp)
    for idx, feature_name in enumerate(feature_set):
        series = pd.Series(severity_trending_datas[:, idx], index=datetime_index)
        if len(series) >= 400:
            series = series[~series.index.duplicated(keep='first')]
            series = series.asfreq('15min').ffill()
            result = seasonal_decompose(series, model='additive', period=96 * 2)
            trend = result.trend.dropna()
            x = np.arange(len(trend))
            corr, _ = spearmanr(x, trend)
            if np.isnan(corr) or np.isinf(corr):
                corr = 0
            if corr <= 0:
                priority_parameter[feature_name] = float((corr + 1) * 25)
            else:
                priority_parameter[feature_name] = float(25 + corr * 75)
        else:
            priority_parameter[feature_name] = float(1)
        
    raw_trending_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    raw_trending_datas = raw_trending_datas[::-1]
    data_timestamp = raw_trending_datas[:, 1]
    severity_trending_datas = raw_trending_datas[:, 2:].astype(float)

    vectorized_func = np.vectorize(percentage2severity)
    severity_level_datas = vectorized_func(severity_trending_datas)

    severity_counter_overyear = fetch_column_threshold_counts(start_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings", threshold=5)

    sever_1week_featname = {}
    sever_featname = {}
    sever_count_featname = {}
    for idx, feature_name in enumerate(feature_set):
        sever_featname[feature_name] = int(severity_level_datas[-1, idx])
        sever_1week_featname[feature_name] = severity_level_datas[:, idx]

        temp_severity_counts = Counter(severity_level_datas[:, idx])
        for level in range(1, 7):
            temp_severity_counts.setdefault(level, 0)
        sever_count_featname[feature_name] = {int(k): v for k, v in temp_severity_counts.items()}

    last_severity_featname = dict(sorted(sever_featname.items(), key=lambda item: item[1], reverse=True))
    sever_1week_featname = order_objects_by_keys(sever_1week_featname.copy(), last_severity_featname.keys())
    return data_timestamp[-1], last_severity_featname, sever_1week_featname, sever_count_featname, severity_counter_overyear, priority_parameter


def get_advisoryDetail(start_date, end_date, sensor_id, feat_correlate):
    severity_trending_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    sensor_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "original_sensor")
    
    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:]
    sensor_datas = sensor_datas[:, 2:]

    shutdown_periods = process_shutdownTimestamp(data_timestamp, sensor_datas)

    selected_severity_trending_datas = severity_trending_datas[:, sensor_id].astype(float)
    selected_sensor_datas = sensor_datas[:, sensor_id].astype(float)

    window_size = 30
    kernel = np.ones(window_size) / window_size
    selected_severity_trending_datas = np.convolve(selected_severity_trending_datas, kernel, mode='same')

    correlation_nowparam = correlation_param[feature_set[sensor_id]]
    correlate_sensor_datas = sensor_datas[:, feat_correlate].astype(float)
    correlate_trending_datas = severity_trending_datas[:, feat_correlate].astype(float)

    return data_timestamp, selected_severity_trending_datas, selected_sensor_datas, shutdown_periods, correlation_nowparam, correlate_sensor_datas, correlate_trending_datas
    
