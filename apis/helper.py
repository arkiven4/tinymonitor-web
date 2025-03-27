import pickle
import os
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
import matplotlib.dates as mdates
from datetime import datetime, timedelta

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
    
def calc_counterPercentage(threshold_percentages_sorted):
    counter_feature = {}
    for modex_idx, values_pred in threshold_percentages_sorted.items():
        for name_feat, percentage in values_pred.items():
            if name_feat in counter_feature:
                counter_feature[name_feat]["count"] = counter_feature[name_feat]["count"] + 1
                counter_feature[name_feat]["percentage"] = counter_feature[name_feat]["percentage"] + percentage
            else:
                counter_feature[name_feat] = {"count": 1, "percentage": percentage}

    counter_feature_s1 = dict(sorted(counter_feature.items(), key=lambda item: item[1]['count'], reverse=True)[:10])
    counter_feature_s2 = dict(sorted(counter_feature_s1.items(), key=lambda item: item[1]['percentage'] // len(model_array), reverse=True))

    for key, value in counter_feature_s2.items():
        counter_feature_s2[key]['count'] = (counter_feature_s2[key]['count'] / len(model_array)) * 100
        counter_feature_s2[key]['severity'] = percentage2severity(counter_feature_s2[key]['percentage'] // len(model_array))
        counter_feature_s2[key]['percentage'] = (counter_feature_s2[key]['percentage'] // len(model_array))

    # Find Which Model Have Highest Confidence
    counter_feature_plot = {}
    for index, value in counter_feature_s2.items():
        higher_data = {"model": 0, "percentage": 0}
        for model_idx in threshold_percentages_sorted:
            if index in threshold_percentages_sorted[model_idx]:
                if higher_data["percentage"] <= threshold_percentages_sorted[model_idx][index]:
                    higher_data["model"] = model_idx
                    higher_data["percentage"] = threshold_percentages_sorted[model_idx][index]
        
        counter_feature_plot[index] = higher_data['model']

    return counter_feature_s2, counter_feature_plot

def get_sensorNtrend(start_date, end_date):
    #start_date = "2021-04-28T06:15:00"
    #end_date = "2021-05-28T06:15:00"

    severity_trending_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db\\severity_trendings.db", "severity_trendings")
    sensor_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db\\severity_trendings.db", "original_sensor")

    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:].astype(float)
    sensor_datas = sensor_datas[:, 2:].astype(float)

    window_size = 15
    kernel = np.ones(window_size) / window_size

    for i in range(len(feature_set)):
        sensor_datas[:, i] = np.convolve(sensor_datas[:, i], kernel, mode='same')

    return data_timestamp, severity_trending_datas, sensor_datas

def get_severityNTrend(start_date=None, end_date=datetime.now().strftime("%Y-%m-%dT%H:%M:%S")):
    if start_date == None:
        timestamp = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")
        hours_2before = timestamp - timedelta(hours=2)
        last_30minutes = timestamp - timedelta(minutes=20)
        start_date = hours_2before.strftime("%Y-%m-%dT%H:%M:%S")

    threshold_percentages = {}
    threshold_percentages_sorted = {}
    for idx_model, (model_name) in enumerate(model_array):
        now_fetched = fetch_between_dates(last_30minutes.strftime("%Y-%m-%dT%H:%M:%S"), end_date, settings.MONITORINGDB_PATH + "db\\threshold_data.db", model_name)[0, 2:]

        threshold_pass = {}
        for idx_sensor, sensor_thre in enumerate(now_fetched):
            threshold_pass[feature_set[idx_sensor]] = float(sensor_thre)

        threshold_percentages_sorted[idx_model] = dict(sorted(threshold_pass.items(), key=lambda item: item[1], reverse=True)[:10])
        threshold_percentages[idx_model] = threshold_pass

    temp_original_data = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db\\original_data.db", "original_data")
    df_timestamp, df_feature = temp_original_data[:, 1], temp_original_data[:, 2:].astype(np.float16)
    #df_timestamp = np.array([convert_timestamp(now_str) for now_str in df_timestamp])

    temp_ypreds = {}
    for idx_model, (model_name) in enumerate(model_array):
        temp_ypreds[idx_model] = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db\\pred_data.db", model_name)[:, 2:].astype(np.float16)

    counter_feature_s2, counter_feature_plot = calc_counterPercentage(threshold_percentages_sorted)
    df_feature_send = []
    y_pred_send = []

    feature_index_list = [feature_set.index(feat_name) for feat_name in list(counter_feature_s2.keys())]
    for idx, (feature_index_now) in enumerate(feature_index_list[:4]):
        model_idx_highest = counter_feature_plot[feature_set[feature_index_now]]

        df_feature_send.append(temp_ypreds[model_idx_highest][:, idx])
        y_pred_send.append(df_feature[:, idx])

    df_feature_send = np.vstack(df_feature_send).T
    y_pred_send = np.vstack(y_pred_send).T
    return counter_feature_s2, df_timestamp, df_feature_send, y_pred_send


def get_top10Charts(start_date, end_date):
    if start_date == None:
        timestamp = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")

        hours_2before = timestamp - timedelta(hours=2)
        start_date = hours_2before.strftime("%Y-%m-%dT%H:%M:%S")

        month_before = timestamp - timedelta(days=30)
        start_date_month = month_before.strftime("%Y-%m-%dT%H:%M:%S")

    threshold_percentages = {}
    threshold_percentages_sorted = {}
    for idx_model, (model_name) in enumerate(model_array):
        now_fetched = fetch_between_dates(end_date, end_date, settings.MONITORINGDB_PATH + "db\\threshold_data.db", model_name)[0, 2:]

        threshold_pass = {}
        for idx_sensor, sensor_thre in enumerate(now_fetched):
            threshold_pass[feature_set[idx_sensor]] = float(sensor_thre)

        threshold_percentages_sorted[idx_model] = dict(sorted(threshold_pass.items(), key=lambda item: item[1], reverse=True)[:10])
        threshold_percentages[idx_model] = threshold_pass

    counter_feature_s2, counter_feature_plot = calc_counterPercentage(threshold_percentages_sorted)
    counter_feature_s2 = counter_feature_s2.keys()
    index_top10feat = []
    for feat_top in counter_feature_s2:
        index_top10feat.append(feature_set.index(feat_top))

    #index_top10feat = list(reversed(index_top10feat))
    ## ^ Get 10 Feature

    severity_trending_datas = fetch_between_dates(start_date_month, end_date, settings.MONITORINGDB_PATH + "db\\severity_trendings.db", "severity_trendings")
    sensor_datas = fetch_between_dates(start_date_month, end_date, settings.MONITORINGDB_PATH + "db\\severity_trendings.db", "original_sensor")

    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:][:, index_top10feat].astype(float)
    sensor_datas = sensor_datas[:, 2:][:, index_top10feat].astype(float)

    window_size = 15
    kernel = np.ones(window_size) / window_size

    for i in range(sensor_datas.shape[-1]):
        sensor_datas[:, i] = np.convolve(sensor_datas[:, i], kernel, mode='same')

    return counter_feature_s2, data_timestamp, severity_trending_datas, sensor_datas


def get_advisoryTable():
    raw_trending_datas = fetch_last_rows(10, settings.MONITORINGDB_PATH + "db\\severity_trendings.db", "severity_trendings")
    raw_trending_datas = raw_trending_datas[::-1] # Reverse order
    data_timestamp = raw_trending_datas[:, 1]
    severity_trending_datas = raw_trending_datas[:, 2:].astype(float)

    vectorized_func = np.vectorize(percentage2severity)
    severity_level_datas = vectorized_func(severity_trending_datas)

    sever_featname = {}
    for idx, feature_name in enumerate(feature_set):
        sever_featname[feature_name] = int(severity_level_datas[-1, idx])

    last_severity_featname = dict(sorted(sever_featname.items(), key=lambda item: item[1], reverse=True))
    return data_timestamp[-1], last_severity_featname

def get_advisoryDetail(sensor_id):
    datetime_now = datetime.strptime("2021-05-28T06:15:00", "%Y-%m-%dT%H:%M:%S") #datetime.utcnow()
    start_date = datetime_now - timedelta(days=30)
    start_date = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    end_date = datetime_now.strftime("%Y-%m-%dT%H:%M:%S")

    severity_trending_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db\\severity_trendings.db", "severity_trendings")
    sensor_datas = fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db\\severity_trendings.db", "original_sensor")

    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, sensor_id].astype(float)
    sensor_datas = sensor_datas[:, sensor_id].astype(float)

    return data_timestamp, severity_trending_datas, sensor_datas
    
