import pickle
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime, timedelta
from collections import Counter
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
import heapq

from django.conf import settings
import apis.commons as commons

with open('model_thr.pickle', 'rb') as handle:
    model_thr = pickle.load(handle)

with open('normalize_2023.pickle', 'rb') as handle:
    normalize_obj = pickle.load(handle)
    min_a, max_a = normalize_obj['min_a'], normalize_obj['max_a']

with open('param_statistic.pickle', 'rb') as handle:
    param_statistic = pickle.load(handle)  # (4, 30)

with open('correlation.pickle', 'rb') as handle:
    correlation_param = pickle.load(handle)  # (4, 30)

def calculate_priority(parameter_name, recap_severity, current_severity, equipment_critical_list):
    o = recap_severity['Level'] * recap_severity['Proportion']
    o = np.sum(o)
    s = current_severity
    if parameter_name in equipment_critical_list:
        c = 2
    else:
        c = 1

    p = o * s * c
    return p

recap_severity = pd.DataFrame({'Level': [1, 2, 3, 4, 5, 6], 'Proportion': [0.2, 0.1, 0.1, 0.5, 0.1, 0]})
equipment_critical_list = [
    'UGB X Displacement',
    'UGB Y Displacement',
    'LGB X Displacement',
    'LGB Y Displacement',
    'TGB X Displacement',
    'TGB Y Displacement',
    'Governor Actual Speed',
    'TGB Temperature',
    'Stator Winding Temperature 13',
    'Stator Winding Temperature 14',
    'Stator Winding Temperature 15',
    'Penstock Pressure',
    'UGB Metal Air Outlet Temperature'
]

#########################################################
#
# Main Function For Web
#
#########################################################


def get_FixedDate(start_date=None, end_date=None, ignore=False):
    if ignore:
        return start_date, end_date

    end_dateT1 = datetime.strptime(
        end_date, "%Y-%m-%dT%H:%M:%S") - timedelta(minutes=30)
    try:
        now_fetched = commons.fetch_between_dates(end_dateT1.strftime(
            "%Y-%m-%dT%H:%M:%S"), end_date, settings.MONITORINGDB_PATH + "db/threshold_data.db", commons.model_array[0])[-1, 2:]
    except:
        datetime_last = commons.get_LastdateLastRow(
            settings.MONITORINGDB_PATH + "db/original_data.db")
        start_date = (datetime_last - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        end_date = datetime_last.strftime("%Y-%m-%dT%H:%M:%S")

    return start_date, end_date

def get_PanelSummary(start_date=None, end_date=None):
    start_date, end_date = get_FixedDate(start_date, end_date)

    sensor_datas = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/original_data.db", "original_data")
    severtrend_datas = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")

    sensor_datas = sensor_datas[::-1]  # Reverse Array
    severtrend_datas = severtrend_datas[::-1]

    data_timestamp = sensor_datas[:, 1]
    sensor_datas = sensor_datas[:, 2:].astype(float)
    severtrend_datas = severtrend_datas[:, 2:].astype(float)

    vectorized_func = np.vectorize(commons.percentage2severity)
    severity_level_datas = vectorized_func(severtrend_datas)

    last_severity_featname = {}
    last_sensor_featname = {}
    sensor_featname = {}
    sever_featname = {}
    sever_count_featname = {}
    for idx, feature_name in enumerate(commons.feature_set):
        sever_featname[feature_name] = severity_level_datas[:, idx]
        last_severity_featname[feature_name] = int(
            severity_level_datas[-1, idx])

        sensor_featname[feature_name] = sensor_datas[:, idx]
        last_sensor_featname[feature_name] = int(sensor_datas[-1, idx])

        temp_severity_counts = Counter(severity_level_datas[:, idx])
        for level in range(1, 7):
            temp_severity_counts.setdefault(level, 0)
        sever_count_featname[feature_name] = {
            int(k): v for k, v in temp_severity_counts.items()}

    ordered_feature_name = list(dict(sorted(last_severity_featname.items(
    ), key=lambda item: item[1], reverse=True)).keys())[:5]

    # Try Priority
    start_dateLate = datetime.strptime(
        end_date, "%Y-%m-%dT%H:%M:%S") - timedelta(days=30)
    severtrend_datas = commons.fetch_between_dates(
        start_dateLate, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    sensor_datas = commons.fetch_between_dates(
        start_dateLate, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "original_sensor")
    data_timestamp = sensor_datas[:, 1]
    severtrend_datas = severtrend_datas[:, 2:].astype(float)
    sensor_datas = sensor_datas[:, 2:].astype(float)
    priority_parameter = {}
    for idx, feature_name in enumerate(commons.feature_set):
        current_severity = commons.percentage2severity(float(severtrend_datas[:, idx].mean()))
        priority = calculate_priority(feature_name, recap_severity, current_severity, equipment_critical_list)
        priority_parameter[feature_name] = float(priority)
        # if len(series) >= 400:
        #     series = series[~series.index.duplicated(keep='first')]
        #     series = series.asfreq('15min').ffill()
        #     result = seasonal_decompose(
        #         series, model='additive', period=96 * 2)
        #     trend = result.trend.dropna()
        #     x = np.arange(len(trend))
        #     corr, _ = spearmanr(x, trend)
        #     if np.isnan(corr) or np.isinf(corr):
        #         corr = 0
        #     if corr <= 0:
        #         priority_parameter[feature_name] = float((corr + 1) * 25)
        #     else:
        #         priority_parameter[feature_name] = float(25 + corr * 75)
        # else:
        #     priority_parameter[feature_name] = float(1)

    return data_timestamp[-1], last_sensor_featname, sensor_featname, last_severity_featname, sever_featname, ordered_feature_name, sever_count_featname, priority_parameter


def get_OperationDistribution(start_date=None, end_date=None, units=None):
    if units == None:
        units = ['LGS1']

    operation_mode = commons.process_operationMode(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/kpi.db", units[0] + "_timeline")
    operation_zone = commons.process_operationZone(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/kpi.db", units[0] + "_timeline")

    return operation_mode, operation_zone


def get_OperationDistributionTimeline(start_date=None, end_date=None, units=None):
    if units == None:
        units = ['LGS1']

    sensor_datas = commons.fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/kpi.db", units[0] + "_timeline")

    data_timestamp = sensor_datas[:, 1]
    sensor_datas = sensor_datas[:, 2:].astype(float)
    activepow_data = sensor_datas[:, 0].astype(float)
    rpm_data = sensor_datas[:, 1].astype(float)
    df = pd.DataFrame({
        'Timestap': data_timestamp,
        'Active Power': activepow_data,
        'Governor speed actual': rpm_data
    })
    aux_1 = sensor_datas[:, 3].astype(float)
    df['Load Label'] = df.apply(commons.label_load, axis=1)
    df['Load Code'] = df['Load Label'].map(commons.label_to_code)
    #df = df[df['Load Code'] != df['Load Code'].shift()].reset_index(drop=True)
    return df['Timestap'].values, df['Load Code'].values, aux_1

def get_unit_status(start_date=None, end_date=None, unit='LGS1'):
    """
    Returns 'alive' or 'shutdown' for a single unit based on its latest Load Code.
    """
    # Fetch sensor data
    sensor_datas = commons.fetch_between_dates(
        start_date, end_date,
        settings.MONITORINGDB_PATH + "db/kpi.db",
        unit + "_timeline"
    )

    if sensor_datas.shape[0] == 0:
        return "shutdown"  # no data -> assume shutdown

    data_timestamp = sensor_datas[:, 1]
    sensor_datas = sensor_datas[:, 2:].astype(float)
    activepow_data = sensor_datas[:, 0]
    rpm_data = sensor_datas[:, 1]

    df = pd.DataFrame({
        'Timestap': data_timestamp,
        'Active Power': activepow_data,
        'Governor speed actual': rpm_data
    })

    # Add Load Label and Load Code
    df['Load Label'] = df.apply(commons.label_load, axis=1)
    df['Load Code'] = df['Load Label'].map(commons.label_to_code)
    latest_code = df['Load Code'].iloc[-1]

    return "shutdown" if latest_code == 0 else "alive"

def get_units_status(start_date=None, end_date=None, units=None):
    """
    labels_dict: dict of unit -> Load Label
        e.g., {'LGS1': 'Efficient Load', 'LGS2': 'Shutdown'}
    Returns dict of unit -> status ('alive' or 'shutdown')
    """
    status_dict = {}
    for unit in units:
        status_dict[unit] = get_unit_status(start_date=start_date, end_date=end_date, unit='LGS1')
    return status_dict

def get_KPIData(start_date=None, end_date=None, units=None, noe_metric="noe"):
    if units == None:
        units = ['LGS1', 'LGS2', 'LGS3', 'BGS1', 'BGS2', 'KGS1', 'KGS2']

    kpi_results = {}
    for unit in units:
        kpi_data = commons.fetch_between_dates(
            start_date, end_date, settings.MONITORINGDB_PATH + "db/kpi.db", unit
        )
        if kpi_data is None or len(kpi_data) == 0:
            continue
        
        timestamps = kpi_data[:, 1]
        oee = kpi_data[:, 2].astype(float)
        phy_avail = kpi_data[:, 3].astype(float)
        performance = kpi_data[:, 4].astype(float)
        uo_Avail = kpi_data[:, 5].astype(float)
        aux_0 = kpi_data[:, 6].astype(float)
        aux_1 = kpi_data[:, 7].astype(float)

        # Store results
        kpi_results[unit] = {
            'timestamp': timestamps,
            'oee': oee,
            'phy_avail': phy_avail,
            'performance': performance,
            'uo_Avail': uo_Avail,
            'aux_0': aux_0,
            'aux_1': aux_1,
        }

    # Fetch plant-wide data (lpd, hpd, ahpa)
    plant_data = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/kpi.db", "PowerProd"
    )

    if plant_data is not None and len(plant_data) > 0:
        timestamps = plant_data[:, 1]
        hpd = plant_data[:, 2].astype(float)
        ahpa = plant_data[:, 3].astype(float)
        lpd = plant_data[:, 4].astype(float)
        bpd = plant_data[:, 5].astype(float)
        kpd = plant_data[:, 6].astype(float)

        plant_values = {
            'timestamp': timestamps,
            'hpd': hpd,
            'ahpa': ahpa,
            'lpd': lpd,
            'bpd': bpd,
            'kpd': kpd,
        }

        kpi_results['plant'] = plant_values

    loaded_df = pd.read_pickle(settings.MONITORINGDB_PATH + "db/number_of_event.pickle")
    #loaded_df = loaded_df[(loaded_df['Start'] >= pd.to_datetime(start_date)) & (loaded_df['Start'] <= pd.to_datetime(end_date))]
    loaded_df = loaded_df[loaded_df['Plant'].isin(units)]
    loaded_df['Duration'] = loaded_df['End'] - loaded_df['Start']
    loaded_df['Duration_hours'] = np.round(loaded_df['Duration'].dt.total_seconds() / 3600,2)
    
    if noe_metric == 'noe':
        groupen_df1 = loaded_df.groupby(['Plant', 'Category']).size().unstack(fill_value=0)
        kpi_results['noe'] = {'data': groupen_df1, 'labels': groupen_df1.index}
    else:
        groupen_df2 = loaded_df.groupby(['Plant', 'Category'])['Duration_hours'].sum().unstack(fill_value=0)
        kpi_results['noe'] = {'data': groupen_df2, 'labels': groupen_df2.index}

    return kpi_results


import time
import numpy as np

def get_SeverityNLoss(start_date=None, end_date=None):
    timings = {}

    t0 = time.perf_counter()
    start_date, end_date = get_FixedDate(start_date, end_date)
    timings['date_setup'] = time.perf_counter() - t0

    t1 = time.perf_counter()
    conn = sqlite3.connect(settings.MONITORINGDB_PATH + "db/threshold_data.db")
    cursor = conn.cursor()
    threshold_percentages = {}
    threshold_percentages_sorted = {}
    for idx_model, model_name in enumerate(commons.model_array):
        row = commons.fetch_between_dates(cursor, start_date, end_date, model_name)
        if row.size == 0:
            continue
        now_fetched = row[2:]  # already only one row
        
        threshold_pass = {
            commons.feature_set[idx_sensor]: float(sensor_thre)
            for idx_sensor, sensor_thre in enumerate(now_fetched)
        }
        threshold_percentages_sorted[idx_model] = dict(
            heapq.nlargest(10, threshold_pass.items(), key=lambda x: x[1])
        )
        threshold_percentages[idx_model] = threshold_pass
    conn.close()
    timings['fetch_thresholds'] = time.perf_counter() - t1

    t2 = time.perf_counter()
    temp_original_data = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/original_data.db", "original_data")
    df_timestamp, df_feature = temp_original_data[:, 1], temp_original_data[:, 2:].astype(np.float16)
    timings['fetch_original_data'] = time.perf_counter() - t2

    t3 = time.perf_counter()
    temp_ypreds = {}
    for idx_model, model_name in enumerate(commons.model_array):
        temp_ypreds[idx_model] = commons.fetch_between_dates(start_date, end_date, settings.MONITORINGDB_PATH + "db/pred_data.db", model_name)[:, 2:].astype(np.float16)
    timings['fetch_predictions'] = time.perf_counter() - t3

    t4 = time.perf_counter()
    counter_feature_s2, counter_feature_plot = commons.calc_counterPercentage(threshold_percentages_sorted)
    timings['calc_counterPercentage'] = time.perf_counter() - t4

    t5 = time.perf_counter()
    df_feature_send = []
    y_pred_send = []
    loss_send = []
    thr_now_model = []

    feature_index_list = [commons.feature_set.index(feat_name) for feat_name in list(counter_feature_s2.keys())]
    for idx, feature_index_now in enumerate(feature_index_list[:4]):
        model_idx_highest = counter_feature_plot[commons.feature_set[feature_index_now]]

        y_true, _, _ = commons.normalize3(df_feature, min_a, max_a)
        y_pred, _, _ = commons.normalize3(temp_ypreds[model_idx_highest], min_a, max_a)

        loss = commons.denormalize3((y_true - y_pred) ** 2, min_a, max_a)
        model_thr_temp = commons.denormalize3(model_thr[commons.model_array[model_idx_highest]], min_a, max_a)

        loss_send.append(loss[:, feature_index_now])
        thr_now_model.append(float(model_thr_temp[feature_index_now]))

        df_feature_send.append(df_feature[:, feature_index_now])
        y_pred_send.append(temp_ypreds[model_idx_highest][:, feature_index_now])

    df_feature_send = np.vstack(df_feature_send).T
    y_pred_send = np.vstack(y_pred_send).T
    loss_send = np.vstack(loss_send).T
    timings['feature_processing'] = time.perf_counter() - t5

    # Print timing results
    for step, duration in timings.items():
        print(f"{step}: {duration:.4f} sec")

    return counter_feature_s2, df_timestamp, df_feature_send, y_pred_send, loss_send, thr_now_model



def get_top10Charts(start_date, end_date):
    start_date, end_date = get_FixedDate(start_date, end_date)

    threshold_percentages = {}
    for idx_model, (model_name) in enumerate(commons.model_array):
        now_fetched = commons.fetch_between_dates(
            start_date, end_date, settings.MONITORINGDB_PATH + "db/threshold_data.db", model_name)[-1, 2:]

        threshold_pass = {}
        for idx_sensor, sensor_thre in enumerate(now_fetched):
            threshold_pass[commons.feature_set[idx_sensor]
                           ] = float(sensor_thre)

        threshold_percentages[idx_model] = threshold_pass

    counter_feature_s2, counter_feature_plot = commons.calc_counterPercentage(
        threshold_percentages)
    counter_feature_s2 = counter_feature_s2.keys()
    index_top10feat = []
    for feat_top in counter_feature_s2:
        index_top10feat.append(commons.feature_set.index(feat_top))

    # index_top10feat = list(reversed(index_top10feat))
    # ^ Get 10 Feature

    severity_trending_datas = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    sensor_datas = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "original_sensor")

    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:][:, index_top10feat].astype(
        float)
    sensor_datas = sensor_datas[:, 2:][:, index_top10feat].astype(float)
    sensor_statistic_current = param_statistic[:, index_top10feat]

    # window_size = 15
    # kernel = np.ones(window_size) / window_size

    # for i in range(sensor_datas.shape[-1]):
    #     sensor_datas[:, i] = np.convolve(sensor_datas[:, i], kernel, mode='same')

    return counter_feature_s2, data_timestamp, severity_trending_datas, sensor_datas, sensor_statistic_current


def get_sensorNtrend(start_date, end_date):
    severity_trending_datas = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    sensor_datas = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "original_sensor")

    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:].astype(float)
    sensor_datas = sensor_datas[:, 2:].astype(float)

    window_size = 15
    kernel = np.ones(window_size) / window_size

    for i in range(len(commons.feature_set)):
        sensor_datas[:, i] = np.convolve(
            sensor_datas[:, i], kernel, mode='same')

    return data_timestamp, severity_trending_datas, sensor_datas


def get_advisoryTable(start_date, end_date):  # 2529
    # Try Priority
    start_dateLate = datetime.strptime(
        end_date, "%Y-%m-%dT%H:%M:%S") - timedelta(days=30)
    severity_trending_datas = commons.fetch_between_dates(
        start_dateLate, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    sensor_datas = commons.fetch_between_dates(
        start_dateLate, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "original_sensor")
    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:].astype(float)
    sensor_datas = sensor_datas[:, 2:].astype(float)
    for i in range(len(commons.feature_set)):
        severity_trending_datas[:, i] = commons.hampel_filter(
            severity_trending_datas[:, i], window_size=93, n_sigmas=3)
    priority_parameter = {}
    datetime_index = pd.to_datetime(data_timestamp)
    for idx, feature_name in enumerate(commons.feature_set):
        series = pd.Series(
            severity_trending_datas[:, idx], index=datetime_index)
        if len(series) >= 400:
            series = series[~series.index.duplicated(keep='first')]
            series = series.asfreq('15min').ffill()
            result = seasonal_decompose(
                series, model='additive', period=96 * 2)
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

    raw_trending_datas = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    raw_trending_datas = raw_trending_datas[::-1]
    data_timestamp = raw_trending_datas[:, 1]
    severity_trending_datas = raw_trending_datas[:, 2:].astype(float)

    vectorized_func = np.vectorize(commons.percentage2severity)
    severity_level_datas = vectorized_func(severity_trending_datas)

    severity_counter_overyear = commons.fetch_column_threshold_counts(
        start_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings", threshold=5)

    sever_1week_featname = {}
    sever_featname = {}
    sever_count_featname = {}
    for idx, feature_name in enumerate(commons.feature_set):
        sever_featname[feature_name] = int(severity_level_datas[-1, idx])
        sever_1week_featname[feature_name] = severity_level_datas[:, idx]

        temp_severity_counts = Counter(severity_level_datas[:, idx])
        for level in range(1, 7):
            temp_severity_counts.setdefault(level, 0)
        sever_count_featname[feature_name] = {
            int(k): v for k, v in temp_severity_counts.items()}

    last_severity_featname = dict(
        sorted(sever_featname.items(), key=lambda item: item[1], reverse=True))
    sever_1week_featname = commons.order_objects_by_keys(
        sever_1week_featname.copy(), last_severity_featname.keys())
    return data_timestamp[-1], last_severity_featname, sever_1week_featname, sever_count_featname, severity_counter_overyear, priority_parameter


def get_advisoryDetail(start_date, end_date, sensor_id, feat_correlate):
    severity_trending_datas = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "severity_trendings")
    sensor_datas = commons.fetch_between_dates(
        start_date, end_date, settings.MONITORINGDB_PATH + "db/severity_trendings.db", "original_sensor")

    data_timestamp = sensor_datas[:, 1]
    severity_trending_datas = severity_trending_datas[:, 2:]
    sensor_datas = sensor_datas[:, 2:]

    shutdown_periods = commons.process_shutdownTimestamp(
        data_timestamp, sensor_datas)

    selected_severity_trending_datas = severity_trending_datas[:, sensor_id].astype(
        float)
    selected_sensor_datas = sensor_datas[:, sensor_id].astype(float)

    window_size = 30
    kernel = np.ones(window_size) / window_size
    selected_severity_trending_datas = np.convolve(
        selected_severity_trending_datas, kernel, mode='same')

    correlation_nowparam = correlation_param[commons.feature_set[sensor_id]]
    correlate_sensor_datas = sensor_datas[:, feat_correlate].astype(float)
    correlate_trending_datas = severity_trending_datas[:, feat_correlate].astype(
        float)

    return data_timestamp, selected_severity_trending_datas, selected_sensor_datas, shutdown_periods, correlation_nowparam, correlate_sensor_datas, correlate_trending_datas
