from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework import status
from datetime import datetime, timedelta

import apis.helper as helper_fun


@api_view(['GET'])
def panel_summary(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    last_timestamp, last_sensor_featname, sensor_featname, last_severity_featname, sever_featname, ordered_feature_name, sever_count_featname, priority_parameter = helper_fun.get_PanelSummary(
        start_date, end_date)

    data = {
        'last_timestamp': last_timestamp,
        'last_sensor_featname': last_sensor_featname,
        'sensor_featname': sensor_featname,
        'last_severity_featname': last_severity_featname,
        'sever_featname': sever_featname,
        'ordered_feature_name': ordered_feature_name,
        'sever_count_featname': sever_count_featname,
        'priority_parameter': priority_parameter
    }

    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def zone_distribution(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    tags_param = request.GET.get('tags')
    tags = tags_param.split(",") if tags_param else None

    operation_mode, operation_zone = helper_fun.get_OperationDistribution(
        start_date, end_date, tags)

    data = {
        'operation_mode': dict(operation_mode),
        'operation_zone': dict(operation_zone),
    }

    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def zone_distributionTimeline(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    tags_param = request.GET.get('tags')
    tags = tags_param.split(",") if tags_param else None

    data_timestamp, load_datas, grid_datas = helper_fun.get_OperationDistributionTimeline(
        start_date, end_date, tags)

    data = {
        'data_timestamp': data_timestamp,
        'load_datas': load_datas,
        'grid_datas': grid_datas,
    }

    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def unitStatus(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    tags = ['LGS1', 'LGS2', 'LGS3', 'BGS1', 'BGS2', 'KGS1', 'KGS2']

    status_dict = helper_fun.get_units_status(
        start_date, end_date, tags)

    data = {
        'status_dict': status_dict,
    }

    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def kpi(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    tags_param = request.GET.get('tags')
    tags = tags_param.split(",") if tags_param else None
    noe_metric = request.GET.get('noe_metric') or "noe"

    kpi_datas = helper_fun.get_KPIData(
        start_date, end_date, tags, noe_metric)

    # data = {
    #     'oee': oee,
    #     # 'phy_avail': phy_avail,
    #     # 'performance': performance,
    #     # 'uo_Avail': uo_Avail,
    #     # 'data_timestamp': data_timestamp,
    # }
    return Response(kpi_datas, status=status.HTTP_200_OK)

@api_view(['GET'])
def severity_plot(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    counter_feature_s2, df_timestamp, df_feature_send, y_pred_send, loss_send, thr_now_model = helper_fun.get_SeverityNLoss(
        start_date, end_date)
    data = {
        'counter_feature_s2': counter_feature_s2,
        'df_timestamp': df_timestamp.tolist(),
        'df_feature_send': df_feature_send.tolist(),
        'y_pred_send': y_pred_send.tolist(),
        'loss_send': loss_send.tolist(),
        'thr_now_model': thr_now_model,
    }

    return Response(data, status=status.HTTP_200_OK)


@api_view(['GET'])
def top10_charts(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    counter_feature_s2, data_timestamp, severity_trending_datas, sensor_datas, sensor_statistic_current = helper_fun.get_top10Charts(
        start_date, end_date)
    data = {
        'counter_feature_s2': counter_feature_s2,
        'df_timestamp': data_timestamp.tolist(),
        'severity_trending_datas': severity_trending_datas.tolist(),
        'sensor_datas': sensor_datas.tolist(),
        'sensor_statistic_current': sensor_statistic_current.tolist(),
    }
    return Response(data, status=status.HTTP_200_OK)


@api_view(['GET'])
def advisory_table(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    last_timestamp, last_severity_featname, sever_1week_featname, sever_count_featname, severity_counter_overyear, priority_parameter = helper_fun.get_advisoryTable(
        start_date, end_date)
  
    data = {
        'last_timestamp': last_timestamp,
        'last_severity_featname': last_severity_featname,
        'sever_1week_featname': sever_1week_featname,
        'sever_count_featname': sever_count_featname,
        'severity_counter_overyear': severity_counter_overyear,
        'priority_parameter': priority_parameter
    }
    return Response(data, status=status.HTTP_200_OK)


@api_view(['GET'])
def advisory_detail(request, feat_id=0):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    feat_correlate_param = request.GET.get('feat_correlate', '')
    maximum_points = int(request.GET.get('maximum_points', 250)) if str(request.GET.get('maximum_points', '250')).isdigit() else 250
    
    try:
        feat_correlate = [int(i) for i in feat_correlate_param.split(',') if i.strip().isdigit()]
    except ValueError:
        feat_correlate = []

    data_timestamp, severity_trending_datas, priority_data, sensor_datas, shutdown_periods, correlation_nowparam, correlate_sensor_datas, correlate_trending_datas = helper_fun.get_advisoryDetail(
        start_date, end_date, feat_id, feat_correlate, maximum_points)
    data = {
        'feat_id': feat_id,
        'data_timestamp': data_timestamp.tolist(),
        'severity_trending_datas': severity_trending_datas.tolist(),
        'priority_data': priority_data,
        'sensor_datas': sensor_datas.tolist(),
        'shutdown_periods': shutdown_periods,
        'correlation_nowparam': correlation_nowparam,
        'correlate_sensor_datas': correlate_sensor_datas,
        'correlate_trending_datas': correlate_trending_datas
    }
    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def timeinfo_detail(request):
    datetime_last, next_update  = helper_fun.get_TimeInformastion()
    data = {
        'datetime_last': datetime_last,
        'next_update': next_update,
    }
    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def adjust_threshold_settings(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    datetime_last  = helper_fun.get_adjustthr(start_date, end_date)
    data = {
        'datetime_last': datetime_last,
    }
    return Response(data, status=status.HTTP_200_OK)
