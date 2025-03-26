from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response 
from django.http import JsonResponse
from rest_framework import status

from apis.helper import get_sensorNtrend, get_severityNTrend, get_advisoryTable, get_advisoryDetail, get_top10Charts

@api_view(['GET'])
def hello(request):
    start_date = "2021-04-28T06:15:00"
    end_date = "2021-05-28T06:15:00"
    data_timestamp, severity_trending_datas, sensor_datas = get_sensorNtrend(start_date, end_date)

    name = request.GET.get('name', 'guest')
    data = {
        'name': name,
        'message': f"Hello {name}, your first API endpoint has been created successfully!",
        'data_timestamp': data_timestamp.tolist(),
        'severity_trending_datas': severity_trending_datas.tolist(),
        'sensor_datas': sensor_datas.tolist(),
    }
    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def advisory_table(request):
    last_timestamp, last_severity_featname  = get_advisoryTable()
    data = {
        'last_timestamp': last_timestamp,
        'last_severity_featname': last_severity_featname
    }
    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def advisory_detail(request, feat_id=0):
    data_timestamp, severity_trending_datas, sensor_datas = get_advisoryDetail(feat_id)
    data = {
        'feat_id': feat_id,
        'data_timestamp': data_timestamp.tolist(),
        'severity_trending_datas': severity_trending_datas.tolist(),
        'sensor_datas': sensor_datas.tolist(),
    }
    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def severity_plot(request):
    start_date = None
    end_date = "2021-05-28T03:55:00" # datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    counter_feature_s2, df_timestamp, df_feature_send, y_pred_send = get_severityNTrend(start_date, end_date)
    data = {
        'counter_feature_s2': counter_feature_s2,
        'df_timestamp': df_timestamp.tolist(),
        'df_feature_send': df_feature_send.tolist(),
        'y_pred_send': y_pred_send.tolist(),
    }
    return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def top10_charts(request):
    start_date = None
    end_date = "2021-05-28T03:55:00" # datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    counter_feature_s2, data_timestamp, severity_trending_datas, sensor_datas = get_top10Charts(start_date, end_date)
    data = {
        'counter_feature_s2': counter_feature_s2,
        'df_timestamp': data_timestamp.tolist(),
        'severity_trending_datas': severity_trending_datas.tolist(),
        'sensor_datas': sensor_datas.tolist(),
    }
    return Response(data, status=status.HTTP_200_OK)


# def hello(request):
#     start_date = "2021-04-28T06:15:00"
#     end_date = "2021-05-28T06:15:00"
#     data_timestamp, severity_trending_datas, sensor_datas = get_sensorNtrend(start_date, end_date)

#     name = request.GET.get('name', 'guest')
#     data = {
#         'name': name,
#         'message': f"Hello {name}, your first API endpoint has been created successfully!",
#         'data_timestamp': data_timestamp.tolist(),
#         'severity_trending_datas': severity_trending_datas.tolist(),
#         'sensor_datas': sensor_datas.tolist(),
#     }
#     return Response(data, status=status.HTTP_200_OK)
