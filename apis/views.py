from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response 
from django.http import JsonResponse
from rest_framework import status
from datetime import datetime, timedelta

from apis.helper import get_sensorNtrend, get_severityNTrend, get_advisoryTable, get_advisoryDetail, get_top10Charts

@api_view(['GET'])
def severity_plot(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    counter_feature_s2, df_timestamp, df_feature_send, y_pred_send, loss_send, thr_now_model = get_severityNTrend(start_date, end_date)
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
    start_date = None
    end_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    counter_feature_s2, data_timestamp, severity_trending_datas, sensor_datas = get_top10Charts(start_date, end_date)
    data = {
        'counter_feature_s2': counter_feature_s2,
        'df_timestamp': data_timestamp.tolist(),
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
def advisory_detail(request, feat_id=0, minusdays=7):
    end_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    start_date = (datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S") - timedelta(days=minusdays)).strftime("%Y-%m-%dT%H:%M:%S")

    data_timestamp, severity_trending_datas, sensor_datas = get_advisoryDetail(start_date, end_date, feat_id)
    data = {
        'feat_id': feat_id,
        'data_timestamp': data_timestamp.tolist(),
        'severity_trending_datas': severity_trending_datas.tolist(),
        'sensor_datas': sensor_datas.tolist(),
    }
    return Response(data, status=status.HTTP_200_OK)



