from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import pandas as pd
import pickle

def index(request):
    return render(request, "home/index.html", {})

def dashboard(request):
    return render(request, "home/dashboard.html", {})

def kpi(request):
    return render(request, "kpi/kpi.html", {})

def kpi_units(request):
    return render(request, "kpi/kpi_units.html", {})

def kpi_manualdata(request):
    if request.method == 'POST' and request.FILES.get('file'):
        if not request.FILES.get('file'):
            return JsonResponse({'message': 'No file uploaded', 'type': 'danger'})
        
        uploaded_file = request.FILES['file']
        try:
            if request.POST.dict()['type_data'] == "noe":
                df = pd.read_excel(uploaded_file)
                df['Start'] = pd.to_datetime(df['Start Date'] + ' ' + df['Start Time'])
                df['End'] = pd.to_datetime(df['End Date'] + ' ' + df['End Time'])
                df.to_pickle(settings.MONITORINGDB_PATH + 'db/number_of_event.pickle')
            elif request.POST.dict()['type_data'] == "other_kpi":
                df = pd.read_excel(uploaded_file)
                df['Month_num'] = pd.to_datetime(df['Month'], format='%B').dt.month
                df['Start'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month_num'].astype(str) + '-01') \
                                + pd.to_timedelta((df['Week']-1)*7, unit='d')
                df['End'] = df['Start'] + pd.Timedelta(days=6)
                df = df.drop(columns=['Year', 'Month', 'Week', 'Month_num'])
                df.to_pickle(settings.MONITORINGDB_PATH + 'db/other_kpis.pickle')
            return JsonResponse({'message': 'File read successfully', 'type': 'success'})
        except Exception as e:
            return JsonResponse({'message': f'Error reading file: {str(e)}', 'type': 'danger'})
        
    return render(request, "kpi/update_manualdata.html", {})

def advisory(request):
    return render(request, "advisory/advisory.html", {})

def advisory_chart(request):
    return render(request, "advisory/advisory_chart.html", {})

def charts(request):
    return render(request, "charts/charts.html", {})


def settings_page(request):
    return render(request, "settings/settings.html", {})
