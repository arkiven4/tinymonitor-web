from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import pickle

def index(request):
    return render(request, "home/index.html", {})

def dashboard(request):
    return render(request, "home/dashboard.html", {})

def kpi(request):
    return render(request, "kpi/kpi.html", {})

def kpi_updatenoe(request):
    if request.method == 'POST' and request.FILES.get('file'):
        if not request.FILES.get('file'):
            return JsonResponse({'message': 'No file uploaded', 'type': 'danger'})
        
        uploaded_file = request.FILES['file']
        try:
            df = pd.read_excel(uploaded_file)
            df['Start Date'] = pd.to_datetime(df['Start Date'])
            df['Year'] = df['Start Date'].dt.year
            category_counts = df.groupby(['Year', 'Category']).size().unstack(fill_value=0)
            final_data = {'years': list(category_counts.index), 'data': [ { 'label': col, 'data': category_counts[col].tolist(), } for i, col in enumerate(category_counts.columns) ]}
            with open('db/number_of_event.pickle', 'wb') as handle:
                pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return JsonResponse({'message': 'File read successfully', 'type': 'success'})
        except Exception as e:
            return JsonResponse({'message': f'Error reading file: {str(e)}', 'type': 'danger'})
        
    return render(request, "kpi/updatenoe.html", {})

def advisory(request):
    return render(request, "advisory/advisory.html", {})

def advisory_chart(request):
    return render(request, "advisory/advisory_chart.html", {})

def charts(request):
    return render(request, "charts/charts.html", {})

def settings(request):
    return render(request, "settings/settings.html", {})
