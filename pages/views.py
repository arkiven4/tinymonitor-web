from django.shortcuts import render

def index(request):
    return render(request, "home/index.html", {})

def dashboard(request):
    return render(request, "home/dashboard.html", {})

def kpi(request):
    return render(request, "home/kpi.html", {})

def advisory(request):
    return render(request, "advisory/advisory.html", {})

def advisory_chart(request):
    return render(request, "advisory/advisory_chart.html", {})

def charts(request):
    return render(request, "charts/charts.html", {})

def settings(request):
    return render(request, "settings/settings.html", {})
