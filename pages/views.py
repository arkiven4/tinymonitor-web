from django.shortcuts import render

def index(request):
    return render(request, "home/index.html", {})

def advisory(request):
    return render(request, "advisory/advisory.html", {})

def advisory_chart(request):
    return render(request, "advisoryadvisory_chart.html", {})

def charts(request):
    return render(request, "charts/charts.html", {})

def settings(request):
    return render(request, "settings/settings.html", {})
