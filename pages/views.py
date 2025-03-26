from django.shortcuts import render

def index(request):
    return render(request, "index.html", {})

def advisory(request):
    return render(request, "advisory.html", {})

def advisory_chart(request):
    return render(request, "advisory_chart.html", {})

def charts(request):
    return render(request, "charts.html", {})
