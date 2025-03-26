from django.urls import path
from pages import views

urlpatterns = [
    path("", views.index, name="index"),
    path("advisory/", views.advisory, name="advisory"),
    path("advisory/chart", views.advisory_chart, name="advisory_chart"),
    path("charts/", views.charts, name="charts"),
]