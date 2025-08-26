from django.urls import path
from pages import views

urlpatterns = [
    path("", views.index, name="index"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("kpi/", views.kpi, name="kpi"),
    path("kpi_units/", views.kpi_units, name="kpi_units"),
    path("kpi/update_manualdata", views.kpi_manualdata, name="kpi_manualdata"),
    path("advisory/", views.advisory, name="advisory"),
    path("advisory/chart", views.advisory_chart, name="advisory_chart"),
    path("charts/", views.charts, name="charts"),
    path("settings/", views.settings_page, name="settings"),
]