from django.urls import path
from apis import views

urlpatterns = [
    path('panel_summary', views.panel_summary, name='panel_summary'),
    path('zone_distribution', views.zone_distribution, name='zone_distribution'),
    path('zone_distributionTimeline', views.zone_distributionTimeline, name='zone_distributionTimeline'),
    path('unitStatus', views.unitStatus, name='unitStatus'),
    path('kpi', views.kpi, name='kpi'),
    path('severity_plot', views.severity_plot, name='severity_plot'),
    path('advisory_table', views.advisory_table, name='advisory_table'), 
    path('advisory_detail/<int:feat_id>', views.advisory_detail, name='advisory_detail'),
    path('top10_charts', views.top10_charts, name='top10_charts'),
    path('timeinfo_detail', views.timeinfo_detail, name='timeinfo_detail'),
] 