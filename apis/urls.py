from django.urls import path
from apis import views

urlpatterns = [
    path('severity_plot', views.severity_plot, name='severity_plot'),
    path('advisory_table', views.advisory_table, name='advisory_table'), 
    path('advisory_detail/<int:feat_id>', views.advisory_detail, name='advisory_detail'),
    path('top10_charts', views.top10_charts, name='top10_charts'),
] 