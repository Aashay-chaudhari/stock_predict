from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [
      path('', views.home, name='home'),
      path('chart', views.chart, name='chart'),
      path('calculatecalls1', views.calculatecalls, name='calculatecalls'),
      path('symbol/<symbol>', views.symbol_data),
]