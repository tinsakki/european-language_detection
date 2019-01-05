from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path(r'^admin/', admin.site.urls),
    # url(r'^about/$', views.about),
    path('', views.prediction,name='predict'),

]