"""URL patterns for the webcalc Django app."""

from django.urls import path

from . import views

app_name = "webcalc"

urlpatterns = [
    path("", views.UploadTrainView.as_view(), name="home"),
    path("about/", views.AboutView.as_view(), name="about"),
    path("media/", views.MediaView.as_view(), name="media"),
    path("output/", views.OutputView.as_view(), name="output"),
    path("uploadDefault/", views.UploadDefaultView.as_view(), name="uploadDefault"),
]
