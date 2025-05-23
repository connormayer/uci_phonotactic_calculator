from django.urls import path

from .views import AboutView, MediaView, OutputView, UploadDefaultView, UploadTrainView

urlpatterns = [
    path("", UploadTrainView.as_view(), name="home"),
    path("datasets", MediaView.as_view(), name="media"),
    path("output", OutputView.as_view(), name="output"),
    path("about", AboutView.as_view(), name="about"),
    path("upload-default", UploadDefaultView.as_view(), name="uploadDefault"),
]
