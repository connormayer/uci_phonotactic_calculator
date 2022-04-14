from django.urls import path
from .views import UploadTrainView, MediaView

urlpatterns = [
    path('', UploadTrainView.as_view(), name='home'),
    path('media', MediaView.as_view(), name='media')
]