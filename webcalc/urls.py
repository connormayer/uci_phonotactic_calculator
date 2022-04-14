from django.urls import path
from .views import UploadTrainView

urlpatterns = [
    path('', UploadTrainView.as_view(), name='home'),
    path('media', UploadTrainView.as_view(), name='media')
]