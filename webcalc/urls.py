from django.urls import path
from .views import UploadTrainView, MediaView, OutputView

urlpatterns = [
    path('', UploadTrainView.as_view(), name='home'),
    path('media', MediaView.as_view(), name='media'),
    path('output', OutputView.as_view(), name='output')
]