from django.urls import path
from .views import UploadTrainView, MediaView, OutputView, AboutView, UploadDefaultView, GroupedMediaView, DescriptionsView

urlpatterns = [
    path('', UploadTrainView.as_view(), name='home'),
    path('datasets', MediaView.as_view(), name='media'),
    path('output', OutputView.as_view(), name='output'),
    path('about', AboutView.as_view(), name='about'),
    path('upload-default', UploadDefaultView.as_view(), name='uploadDefault'),
    path('datasets-grouped', GroupedMediaView.as_view(), name='mediaGrouped'),
    path('descriptions', DescriptionsView.as_view(), name='descriptions')
]