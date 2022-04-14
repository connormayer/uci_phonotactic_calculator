from django.shortcuts import render
from django.views.generic.edit import CreateView
from django.views.generic.base import TemplateView
from django.urls import reverse_lazy
from webcalc_project import settings

from os import listdir
from os.path import isfile, join

# Create your views here.
from .models import UploadTrain

class UploadTrainView(CreateView):
    model = UploadTrain
    fields = ['training_file', 'test_file', 'training_model']
    template_name = 'home.html'
    success_url = reverse_lazy('home')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['documents'] = UploadTrain.objects.all()
        return context

class MediaView(TemplateView):
    template_name = 'media.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        media_path = settings.MEDIA_ROOT
        files = [f for f in listdir(media_path) if isfile(join(media_path, f))]
        context['myfiles'] = files
        return context