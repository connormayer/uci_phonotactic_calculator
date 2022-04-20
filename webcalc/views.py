from django.shortcuts import render
from django.views.generic.edit import CreateView
from django.views.generic.base import TemplateView
from django.urls import reverse_lazy
from webcalc_project import settings

from os import listdir
from os.path import isfile, join, basename

from src import ngram_calculator as calc

# Create your views here.
from .models import UploadTrain

class UploadTrainView(CreateView):
    model = UploadTrain
    fields = ['training_file', 'test_file', 'training_model']
    template_name = 'home.html'
    success_url = reverse_lazy('output')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['documents'] = UploadTrain.objects.all()
        return context

    def form_valid(self, form):
        response = super(UploadTrainView, self).form_valid(form)
        media_path = settings.MEDIA_ROOT
        train_file = join(media_path, basename((self.model.objects.last()).training_file.name))
        test_file = join(media_path, basename((self.model.objects.last()).test_file.name))
        out_file = join(media_path, "outfile.csv")
        calc.run(train_file, test_file, out_file)
        return response

class MediaView(TemplateView):
    template_name = 'media.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        media_path = settings.MEDIA_ROOT
        files = [f for f in listdir(media_path) if isfile(join(media_path, f))]
        context['myfiles'] = files
        return context

class OutputView(TemplateView):
    model = UploadTrain
    #fields = ['training_file', test]
    template_name = 'output.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['output_file'] = 'outfile.csv'#(self.model.objects.last()).test_file.name
        return context