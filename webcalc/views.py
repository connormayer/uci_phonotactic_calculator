from django.shortcuts import render
from django.views.generic.edit import CreateView
from django.urls import reverse_lazy

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
