from audioop import reverse
from email.policy import default
from django.shortcuts import render
from django.views.generic.edit import CreateView
from django.views.generic.base import TemplateView
from django.urls import reverse_lazy
from webcalc_project import settings

from django.contrib import messages

from os import listdir
from os.path import isfile, join, basename

from src import ngram_calculator as calc
from src import utility as util

# Create your views here.
from .models import UploadTrain, DefaultFile, UploadWithDefault

class UploadTrainView(CreateView):
    model = UploadTrain
    fields = ['training_file', 'default_training_file', 'test_file']
    template_name = 'home.html'
    success_url = reverse_lazy('output')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['documents'] = UploadTrain.objects.all() # change to self.model.objects.all() ?
        return context

    def form_invalid(self, form):
        response = super(UploadTrainView, self).form_invalid(form)
        context = self.get_context_data()
        form.data = form.data.copy()  # make copy of form data
        form.data['default_training_file'] = ''
        context['form'] = form # set form in context to updated form
        return response

    def form_valid(self, form):
        response = super(UploadTrainView, self).form_valid(form)
        util.clean_media_folder()
        
        media_path = settings.MEDIA_ROOT

        uploaded_training = ((self.model.objects.last()).training_file.name != '')
        default_training = ((self.model.objects.last()).default_training_file != '')

        # This may not be needed
        if uploaded_training and default_training:
            messages.warning(self.request, 'Either upload a training file OR use a default one (not both)')
            return self.form_invalid(form)

        if not uploaded_training:
            if not default_training:
                messages.warning(self.request, 'Please upload a training file or select a default file')
                return self.form_invalid(form)
            else:
                train_file = join(media_path, 'default', basename((self.model.objects.last()).default_training_file))
        else:
            train_file = join(media_path, 'uploads', basename((self.model.objects.last()).training_file.name))

        
        test_file = join(media_path, 'uploads', basename((self.model.objects.last()).test_file.name))

        # Validate training and test files here
        # If not valid, return form_invalid without calling run
        ###########

        train_success, message = util.valid_file(train_file)
        if not train_success:
            messages.warning(self.request, 'Invalid training file format: {}'.format(message))
            return self.form_invalid(form)
        test_success, message = util.valid_file(test_file)
        if not test_success:
            messages.warning(self.request, 'Invalid test file format: {}'.format(message))
            return self.form_invalid(form)

        test_file_name_sub = basename((self.model.objects.last()).test_file.name)[:4]
        upload_timestamp = self.model.objects.last().current_time.timestamp()
        timestamp_str = str(upload_timestamp).replace('.', '_')
        new_outfile_name = 'outfile_' + test_file_name_sub + '_' + timestamp_str + '.csv'
        out_file = join(media_path, 'uploads', basename(new_outfile_name))
        
        calc.run(train_file, test_file, out_file)

        return response

class MediaView(TemplateView):
    template_name = 'media.html'
    model = DefaultFile

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['objects'] = self.model.objects.all()
        return context

class OutputView(TemplateView):
    model = UploadTrain
    template_name = 'output.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        test_file_name_sub = basename((self.model.objects.last()).test_file.name)[:4]
        upload_timestamp = self.model.objects.last().current_time.timestamp()
        timestamp_str = str(upload_timestamp).replace('.', '_')
        new_outfile_name = 'outfile_' + test_file_name_sub + '_' + timestamp_str + '.csv'
        context['output_file'] = new_outfile_name
        return context

class AboutView(TemplateView):
    template_name = 'about.html'


class UploadDefaultView(CreateView):
    model = UploadWithDefault
    fields = ['training_file', 'default_training_file', 'test_file']
    template_name = 'uploadDefault.html'
    success_url = reverse_lazy('output')

    def form_invalid(self, form):
        response = super(UploadDefaultView, self).form_invalid(form)
        context = self.get_context_data()
        form.data = form.data.copy()  # make copy of form data
        form.data['training_model'] = '' # reset training model dropdown selection in form
        form.data['training_file'] = '' # reset training file dropdown selection in form
        context['form'] = form # set form in context to updated form
        messages.warning(self.request, 'Bad file formatting')
        return response

    def form_valid(self, form):
        response = super(UploadDefaultView, self).form_valid(form)

        media_path = settings.MEDIA_ROOT
        
        train_file = join(media_path, 'default', basename((self.model.objects.last()).training_file))
        test_file = join(media_path, 'uploads', basename((self.model.objects.last()).test_file.name))

        # Validate test file here
        # If not valid, return form_invalid without calling run
        # No need to validate training file since it is default file
        ###########

        if not util.valid_file(test_file):
            return self.form_invalid(form)

        test_file_name_sub = basename((self.model.objects.last()).test_file.name)[:4]
        upload_timestamp = self.model.objects.last().current_time.timestamp()
        timestamp_str = str(upload_timestamp).replace('.', '_')
        new_outfile_name = 'outfile_' + test_file_name_sub + '_' + timestamp_str + '.csv'
        out_file = join(media_path, 'uploads', basename(new_outfile_name))
        calc.run(train_file, test_file, out_file)

        return response