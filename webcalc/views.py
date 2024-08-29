from audioop import reverse
from email.policy import default
from typing import Any, Dict
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
from src.rnn_src import main as rnn

# Create your views here.
from .models import UploadTrain, DefaultFile, UploadWithDefault, GroupDefaultDatasets

class UploadTrainView(CreateView):
    model = UploadTrain
    fields = '__all__'#['training_file', 'test_file', 'training_model']
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
        form.data['training_model'] = '' # reset training model dropdown selection in form
        context['form'] = form # set form in context to updated form
        #messages.warning(self.request, 'Bad file formatting')
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
        old_outfile_name = (self.model.objects.last()).out_file
        new_outfile_name = old_outfile_name.replace('.csv', '') + '_' + test_file_name_sub + '.csv'

        out_file = join(media_path, 'uploads', basename(new_outfile_name))
        
        model = (self.model.objects.last()).training_model
        if model == 'simple':
            calc.run(train_file, test_file, out_file)
        elif model == 'complex':
            rnn.run(train_file, test_file, out_file)#print('run rnn')

        return response

class MediaView(CreateView):
    model = GroupDefaultDatasets
    template_name = 'media.html'
    fields = '__all__'
    success_url = reverse_lazy('mediaGrouped')

    # TODO: ADD FIELD THAT ALLOWS USERS TO SPECIFY GROUPBY METHOD

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        #media_path = join(settings.MEDIA_ROOT, 'default')
        #files = [f for f in listdir(media_path) if isfile(join(media_path, f))]
        #context['myfiles'] = files

        # TODO: CHANGE THIS TO A UTIL FUNCTION THAT GETS FILES GROUPED AS NECESSARY
        context['objects'] = util.get_default_files() #self.model.objects.all()
        return context

    def form_valid(self, form):
        response = super(MediaView, self).form_valid(form)
        return response

class GroupedMediaView(TemplateView):
    template_name = 'mediaGrouped.html'
    model = GroupDefaultDatasets
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        grouping_method = (self.model.objects.last()).grouping_method

        context['grouping_method'] = "\"" + grouping_method[0].upper() + grouping_method[1:] + "\""
        context['objects'] = util.group_datasets(util.get_default_files(), grouping_method)
        return context

class DescriptionsView(TemplateView):
    template_name = 'descriptions.html'
    model = GroupDefaultDatasets

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['objects'] = util.get_default_files()

        return context

class OutputView(TemplateView):
    model = UploadTrain
    #fields = ['training_file', test]
    template_name = 'output.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        test_file_name_sub = basename((self.model.objects.last()).test_file.name)[:4]
        old_outfile_name = (self.model.objects.last()).out_file
        new_outfile_name = old_outfile_name.replace('.csv', '') + '_' + test_file_name_sub + '.csv'

        context['output_file'] = new_outfile_name#(self.model.objects.last()).out_file
        return context

class AboutView(TemplateView):
    template_name = 'about.html'


class UploadDefaultView(CreateView):
    model = UploadWithDefault
    fields = '__all__'
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
        old_outfile_name = (self.model.objects.last()).out_file
        new_outfile_name = old_outfile_name.replace('.csv', '') + '_' + test_file_name_sub + '.csv'

        out_file = join(media_path, 'uploads', basename(new_outfile_name))
        calc.run(train_file, test_file, out_file)

        # clear media folder here

        return response