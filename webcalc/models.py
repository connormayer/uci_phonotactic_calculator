from django.db import models

from src import ngram_calculator as calc
from os import listdir
from os.path import join, basename, relpath
from webcalc_project import settings

from datetime import datetime

# When making changes to this model, comment/uncomment
#   default_objects in UploadTrain model.
class DefaultFile(models.Model):
    training_file = models.FileField(upload_to='default')
    description = models.CharField(max_length=200)
    short_desc = models.CharField(max_length=50, default="")

    @property
    def file_name(self):
        cur_name = self.training_file.name
        return cur_name.replace('default/', '')


class UploadTrain(models.Model):
    # Removed default_objects = DefaultFile.objects.all()
    # Removed files_list = [(x.file_name, x.short_desc) for x in default_objects]

    training_file = models.FileField(upload_to='uploads', blank=True)
    default_training_file = models.CharField(max_length=200, blank=True)
    test_file = models.FileField(upload_to='uploads')
    models_list = [('simple', 'Unigram/Bigram Scores'), ('complex', 'RNN Model')]
    training_model = models.CharField(choices=models_list, max_length=128)
    timeStr = datetime.now().strftime('%m_%d_%H%M')
    out_file = "outfile_" + timeStr + ".csv"

    def get_default_files_list(self):
        """
        Call this method (e.g., in a view or form) when you need the choices.
        """
        default_objects = DefaultFile.objects.all()
        return [(x.file_name, x.short_desc) for x in default_objects]

    def save(self, *args, **kwargs):
        return super(UploadTrain, self).save(*args, **kwargs)


class UploadWithDefault(models.Model):
    # Removed default_objects = DefaultFile.objects.all()
    # Removed files_list = [(x.file_name, x.file_name) for x in default_objects]

    training_file = models.CharField(max_length=200)
    test_file = models.FileField(upload_to='uploads')
    models_list = [
        ('unigram', 'Unigram Probability'),
        ('bigram', 'Bigram Probability'),
        ('posUnigram', 'Positional Unigram Score'),
        ('posBigram', 'Positional Bigram Score')
    ]
    training_model = models.CharField(choices=models_list, max_length=128)
    timeStr = datetime.now().strftime('%m_%d_%H%M')
    out_file = "outfile_" + timeStr + ".csv"

    def get_default_files_list(self):
        """
        Similar helper method for dynamic retrieval of DefaultFile objects.
        """
        default_objects = DefaultFile.objects.all()
        return [(x.file_name, x.file_name) for x in default_objects]