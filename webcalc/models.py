from django.db import models

from src import ngram_calculator as calc
from os.path import join, basename, relpath
from webcalc_project import settings

from datetime import datetime


class DefaultFile(models.Model):
    training_file = models.FileField(upload_to='default')
    description = models.CharField(max_length=200)

    @property
    def file_name(self):
        cur_name = self.training_file.name
        return cur_name.replace('default/', '')

class UploadTrain(models.Model):
    # upload files go to media\uploads
    default_objects = DefaultFile.objects.all()
    files_list = [(x.file_name, x.file_name) for x in default_objects]
    training_file = models.FileField(upload_to='uploads', blank=True)
    default_training_file = models.CharField(choices=files_list, max_length=200, blank=True)
    test_file = models.FileField(upload_to='uploads')
    models_list = [('unigram', 'Unigram Probability'), ('bigram', 'Bigram Probability'), \
        ('posUnigram', 'Positional Unigram Score'), ('posBigram', 'Positional Bigram Score')]
    training_model = models.CharField(choices=models_list, max_length=128)
    timeStr = datetime.now().strftime('%m_%d_%H%M')
    #media_path = settings.MEDIA_ROOT
    out_file = "outfile_" + timeStr + ".csv"
    #out_file = None

    def save(self, *args, **kwargs):
        # if not self.pk:
        #     media_path = settings.MEDIA_ROOT
        #     train_file = join(media_path, basename(self.training_file.name))
        #     test_file = join(media_path, basename(self.test_file.name))
        #     out_file = join(media_path, "outfile.csv")
        #     calc.run(train_file, test_file, out_file)
        return super(UploadTrain, self).save(*args, **kwargs)


class UploadWithDefault(models.Model):
    default_objects = DefaultFile.objects.all()
    files_list = [(x.file_name, x.file_name) for x in default_objects]
    training_file = models.CharField(choices=files_list, max_length=200)
    test_file = models.FileField(upload_to='uploads')
    models_list = [('unigram', 'Unigram Probability'), ('bigram', 'Bigram Probability'), \
        ('posUnigram', 'Positional Unigram Score'), ('posBigram', 'Positional Bigram Score')]
    training_model = models.CharField(choices=models_list, max_length=128)

    timeStr = datetime.now().strftime('%m_%d_%H%M')
    #media_path = settings.MEDIA_ROOT
    out_file = "outfile_" + timeStr + ".csv"
    #out_file = None
