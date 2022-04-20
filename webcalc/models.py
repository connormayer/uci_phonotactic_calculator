from django.db import models

from src import ngram_calculator as calc
from os.path import join, basename
from webcalc_project import settings

class UploadTrain(models.Model):
    training_file = models.FileField()
    test_file = models.FileField()
    models_list = [('unigram', 'Unigram Probability'), ('bigram', 'Bigram Probability'), \
        ('posUnigram', 'Positional Unigram Score'), ('posBigram', 'Positional Bigram Score')]
    training_model = models.CharField(choices=models_list, max_length=128)

    def save(self, *args, **kwargs):
        if not self.pk:
            media_path = settings.MEDIA_ROOT
            train_file = join(media_path, basename(self.training_file.name))
            test_file = join(media_path, basename(self.test_file.name))
            out_file = join(media_path, "outfile.csv")
            calc.run(train_file, test_file, out_file)
        return super(UploadTrain, self).save(*args, **kwargs)
