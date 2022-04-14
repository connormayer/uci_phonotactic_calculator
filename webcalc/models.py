from django.db import models

class UploadTrain(models.Model):
    training_file = models.FileField()
    test_file = models.FileField()
    models_list = [('unigram', 'Unigram Probability'), ('bigram', 'Bigram Probability'), \
        ('posUnigram', 'Positional Unigram Score'), ('posBigram', 'Positional Bigram Score')]
    training_model = models.CharField(choices=models_list, max_length=128)