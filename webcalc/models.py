from django.db import models

class UploadTrain(models.Model):
    training_file = models.FileField()