from django.contrib import admin

from .models import DefaultFile, UploadTrain

admin.site.register(UploadTrain)
admin.site.register(DefaultFile)
