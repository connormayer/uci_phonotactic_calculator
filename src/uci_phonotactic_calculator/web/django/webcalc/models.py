"""Models for the UCI Phonotactic Calculator web interface."""

from datetime import datetime
from os import listdir
from os.path import isfile, join

from django.db import models

from uci_phonotactic_calculator.web.django import settings


class DefaultFile(models.Model):
    """Default training file model."""

    training_file = models.FileField(upload_to="default")
    description = models.CharField(max_length=200)
    short_desc = models.CharField(max_length=50, default="")

    @property
    def file_name(self):
        """Get the file name without the 'default/' prefix."""
        cur_name = self.training_file.name
        return cur_name.replace("default/", "")


class UploadTrain(models.Model):
    """Model for uploaded training and test files."""

    # Get file choices from the actual files in media/default/
    def get_file_choices():
        """Get choices for default training files."""
        media_path = settings.MEDIA_ROOT
        default_dir = join(media_path, "default")
        try:
            files = [f for f in listdir(default_dir) if isfile(join(default_dir, f))]
            # Map filenames to human-readable descriptions
            file_descriptions = {
                "english.csv": "English ARPABET",
                "english_freq.csv": "English ARPABET Frequencies",
                "english_needle.csv": "English ARPABET Needle",
                "english_onsets.csv": "English Onsets ARPABET",
                "finnish.csv": "Finnish Ortho",
                "french.csv": "French IPA",
                "polish_onsets.csv": "Polish Onsets IPA",
                "samoan.csv": "Samoan IPA",
                "spanish_stress.csv": "Spanish IPA Stress",
                "turkish.csv": "Turkish IPA",
            }
            return [(f, file_descriptions.get(f, f)) for f in files]
        except (FileNotFoundError, OSError):
            # Fallback if directory can't be read
            return []

    # upload files go to media\uploads
    training_file = models.FileField(upload_to="uploads", blank=True)
    default_training_file = models.CharField(
        choices=get_file_choices, max_length=200, blank=True
    )
    test_file = models.FileField(upload_to="uploads")
    training_model = models.CharField(default="simple", max_length=128)
    current_time = models.DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        """Save the model."""
        return super(UploadTrain, self).save(*args, **kwargs)


class UploadWithDefault(models.Model):
    """Model for uploads using a default training file."""

    default_objects = []  # DefaultFile.objects.all()
    files_list = [(x.file_name, x.file_name) for x in default_objects]
    training_file = models.CharField(choices=files_list, max_length=200)
    test_file = models.FileField(upload_to="uploads")
    training_model = models.CharField(default="simple", max_length=128)
    timeStr = datetime.now().strftime("%m_%d_%H%M")
    out_file = "outfile_" + timeStr + ".csv"
