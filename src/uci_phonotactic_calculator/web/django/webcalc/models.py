"""Models for the UCI Phonotactic Calculator web interface."""

from datetime import datetime
from django.utils import timezone
from os import listdir
from os.path import isfile, join
from typing import Any, cast, Dict, List, Tuple

from django.db import models

from uci_phonotactic_calculator.web.django import settings


def get_file_choices() -> List[Tuple[str, str]]:
    """Get choices for default training files."""
    media_path = settings.MEDIA_ROOT
    default_dir = join(media_path, "default")
    try:
        files = [f for f in listdir(default_dir) if isfile(join(default_dir, f))]
        # Map filenames to human-readable descriptions
        file_descriptions = {
            "english.csv": "A subset of the CMU Pronouncing Dictionary with CELEX frequencies > 1. This is notated in ARPABET. Numbers indicating vowel stress have been removed.",
            "english_freq.csv": "A subset of the CMU Pronouncing Dictionary with CELEX frequencies. This data is represented in ARPABET.",
            "english_needle.csv": "Data set from Needle et al. (2022). Consists of about 11,000 monomorphemic words from CELEX (Baayen et al. 1995) in ARPABET transcription.",
            "english_onsets.csv": "55 English onsets and their CELEX type frequencies in ARPABET format from Hayes & Wilson (2008). A subset of the onsets in the CMU Pronouncing Dictionary.",
            "finnish.csv": "From a word list generated by the Institute for the Languages of Finland (http://kaino.kotus.fi/sanat/nykysuomi/). Represented orthographically. See Mayer (2020) for details.",
            "french.csv": "French corpus used in Goldsmith & Xanthos (2009) and Mayer (2020). Represented in IPA.",
            "polish_onsets.csv": "Polish onsets with type frequencies from Jarosz (2017). Generated from a corpus of child-directed speech consisting of about 43,000 word types (Haman et al. 2011). Represented orthographically.",
            "samoan.csv": "Samoan word list from Milner (1993), compiled by Kie Zuraw. Represented in IPA.",
            "spanish_stress.csv": "A set of about 24,000 word types including inflected forms from the EsPal database (Duchon et al. 2013) in IPA with stress encoded. Frequencies from a large collection of Spanish subtitle data.",
            "turkish.csv": "A set of about 18,000 citation forms from the Turkish Electronic Living Lexicon database (TELL; Inkelas et al. 2000) in IPA.",
        }
        # Short labels for dropdown (user-friendly)
        short_labels = {
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
        # Use short label for display, filename as value
        return [(f, short_labels.get(f, f)) for f in files]
    except (FileNotFoundError, OSError):
        # Fallback if directory can't be read
        return []


class DefaultFile(models.Model):
    """Default training file model."""

    training_file = models.FileField(upload_to="default")
    description = models.CharField(max_length=200)
    short_desc = models.CharField(max_length=50, default="")

    @property
    def file_name(self) -> str:
        """Get the file name without the 'default/' prefix."""
        cur_name = cast(str, self.training_file.name)
        return cur_name.replace("default/", "")


class UploadTrain(models.Model):
    """Model for uploaded training and test files."""



    # upload files go to media\uploads
    training_file = models.FileField(upload_to="uploads", blank=True)
    default_training_file = models.CharField(
        choices=get_file_choices, max_length=200, blank=True
    )
    test_file = models.FileField(upload_to="uploads")
    training_model = models.CharField(default="simple", max_length=128)
    current_time = models.DateTimeField(default=timezone.now)

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Save the model."""
        super().save(*args, **kwargs)


class UploadWithDefault(models.Model):
    """Model for uploads using a default training file."""

    default_objects: List["DefaultFile"] = []  # DefaultFile.objects.all()
    files_list = [(x.file_name, x.file_name) for x in default_objects]
    training_file = models.CharField(choices=files_list, max_length=200)
    test_file = models.FileField(upload_to="uploads")
    training_model = models.CharField(default="simple", max_length=128)
    timeStr = datetime.now().strftime("%m_%d_%H%M")
    out_file = "outfile_" + timeStr + ".csv"
