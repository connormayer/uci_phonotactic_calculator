"""Cron jobs for the webcalc Django app."""

import os
import shutil
from datetime import datetime, timedelta

from django.conf import settings


def clean_media_folder() -> None:
    """
    Clean up the media/uploads folder by removing files older than 24 hours.
    This is run as a cron job to prevent the media folder from filling up.
    """
    uploads_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    if not os.path.exists(uploads_dir):
        return

    now = datetime.now()
    cutoff = now - timedelta(hours=24)

    for filename in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir, filename)
        if os.path.isfile(file_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_mod_time < cutoff:
                os.remove(file_path)
        elif os.path.isdir(file_path):
            dir_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if dir_mod_time < cutoff:
                shutil.rmtree(file_path)
