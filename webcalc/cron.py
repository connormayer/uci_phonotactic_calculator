from distutils.command.upload import upload
from webcalc_project import settings
from os import listdir, unlink
from os.path import join, isfile, getmtime, isdir

import shutil
import time

def clean_media_folder():
    uploads_folder = join(settings.MEDIA_ROOT, 'uploads')
    cur_time = int(time.time())
    # 10 minutes time limit --> 10*60 seconds
    limit = 10 * 60 
    for filename in listdir(uploads_folder):
        path = join(uploads_folder, filename)
        last_mod_time = getmtime(path)
        if cur_time - last_mod_time > limit:
            try:
                if isfile(path):
                    unlink(path)
                elif isdir(path):
                    shutil.rmtree(path)

            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (path, e))
