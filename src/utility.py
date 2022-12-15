# src\utility.py

from distutils.command.upload import upload
from tempfile import TemporaryFile
from webcalc_project import settings
from os import listdir, unlink
from os.path import join, isfile, getmtime

import time

def valid_file(file_path):
    with open(file_path, 'r') as f:
        tokens = f.read()

    tokens = [s.split(',') for s in tokens.split("\n") if s]
    tokens_no_freq = [token[0].split(' ') for token in tokens]

    if any(map(lambda x: '\t' in x, tokens)):
        return (False, 'Files must be comma delimited.')
    
    if all(map(lambda x: len(x) == 1, tokens_no_freq)):
        return (False, 'Phonemes must be separated by spaces.')

    return (True, "")

def clean_media_folder():
    uploads_folder = join(settings.MEDIA_ROOT, 'uploads')
    cur_time = int(time.time())
    limit = 10*60 # 10 minutes time limit --> 10*60 seconds
    for filename in listdir(uploads_folder):
        path = join(uploads_folder, filename)
        last_mod_time = getmtime(path)
        try:
            if isfile(path) and cur_time - last_mod_time > limit:
                unlink(path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (path, e))
