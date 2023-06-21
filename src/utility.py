# src\utility.py

from distutils.command.upload import upload
from tempfile import TemporaryFile
from webcalc_project import settings
from os import listdir, unlink
from os.path import join, isfile, getmtime
from collections import namedtuple

import time

Dataset = namedtuple("Dataset", "file long_desc short_desc frequency audience language")

def valid_file(file_path):
    with open(file_path, encoding='utf-8', mode='r') as f:
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

# TODO: CREATE CLASS FOR DEFAULT FILES AND READ FROM CSV INTO OBJECTS OF THIS CLASS
#   - THIS ACTS AS A REPLACEMENT FOR THE DEFAULTFILE MODEL WHICH NEEDS TO BE REMOVED

def get_default_files():
    file_name = join(settings.MEDIA_ROOT, 'default_file_list.txt')
    # TODO: USE CSV FILE INSTEAD OF TXT FOR STORING DEFAULT FILES AND READ FROM CSV
    with open(file_name) as file:
        # each sublist holds info for Dataset namedtuple
        lines_info = [[x.strip() for x in line.split('~')] for line in file]
        return [Dataset(*x) for x in lines_info]
        # return [(x[0], x[2]) for x in lines_info]

# TODO: METHOD TO GROUP DEFAULT FILES
def group_datasets(datasets, group_method):
    return sorted(datasets, key = (lambda x : getattr(x, group_method)))