import os

def ensure_dir(file_path):
    try:
        os.makedirs(file_path)
    except FileExistsError:
        pass
