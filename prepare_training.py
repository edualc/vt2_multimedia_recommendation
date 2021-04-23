import os
import cv2
import numpy as np
from util.paths import ensure_dir
import shutil

BASE_FOLDER_PATH = '/mnt/all1/ml20m_yt/training_224'

TEST_RATIO = 0.2

count = 0
for subdir, dirs, files in os.walk(BASE_FOLDER_PATH + '/train'):
    num_files = len(files)
    movielens_id = subdir.split('/')[-1]
    
    num_test = int(TEST_RATIO * num_files) + 1
    num_train = num_files - num_test

    if num_train == 0:
        continue

    print(f"Copying {num_test}/{num_files} files, leaving a {num_train}-{num_test} split for {movielens_id}...")

    if num_files > 0:
        # Check if already exists in test
        # 
        target_test_dir = BASE_FOLDER_PATH + '/test/' + str(movielens_id) + '/'

        if not os.path.exists(target_test_dir):
            ensure_dir(target_test_dir)

        for test_subdir, test_dirs, test_files in os.walk(target_test_dir):
            num_test_files = len(test_files)

            if num_test_files > 0:
                # Already has a few files in here
                # TODO?
                #  
                pass

            else:
                # No files in here -> copy
                # 
                
                files_to_copy = np.random.choice(files, num_test, replace=False)

                for test_file in files_to_copy:

                    source_path = BASE_FOLDER_PATH + '/train/' + str(movielens_id) + '/' + str(test_file)
                    target_path = BASE_FOLDER_PATH + '/test/' + str(movielens_id) + '/' + str(test_file)

                    shutil.copyfile(source_path, target_path)

    count += 1
