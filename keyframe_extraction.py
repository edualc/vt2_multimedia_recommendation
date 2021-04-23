import os
import pandas as pd
import numpy as np
from datetime import datetime
import skvideo.io
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import time

from util.paths import ensure_dir


BASE_FOLDER_PATH = '/mnt/all1/ml20m_yt'
SECONDS_PER_KEY_FRAME = 2
FRAMES_PER_SECOND = 24 # 23.97-ish
START_FRAME = FRAMES_PER_SECOND // 2


df = pd.read_csv('datasets/ml20m_youtube/youtube_extracted.csv', index_col=0)

def read_video(path):
    try:
        video = skvideo.io.vread(path)
    
    # lehl@2021-04-23: Due to memory issues, scikit-video might return errors stating
    # that it is unable to find the width and height of the video. This might be due
    # to the machine running out of memory, such that ffmpeg/ffprobe cannot run.
    # 
    except Exception as e:
        return e
    else:
        return video

def write_image(path, data):
    try:
        skvideo.io.vwrite(path, data)

    # lehl@2021-04-16: Sometimes, the machine cannot allocate
    # enough memory to perform this action.
    # 
    except OSError:
        pass
    
    # lehl@20201-04-20: Sometimes, the machine is unable to get
    # the dimensions of the movie file:
    # 
    # - ValueError: No way to determine width or height from video.
    #   Need `-s` in `inputdict`. Consult documentation on I/O.
    # 
    # lehl@2021-04-23: Due to memory issues, scikit-video might return errors stating
    # that it is unable to find the width and height of the video. This might be due
    # to the machine running out of memory, such that ffmpeg/ffprobe cannot run.
    # 
    except ValueError:
        pass
    
    # lehl@2021-04-20: Catch everything else to ensure it runs
    # with as many movie files as possible
    # 
    except Exception:
        pass

# Entropy calculation of an image, taken from 
# https://stackoverflow.com/questions/50313114/what-is-the-entropy-of-an-image-and-how-is-it-calculated
# 
def np_entropy(img):
    marg = np.histogramdd(np.ravel(img), bins = 256)[0]/img.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    return -np.sum(np.multiply(marg, np.log2(marg)))
    
def plot_entropy(entropy, movielens_id, youtube_id):
    entropy_diff = np.ediff1d(entropy)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,10))

    ax1.plot(entropy)
    ax1.legend(['Entropy'])

    ax2.plot(entropy_diff)
    ymin, ymax = ax2.get_ylim()
    ax2.legend(['Entropy Difference'])
    
    plt.savefig('entropy_' + str(movielens_id) + '_' + str(youtube_id) + '.png')

count = 0

for subdir, dirs, files in os.walk(BASE_FOLDER_PATH + '/videos_resized'):
    movielens_id = subdir.split('/')[-1]

    for file in files:
        youtube_id = file.split('.')[0]
        file_path = os.path.join(subdir, file)

        # initialize diskwriter to save data at desired location
        target_path = os.path.join(BASE_FOLDER_PATH, 'videos_resized', movielens_id)
        target_file_path = target_path + '/' + file

        num_keyframes = int(df[df.index == int(movielens_id)]['duration'] / SECONDS_PER_KEY_FRAME)
        
        count += 1

        # Check if there are already at least :num_keyframes images in the folder
        # and abort if that is the case (already generated the keyframes)
        # 
        # If not, create keyframes for this trailer
        # 
        target_directory = f"{BASE_FOLDER_PATH}/keyframes_224/{movielens_id}"
        
        if os.path.isdir(target_directory):
            num_files_in_dir = len([name for name in os.listdir(target_directory) if os.path.isfile(f"{target_directory}/{name}")])
    
            if num_files_in_dir >= 0.8 * num_keyframes:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\t\t[{count}] SKIP:   {num_files_in_dir} keyframes found for {movielens_id} at {target_directory}")
                continue

        # =====================
        # Read video
        # =========================================================
        # 
        video = read_video(target_file_path)

        # import code; code.interact(local=dict(globals(), **locals()))

        if type(video) == np.ndarray:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\t\t[{count}] CREATE: Extracting {num_keyframes} keyframes for {movielens_id} at {target_directory}")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\t\t[{count}] EXCEPTION: Unable to extract keyframes for {movielens_id} at {target_directory}")
            print(f"\t[{type(video)}] --> {str(video)}")
            continue

        print(f"\t{video.shape}")

        # mean_image = np.mean(video, axis=0)
        # residual_video = video - mean_image
        # images = residual_video[START_FRAME::FRAMES_PER_SECOND, :, :, :]

        raw_images = video[START_FRAME::FRAMES_PER_SECOND, :, :, :]
        
        # # =====================
        # # Calculate and plot entropy
        # # =========================================================
        # # 
        # entropy = np.zeros(shape=images.shape[0])

        # for i in range(images.shape[0]):
        #     image = images[i, :, :, :]
        #     entropy[i] = np_entropy(image)

        # plot_entropy(entropy, movielens_id, youtube_id)

        # =====================
        # Extract keyframes
        # =========================================================
        #
        ensure_dir(f"{BASE_FOLDER_PATH}/keyframes_224/{movielens_id}/")

        # Individual keyframes as png
        #
        for i in range(raw_images.shape[0]):
            image = raw_images[i, :, :, :]
            image_name = f"{BASE_FOLDER_PATH}/keyframes_224/{movielens_id}/{youtube_id}_{i:05d}.png"

            write_image(image_name, image)
            del image

        # # Full keyframes as numpy array
        # # 
        # with open(f"{BASE_FOLDER_PATH}/keyframes_224/{movielens_id}/224x224x3.npy", 'wb') as f:
        #     np.save(f, raw_images)

        # del images
        del raw_images

