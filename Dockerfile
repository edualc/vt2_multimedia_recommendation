# Based upon the AutoDL Dockerfile here:
# - https://github.com/zhengying-liu/autodl/blob/master/docker/Dockerfile
#
# GPULogin CUDA v11.1
#
FROM tensorflow/tensorflow:latest-gpu-py3
# tensorflow==1.13.1

RUN pip install numpy==1.16.2
RUN pip install pandas==0.24.2
RUN pip install jupyter==1.0.0
RUN pip install seaborn==0.9.0
RUN pip install scipy==1.2.1
RUN pip install matplotlib==3.0.3
RUN pip install scikit-learn==0.20.3
RUN pip install pyyaml==5.1.1
RUN pip install psutil==5.6.3
RUN pip install h5py==2.9.0
RUN pip install keras==2.2.4
RUN pip install playsound==1.2.2
RUN pip install librosa==0.7.1
RUN pip install protobuf==3.7.1
RUN pip install xgboost==0.90
RUN pip install pyyaml==5.1.1
RUN pip install lightgbm==2.2.3
# RUN pip install torch==1.3.1
# RUN pip install torchvision==0.4.2
RUN pip install tqdm==4.60.0
RUN pip install python-decouple==3.3
RUN pip install google-api-python-client==1.12.8
# RUN pip install python3-bs4==4.9.3
RUN pip install youtube_dl==2021.4.7

# Packages from AutoNLP
# More info: https://hub.docker.com/r/wahaha909/autonlp
# RUN pip install nltk==3.4.4
# RUN pip install spacy==2.1.6
# RUN pip install gensim==3.8.0
# RUN pip install polyglot==16.7.4
RUN pip install transformers==2.2.0



# TODO:
# - wandb
# - opencv / cv2
# - skimage
# - scikit-video
# - tensorflow-datasets
# - tensorflow (current version)