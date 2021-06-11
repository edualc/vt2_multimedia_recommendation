import os

for split_num in range(1,11):
    os.system("/home/claude/development/libfm/bin/libFM --dim '1,1,32' --init_stdev 0.1 --iter 100 --task r --verbosity 1 --train /mnt/all1/ml20m_yt/ml20m/crossvalidation/train_" + str(split_num) + ".dat.libfm --test /mnt/all1/ml20m_yt/ml20m/crossvalidation/test_" + str(split_num) + ".dat.libfm --out /mnt/all1/ml20m_yt/ml20m/crossvalidation/libfm_experiment_" + str(split_num) + ".pred --rlog /mnt/all1/ml20m_yt/ml20m/crossvalidation/libfm_experiment_" + str(split_num) + ".log")
