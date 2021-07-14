import os

# for split_num in range(7,11):
#     os.system("/home/claude/development/libfm/bin/libFM --dim '1,1,32' --init_stdev 0.1 --iter 100 --task r --verbosity 1 --train /mnt/all1/ml20m_yt/ml20m/10fold_cv/ml20m_train_" + str(split_num) + ".dat.libfm --test /mnt/all1/ml20m_yt/ml20m/10fold_cv/ml20m_test_" + str(split_num) + ".dat.libfm --out /mnt/all1/ml20m_yt/ml20m/10fold_cv/libfm_preds/libfm_" + str(split_num) + ".pred --rlog /mnt/all1/ml20m_yt/ml20m/10fold_cv/libfm_preds/libfm_" + str(split_num) + ".log")

split_num = 1
os.system("/home/claude/development/libfm/bin/libFM --dim '1,1,32' --init_stdev 0.1 --iter 100 --task r --verbosity 1 --train /mnt/all1/ml20m_yt/ml20m/10fold_cv/ml20m_train_" + str(split_num) + ".dat.libfm --test /mnt/all1/ml20m_yt/ml20m/10fold_cv/ml20m_test_" + str(split_num) + ".dat.libfm --out /mnt/all1/ml20m_yt/ml20m/10fold_cv/libfm_preds/libfm_" + str(split_num) + "-2.pred --rlog /mnt/all1/ml20m_yt/ml20m/10fold_cv/libfm_preds/libfm_" + str(split_num) + "-2.log")

