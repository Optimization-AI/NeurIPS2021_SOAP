__author__ = 'Qi'
# Created by on 8/1/21.


import pandas as pd
import os
import shutil

train_dat = pd.read_csv("./data/melanoma_split_inds/train_split.csv")
test_dat = pd.read_csv("./data/melanoma_split_inds/test_split.csv")
valid_dat = pd.read_csv("./data/melanoma_split_inds/valid_split.csv")

print(test_dat.shape[0] + valid_dat.shape[0])


j = 0
for img in os.listdir('/data/qiuzh/melanoma/jpeg/mytrain/0'):
    for i in range(test_dat.shape[0]):
        if test_dat['image_name'][i] in img:
            j+=1
            shutil.move(os.path.join("/data/qiuzh/melanoma/jpeg/mytrain/0", img), os.path.join("/data/qiuzh/melanoma/jpeg/mytest/0", img))
    for i in range(valid_dat.shape[0]):
        if valid_dat['image_name'][i] in img:
            j+=1
            shutil.move(os.path.join("/data/qiuzh/melanoma/jpeg/mytrain/0", img), os.path.join("/data/qiuzh/melanoma/jpeg/myval/0", img))
print("Moved Nagative Samples:", j)

j=0
for img in os.listdir('/data/qiuzh/melanoma/jpeg/mytrain/1'):
    for i in range(test_dat.shape[0]):
        if test_dat['image_name'][i] in img:
            j += 1
            shutil.move(os.path.join("/data/qiuzh/melanoma/jpeg/mytrain/1", img), os.path.join("/data/qiuzh/melanoma/jpeg/mytest/1", img))
    for i in range(valid_dat.shape[0]):
        if valid_dat['image_name'][i] in img:
            j += 1
            shutil.move(os.path.join("/data/qiuzh/melanoma/jpeg/mytrain/1", img), os.path.join("/data/qiuzh/melanoma/jpeg/myval/1", img))
print("Moved Positive Samples:", j)

