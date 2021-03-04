import numpy as np
import os

# img_h, img_w = 32, 32
img_h, img_w = 384, 384  # 根据自己数据集适当调整，影响不大
frames = 49
img_list = []

imgs_path = './data/data_trans'
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    files = os.listdir(os.path.join(imgs_path, item))
    for file in files:
        if '_IR069' in file:
            img = np.load(os.path.join(imgs_path, item, file))
            # print(img.shape)
            # break
            img = np.resize(img, (img_w, img_h, frames))
            img = img[:, :, :, np.newaxis]
            img_list.append(img)
            i += 1
            print(i, '/', len_)
        else:
            pass

imgs = np.concatenate(img_list, axis= 3)
print(imgs.shape)
pixels = imgs.ravel()  # 拉成一行
means = np.mean(pixels)
stdevs = np.std(pixels)



print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))