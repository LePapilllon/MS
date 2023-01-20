import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import cv2
from osgeo import gdal

class VOC_SEG_Dataset(Dataset):
    def __init__(self, root, width, height, transforms=None):
        # 图像统一剪切尺寸（width, height）
        self.width = width
        self.height = height

        self.fnum = 0
        if transforms is None:
            # normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.ToTensor(),
                # normalize
            ])

        txt_fname = root + "/ImageSets/Segmentation/val.txt"
        with open(txt_fname, 'r') as f:
            images = f.read().split()
        imgs = [os.path.join(root, "Images/myimg" + item[7:] + ".tif") for item in images]
        labels = [os.path.join(root, "SegmentationClass", item + ".tif") for item in images]
        self.imgs = self._filter(imgs)
        self.labels = self._filter(labels)
        print("Test：loaded " + str(len(self.imgs)) + " images and labels" + ",filtered " + str(self.fnum) + " images")

    def _crop(self, data, label):
        """
        切割函数，默认都是从图片的左上角开始切割。切割后的图片宽是width,高是height
        data和label都是Image对象
        """
        box = (0, 0, self.width, self.height)
        data = data.crop(box)
        label = label.crop(box)
        return data, label

    def _image_transforms(self, data, label):
        data, label = self._crop(data, label)
        data = self.transforms(data)
        # label = self._image2label(label)
        # label = self.transforms(label)
        label = np.array(label, dtype="int64")
        label = torch.from_numpy(label)
        return data, label

    def _filter(self, imgs):
        img = []
        print("test images or labels are loading... please wait")
        for im in imgs:
            if (cv2.imread(im).shape[1] >= self.height and
                    cv2.imread(im).shape[0] >= self.width):
                img.append(im)
            else:
                self.fnum = self.fnum + 1
        return img

    def __getitem__(self, index: int):
        img_path = self.imgs[index]
        label_path = self.labels[index]
        # img = cv2.imread(img_path)

        img = gdal.Open(img_path)
        img_width = img.RasterXSize  # 栅格矩阵的列数
        img_height = img.RasterYSize  # 栅格矩阵的行数
        img = img.ReadAsArray(0, 0, img_width, img_height)
        img = np.transpose(np.array(img), (1, 2, 0))
        img = Image.fromarray(np.uint8(img))

        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label = gdal.Open(label_path)
        label_width = label.RasterXSize  # 栅格矩阵的列数
        label_height = label.RasterYSize  # 栅格矩阵的行数
        label = label.ReadAsArray(0, 0, label_width, label_height)
        label = Image.fromarray(np.uint8(label))

        img, label = self._image_transforms(img, label)
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    root = "data"
    width = 512
    height = 512
    voc_test = VOC_SEG_Dataset(root, width, height)

    for data, label in voc_test:
        print(data.shape)
        print(label.shape)
        break


