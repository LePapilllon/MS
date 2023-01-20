# -*- coding:utf-8 -*-

import  shutil
import os

oldpath = r''          # 原数据路径
newpath = r''       # 移动到新文件夹的路径
file_path = r'train.txt'       # txt中指定移动文件的文件信息

# 示例
# oldpath = r'F:\ML\LH_KDSegmentation_Version3\LH_KDSegmentation_Version3\data\Images'          # 原数据路径
# newpath = r'F:\moco-main\Images\train\mytrain'       # 移动到新文件夹的路径 即本目录下的Images\train\mytrain 以及 Images\val\myval
# file_path = r'F:\moco-main\train.txt'       # txt中指定移动文件的文件信息


#从文件中获取要拷贝的文件的信息
def get_filename_from_txt(file):
    filename_lists = []
    with open(file,'r',encoding='utf-8') as f:
        lists =  f.readlines()
        for list in lists:
            filename=str(list).strip('\n')
            name="myimg_"+filename[8:-1]+".tif"
            filename_lists.append(name)
    return filename_lists

#拷贝文件到新的文件夹中
def mycopy(srcpath,dstpath,filename):
    if not os.path.exists(srcpath):
        print("srcpath not exist!")
    if not os.path.exists(dstpath):
        print("dstpath not exist!")
    for root,dirs,files in os.walk(srcpath,True):

        if filename in files:
            # 如果存在就拷贝
            shutil.copy(os.path.join(root,filename),dstpath)
        else:
            # 不存在的话将文件信息打印出来
            print(filename)

if __name__ == "__main__":
    #执行获取文件信息的程序
    filename_lists = get_filename_from_txt(file_path)
    #根据获取的信息进行遍历输出
    for filename in filename_lists:
        mycopy(oldpath,newpath,filename)
