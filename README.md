# Lung_nodule_detection_with_LNDb
powered by google colab
We are going to learn to use 3D raw CT data mhd labels to do the training
## First, we have to import needed pachages and define some commonly used functions 
```
import csv
import cv2
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
from scipy.ndimage import zoom
import scipy.ndimage as ndi
import imageio
import glob
%matplotlib inline
import math
import skimage
from skimage.measure import label,regionprops, perimeter
from skimage import measure, feature
import scipy.misc
from matplotlib import patches
from collections import defaultdict
from sklearn.model_selection import train_test_split
from math import log
from skimage.filters import median
import matplotlib
### Basic functions for reading raw files
def readCsv(csvfname):
     # read csv to list of lists
     with open(csvfname, 'r') as csvf:
         reader = csv.reader(csvf)
         csvlines = list(reader)
     return csvlines
def readMhd(filename):
     # read mhd/raw image
     itkimage = sitk.ReadImage(filename)
     scan = sitk.GetArrayFromImage(itkimage) #3D image
     spacing = itkimage.GetSpacing() #voxelsize
     origin = itkimage.GetOrigin() #world coordinates of origin
     transfmat = itkimage.GetDirection() #3D rotation matrix
     return scan, spacing, origin, transfmat
def getImgWorldTransfMats(spacing,transfmat):
     # calc image to world to image transformation matrixes
     transfmat = np.array([transfmat[0:3],transfmat[3:6],transfmat[6:9]])
     for d in range(3):
         transfmat[0:3,d] = transfmat[0:3,d]*spacing[d]
     transfmat_toworld = transfmat #image to world coordinates conversion matrix
     transfmat_toimg = np.linalg.inv(transfmat) #world to image coordinates conversion matrix
     return transfmat_toimg,transfmat_toworld
def convertToImgCoord(xyz,origin,transfmat_toimg):
     # convert world to image coordinates
     xyz = xyz - origin
     xyz = np.round(np.matmul(transfmat_toimg,xyz))
     return xyz
### Confirmed nodules by at least 2 radiologists
confirmed = pd.read_csv('/content/gdrive/MyDrive/training_data_lung_nodule/trainNodule_with_filename.csv', index_col='LNDbID')
print(confirmed.shape)
confirmed.index
```
## Datasets used for loading the transformed 2D CT slices with nodule
```
os.makedirs("/content/train_jpg/zero_center", exist_ok=True)
os.makedirs("/content/train_jpg/normalize_log", exist_ok=True)
os.makedirs("/content/train_jpg/median", exist_ok=True)
os.makedirs("/content/train_jpg/gray_conv", exist_ok=True)
os.makedirs("/content/labels", exist_ok=True)
```
## Start converting .RAW to .jpg
```
!python zero_center.py
#!python 
```
