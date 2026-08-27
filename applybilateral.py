#%%
import os
import cv2
import numpy as np
from cv2 import bilateralFilter as bifil

outputpath = "Generated_Samples/Generated_Samples_LinearTrainSigmoidSample(a=8)L10.01FidelityLoss/Images"

d = 29
sigC = 40
sigS = 20

for imgname in os.listdir(outputpath):
    
    imgdir = os.path.join(outputpath, imgname)
    imgdir = os.path.join(imgdir, '0'+imgname+'.png')
    img = cv2.imread(imgdir)
    img_filt = bifil(img, d = d, sigmaColor = sigC, sigmaSpace = sigS)

    os.makedirs(f"Generated_Samples/Bilateral_Filtered_{d}_{sigC}_{sigS}", exist_ok = True)
    os.makedirs(f"Generated_Samples/Bilateral_Filtered_{d}_{sigC}_{sigS}/Images", exist_ok = True)
    os.makedirs(f"Generated_Samples/Bilateral_Filtered_{d}_{sigC}_{sigS}/Images/"+imgname, exist_ok = True)

    cv2.imwrite(os.path.join(f"Generated_Samples/Bilateral_Filtered_{d}_{sigC}_{sigS}/Images/"+imgname, '0' + imgname+'.png'), img_filt)

    print("Saved ", imgname)
# %%
